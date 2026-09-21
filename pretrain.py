import math
import time
from typing import Any

import jax
import jax.numpy as jnp
import optax

from configs import Config, load_config
from data.pretraining import PackedPretrainingDataset
from generation.generator import JaxGenerator
from models.config import ModelConfig
from models.language_model import LanguageModel
from training.checkpoint import CheckpointManager
from training.experiment import build_experiment_metadata, write_experiment_metadata
from training.metrics import MetricsLogger
from training.optimizer import make_muon_adamw_optimizer
from training.pytree import global_norm, tree_add, tree_scale
from training.schedules import warmup_cosine_schedule
from utils.model_stats import print_model_stats


def cross_entropy_loss(logits: jax.Array, targets: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]

    return -jnp.mean(target_log_probs)


def build_validation_batches(cfg: Config) -> list[tuple[jax.Array, jax.Array]]:
    """
    Materialize a small fixed validation set in RAM
    """
    dataset = PackedPretrainingDataset(
        tokenizer_name=cfg.tokenizer_name,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=cfg.val_split,
        shuffle=False,
        shuffle_buffer=cfg.shuffle_buffer,
        seed=cfg.seed,
    )

    batches = []

    print(f"loading {cfg.val_batches} validation batches...")
    for _ in range(cfg.val_batches):
        x, y = dataset.next_batch()

        batches.append((jnp.asarray(x), jnp.asarray(y)))

    validation_tokens = cfg.val_batches * cfg.batch_size * cfg.seq_len

    print(f"validation tokens: {validation_tokens:,}")

    return batches


def sample_model(generator: JaxGenerator, params: Any, tokenizer: Any, cfg: Config, step: int):
    print()
    print("=" * 10)
    print(f"samples at step {step}")

    print("=" * 10)
    for i, prompt in enumerate(cfg.sample_prompts):
        text = generator.generate(
            params=params,
            prompt=prompt,
            max_new_tokens=cfg.sample_max_new_tokens,
            seed=cfg.seed + step + i,
        )
        print()
        print(f"[Prompt {i + 1}]")
        print(text)

    print()
    print("=" * 10)
    print()


def main():
    cfg = load_config("configs/pretrain.yaml")

    model_cfg = ModelConfig(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        n_experts=cfg.n_experts,
        top_k=cfg.top_k,
        capacity_factor=cfg.capacity_factor,
        router_aux_loss_coef=cfg.router_aux_loss_coef,
        router_z_loss_coef=cfg.router_z_loss_coef,
        n_loops=cfg.n_loops,
        n_core_blocks=cfg.n_core_blocks,
        loop_residual_scale=cfg.loop_residual_scale,
        depth_embedding_scale=cfg.depth_embedding_scale,
    )

    model = LanguageModel(model_cfg)
    key = jax.random.PRNGKey(cfg.seed)

    dummy = jnp.zeros((cfg.batch_size, cfg.seq_len), dtype=jnp.int32)

    params = model.init(key, dummy)["params"]
    _ = print_model_stats(params=params, n_experts=cfg.n_experts, top_k=cfg.top_k, n_loops=cfg.n_loops)

    lr_schedule = warmup_cosine_schedule(
        peak_lr=cfg.learning_rate,
        total_steps=cfg.steps,
        warmup_steps=cfg.warmup_steps,
        min_lr=cfg.min_learning_rate,
    )
    optimizer = make_muon_adamw_optimizer(
        params,
        lr_schedule,
        weight_decay=cfg.weight_decay,
        muon_momentum=cfg.muon_momentum,
        muon_ns_steps=cfg.muon_ns_steps,
    )
    opt_state = optimizer.init(params)

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(directory=cfg.checkpoint_dir, keep=cfg.keep_checkpoints)
    metrics_logger = MetricsLogger(
        directory=cfg.checkpoint_dir,
        append=cfg.resume and checkpoint_manager.latest_step() is not None,
    )
    start_step = 0
    tokens_seen = 0
    best_val_loss = math.inf

    train_dataset = PackedPretrainingDataset(
        tokenizer_name=cfg.tokenizer_name,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split="train",
        shuffle=True,
        shuffle_buffer=cfg.shuffle_buffer,
        seed=cfg.seed,
    )
    tokenizer = train_dataset.tokenizer
    actual_vocab_size = len(train_dataset.tokenizer)
    if actual_vocab_size != cfg.vocab_size:
        raise ValueError(f"Configured vocab_size={cfg.vocab_size} but tokenizer has {actual_vocab_size} tokens.")

    if cfg.resume:
        restored = checkpoint_manager.restore_latest(params, opt_state)

        if restored is not None:
            params = restored["params"]
            opt_state = restored["opt_state"]
            start_step = int(restored["step"]) + 1
            tokens_seen = int(restored["tokens_seen"])
            best_val_loss = float(restored["best_val_loss"])
            train_dataset.load_state_dict(restored["data_state"])

            print(f"resume step: {start_step}")
            print(f"tokens seen: {tokens_seen:,}")
            print(f"best val loss: {best_val_loss:.4f}")

    @jax.jit
    def compute_grads(params: Any, x: jax.Array, y: jax.Array):
        def loss_fn(params):
            logits, stats = model.apply({"params": params}, x)

            lm_loss = cross_entropy_loss(logits, y)

            total_loss = (
                lm_loss
                + cfg.router_aux_loss_coef * stats["router_aux_loss"]  # type: ignore
                + cfg.router_z_loss_coef * stats["router_z_loss"]  # type: ignore
            )

            return total_loss, {"lm_loss": lm_loss, **stats}

        (loss, stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        return loss, stats, grads

    @jax.jit
    def apply_grads(params: Any, opt_state: Any, grads: Any):
        grad_norm = global_norm(grads)
        clip_scale = jnp.minimum(1.0, cfg.max_grad_norm / (grad_norm + 1e-6))

        grads = jax.tree_util.tree_map(lambda g: g * clip_scale, grads)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)

        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, grad_norm, clip_scale

    # validation step
    @jax.jit
    def validation_step(params: Any, x: jax.Array, y: jax.Array):
        logits, stats = model.apply({"params": params}, x)
        lm_loss = cross_entropy_loss(logits, y)

        return lm_loss, stats

    validation_batches = build_validation_batches(cfg)

    def evaluate(params: Any) -> dict:
        losses = []

        router_aux = []
        router_z = []

        dropped = []
        entropy = []

        for x, y in validation_batches:
            loss, stats = validation_step(params, x, y)

            losses.append(loss)
            router_aux.append(stats["router_aux_loss"])
            router_z.append(stats["router_z_loss"])
            dropped.append(jnp.mean(stats["dropped_fraction"]))
            entropy.append(jnp.mean(stats["router_entropy"]))

        # Synchronize once
        losses = jnp.stack(losses)
        losses.block_until_ready()
        val_loss = float(jnp.mean(losses))

        # Perplexity
        display_loss = min(val_loss, 20.0)
        perplexity = math.exp(display_loss)
        metrics = {
            "loss": val_loss,
            "perplexity": perplexity,
            "router_aux": float(jnp.mean(jnp.stack(router_aux))),
            "router_z": float(jnp.mean(jnp.stack(router_z))),
            "dropped_fraction": float(jnp.mean(jnp.stack(dropped))),
            "router_entropy": float(jnp.mean(jnp.stack(entropy))),
        }

        return metrics

    tokens_per_step = cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps

    experiment_metadata = build_experiment_metadata(cfg=cfg, vocab_size=actual_vocab_size)
    write_experiment_metadata(directory=cfg.checkpoint_dir, metadata=experiment_metadata)

    # Generator
    generator = JaxGenerator(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
        temperature=cfg.sample_temperature,
        top_k=cfg.sample_top_k,
    )

    print("compiling generation path...")
    generator.warmup(params)

    print()
    print("initial validation...")
    val = evaluate(params)
    initial_validation_step = start_step - 1
    metrics_logger.log("validation", initial_validation_step, tokens_seen, val)
    last_validation_step = initial_validation_step
    print(
        f"val_loss={val['loss']:.4f} ppl={val['perplexity']:.2f} drop={val['dropped_fraction']:.4f} Rent={val['router_entropy']:.3f}"
    )
    print()

    for step in range(start_step, cfg.steps):
        step_start = time.perf_counter()

        accumulated_grads = None

        micro_losses = []
        micro_lm_losses = []
        micro_router_aux = []
        micro_router_z = []
        micro_drops = []

        for micro_step in range(cfg.grad_accum_steps):
            x, y = train_dataset.next_batch()

            loss, stats, grads = compute_grads(params, jnp.asarray(x), jnp.asarray(y))

            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_add(accumulated_grads, grads)

            micro_losses.append(loss)
            micro_lm_losses.append(stats["lm_loss"])
            micro_router_aux.append(stats["router_aux_loss"])
            micro_router_z.append(stats["router_z_loss"])
            micro_drops.append(jnp.mean(stats["dropped_fraction"]))

        # Average
        accumulated_grads = tree_scale(accumulated_grads, 1.0 / cfg.grad_accum_steps)
        params, opt_state, grad_norm, clip_scale = apply_grads(params, opt_state, accumulated_grads)

        grad_norm.block_until_ready()

        mean_loss = jnp.mean(jnp.stack(micro_losses))
        mean_lm_loss = jnp.mean(jnp.stack(micro_lm_losses))
        mean_router_aux = jnp.mean(jnp.stack(micro_router_aux))
        mean_router_z = jnp.mean(jnp.stack(micro_router_z))
        mean_drop = jnp.mean(jnp.stack(micro_drops))

        mean_loss.block_until_ready()

        elapsed = time.perf_counter() - step_start
        tokens_seen += tokens_per_step
        tokens_per_second = tokens_per_step / elapsed

        if step % 50 == 0:
            current_lr = float(jnp.asarray(lr_schedule(step)))

            metrics_logger.log(
                "train",
                step,
                tokens_seen,
                {
                    "loss": mean_loss,
                    "lm_loss": mean_lm_loss,
                    "learning_rate": current_lr,
                    "router_aux_loss": mean_router_aux,
                    "router_z_loss": mean_router_z,
                    "grad_norm": grad_norm,
                    "clip_scale": clip_scale,
                    "dropped_fraction": mean_drop,
                    "tokens_per_second": tokens_per_second,
                },
            )

            print(
                f"step={step:04d} tokens={tokens_seen} loss={float(mean_loss):.4f} lm={float(mean_lm_loss):.4f} lr={current_lr:.4e} aux={float(mean_router_aux):.4f} z={float(mean_router_z):.4f} gnorm={float(grad_norm):.3f} clip={float(clip_scale):.3f} drop={mean_drop:.4f} tok/s={tokens_per_second:,.0f}"
            )

        # validation
        should_eval = (step + 1) % cfg.eval_every == 0
        if should_eval:
            print()
            print(f"validation at step {step}...")
            val = evaluate(params)

            improved = val["loss"] < best_val_loss

            if improved:
                best_val_loss = val["loss"]

            metrics_logger.log("validation", step, tokens_seen, val)
            last_validation_step = step

            marker = " BEST" if improved else ""

            print(
                f"val_loss={val['loss']:.4f} ppl={val['perplexity']:.2f} aux={val['router_aux']:.4f} z={val['router_z']:.4f} drop={val['dropped_fraction']:.4f} Rent={val['router_entropy']:.3f} {marker}"
            )
            print()

        # generation
        should_sample = (step + 1) % cfg.sample_every == 0
        if should_sample:
            sample_model(generator, params, tokenizer, cfg, step)

        # save
        should_save = (step + 1) % cfg.checkpoint_every == 0
        if should_save:
            checkpoint_manager.save(
                step=step,
                params=params,
                opt_state=opt_state,
                tokens_seen=tokens_seen,
                best_val_loss=best_val_loss,
                data_state=train_dataset.state_dict(),
            )

    # Final validation
    print()
    print("Final validation...")
    val = evaluate(params)
    best_val_loss = min(best_val_loss, val["loss"])
    final_step = cfg.steps - 1
    if last_validation_step != final_step:
        metrics_logger.log("validation", final_step, tokens_seen, val)

    print(f"Final val_loss={val['loss']:.4f} ppl={val['perplexity']:.2f}")
    checkpoint_manager.save(
        step=final_step,
        params=params,
        opt_state=opt_state,
        tokens_seen=tokens_seen,
        best_val_loss=best_val_loss,
        data_state=train_dataset.state_dict(),
        force=True,
    )

    print("Training complete.")


if __name__ == "__main__":
    main()
