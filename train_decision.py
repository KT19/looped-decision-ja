import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from transformers import AutoTokenizer

from configs import DecisionConfig, load_decision_config
from data.decision_data import DecisionBatch, DecisionDataset, make_token_mask
from models.decision_model import GeneralDecisionHead
from training.checkpoint import (
    prune_step_checkpoints,
    restore_checkpoint,
    save_checkpoint,
)
from training.decision_loss import decision_loss
from training.metrics import MetricsLogger
from training.optimizer import make_muon_adamw_optimizer
from training.pytree import PyTree, global_norm, tree_add, tree_scale
from training.schedules import warmup_cosine_schedule
from utils.load_backbone import build_backbone


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, np.ndarray]
    accuracy_by_source: dict[str, float]
    examples_by_source: dict[str, int]


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for name, value in metrics.items():
        array = np.asarray(value)
        if array.ndim == 0:
            flattened[name] = float(array)
            continue
        for index, item in enumerate(array.reshape(-1), start=1):
            flattened[f"{name}_{index}"] = float(item)
    return flattened


def evaluation_log_metrics(result: EvaluationResult) -> dict[str, float]:
    metrics = flatten_metrics(result.metrics)
    metrics["evaluation_examples"] = float(sum(result.examples_by_source.values()))
    for source, accuracy in result.accuracy_by_source.items():
        metrics[f"accuracy_{source}"] = accuracy
        metrics[f"examples_{source}"] = float(result.examples_by_source[source])
    return metrics


def restore_pretrained_params(checkpoint_path: str | Path) -> PyTree:
    state = restore_checkpoint(checkpoint_path)
    return state["params"]


def to_device(batch: DecisionBatch) -> dict[str, jax.Array]:
    token_mask = make_token_mask(batch.input_lengths, batch.input_ids.shape[1])
    token_mask &= batch.sample_weight[:, None] > 0.0

    return {
        "input_ids": jax.device_put(batch.input_ids),
        "token_mask": jax.device_put(token_mask),
        "option_mask": jax.device_put(batch.option_mask),
        "option_valid": jax.device_put(batch.option_valid),
        "decision_index": jax.device_put(batch.decision_index),
        "targets": jax.device_put(batch.targets),
        "hard_labels": jax.device_put(batch.hard_labels),
        "sample_weight": jax.device_put(batch.sample_weight),
    }


# Evaluation
def make_eval_function(backbone: Any, head: Any) -> Any:
    @jax.jit
    def eval_step(
        backbone_params: Any,
        head_params: Any,
        input_ids: jax.Array,
        token_mask: jax.Array,
        option_mask: jax.Array,
        option_valid: jax.Array,
        decision_index: jax.Array,
        targets: jax.Array,
        hard_labels: jax.Array,
        sample_weight: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array], jax.Array]:
        loop_states, _ = backbone.apply(
            {"params": backbone_params},
            input_ids,
            return_loop_states=True,
            compute_logits=False,
            token_mask=token_mask,
        )

        logits = cast(
            jax.Array,
            head.apply(
                {"params": head_params},
                loop_states,
                option_mask,
                option_valid,
                decision_index,
            ),
        )

        loss, metrics, final_probabilities = decision_loss(
            logits=logits,
            targets=targets,
            hard_labels=hard_labels,
            sample_weight=sample_weight,
        )

        correct = jnp.argmax(final_probabilities, axis=-1) == hard_labels
        return loss, metrics, correct

    return eval_step


def evaluate(
    eval_fn: Any,
    backbone_params: PyTree,
    head_params: PyTree,
    dataset: DecisionDataset,
    batch_size: int,
) -> EvaluationResult:
    sums: dict[str, np.ndarray] | None = None
    example_count = 0.0
    source_correct: dict[str, float] = {}
    source_examples: dict[str, int] = {}

    for batch in dataset.batches(batch_size, epoch=0, shuffle=False, shuffle_options=False):
        d = to_device(batch)

        _, metrics, correct = eval_fn(backbone_params, head_params, **d)
        metrics = jax.device_get(metrics)
        correct = np.asarray(jax.device_get(correct), dtype=np.float32)

        batch_examples = float(np.sum(batch.sample_weight))
        if sums is None:
            sums = {key: np.asarray(value, dtype=np.float64) * batch_examples for key, value in metrics.items()}
        else:
            for key, value in metrics.items():
                sums[key] += np.asarray(value, dtype=np.float64) * batch_examples

        example_count += batch_examples
        for source, is_correct, weight in zip(batch.sources, correct, batch.sample_weight, strict=True):
            if weight == 0.0:
                continue
            source_correct[source] = source_correct.get(source, 0.0) + float(is_correct)
            source_examples[source] = source_examples.get(source, 0) + 1

    if sums is None or example_count == 0.0:
        raise ValueError("evaluation dataset produced no examples")

    return EvaluationResult(
        metrics={key: value / example_count for key, value in sums.items()},
        accuracy_by_source={source: source_correct[source] / count for source, count in source_examples.items()},
        examples_by_source=source_examples,
    )


def print_evaluation(result: EvaluationResult) -> None:
    metrics = result.metrics
    print(f"acc={float(metrics['accuracy']):.4f} nll={float(metrics['nll']):.4f} brier={float(metrics['brier']):.4f}")
    print("accuracy by source:")
    for source, accuracy in sorted(result.accuracy_by_source.items()):
        count = result.examples_by_source[source]
        print(f"  {source}: {accuracy:.4f} (n={count})")


def evaluate_all(
    eval_fn: Any,
    backbone_params: PyTree,
    head_params: PyTree,
    datasets: dict[str, DecisionDataset],
    batch_size: int,
) -> dict[str, EvaluationResult]:
    return {
        name: evaluate(eval_fn, backbone_params, head_params, dataset, batch_size=batch_size)
        for name, dataset in datasets.items()
    }


def combine_evaluations(
    results: dict[str, EvaluationResult],
) -> EvaluationResult:
    if not results:
        raise ValueError("cannot combine an empty evaluation collection")

    total_examples = sum(sum(result.examples_by_source.values()) for result in results.values())
    if total_examples == 0:
        raise ValueError("combined evaluation contains no examples")

    metric_sums: dict[str, np.ndarray] = {}
    source_correct: dict[str, float] = {}
    source_examples: dict[str, int] = {}

    for result in results.values():
        result_examples = sum(result.examples_by_source.values())
        for key, value in result.metrics.items():
            weighted = np.asarray(value, dtype=np.float64) * result_examples
            if key in metric_sums:
                metric_sums[key] += weighted
            else:
                metric_sums[key] = weighted

        for source, count in result.examples_by_source.items():
            source_correct[source] = source_correct.get(source, 0.0) + (result.accuracy_by_source[source] * count)
            source_examples[source] = source_examples.get(source, 0) + count

    return EvaluationResult(
        metrics={key: value / total_examples for key, value in metric_sums.items()},
        accuracy_by_source={source: source_correct[source] / count for source, count in source_examples.items()},
        examples_by_source=source_examples,
    )


def print_combined_evaluation(results: dict[str, EvaluationResult]) -> None:
    print("---eval: combined---")
    print_evaluation(combine_evaluations(results))


# Main
def train(cfg: DecisionConfig) -> None:
    # Backbone
    metadata, model_cfg, backbone = build_backbone(cfg.pretrained_checkpoint)

    if cfg.max_seq_len > model_cfg.max_seq_len:
        raise ValueError("Decision max_seq_len exceeds pretrained max_seq_len.")

    tokenizer_name = metadata["config"]["tokenizer_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if len(tokenizer) != model_cfg.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match pretrained model.")

    # Data
    train_data = DecisionDataset(
        path=cfg.train_data,
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
        max_options=cfg.max_options,
        seed=cfg.seed,
    )
    eval_datasets = {
        name: DecisionDataset(
            path,
            tokenizer,
            max_seq_len=cfg.max_seq_len,
            max_options=cfg.max_options,
            seed=cfg.seed,
        )
        for name, path in cfg.eval_datasets.items()
    }

    microbatches_per_epoch = math.ceil(len(train_data) / cfg.microbatch_size)
    updates_per_epoch = math.ceil(microbatches_per_epoch / cfg.grad_accum_steps)

    total_updates = updates_per_epoch * cfg.epochs
    effective_batch = cfg.microbatch_size * cfg.grad_accum_steps

    print()
    print("=" * 64)
    print(f"train examples: {len(train_data)}")
    for name, dataset in eval_datasets.items():
        print(f"eval examples [{name}]: {len(dataset)}")
    print(f"updates per epoch: {updates_per_epoch}")
    print(f"effective batch size: {effective_batch}")

    # Pretrained params
    backbone_params = restore_pretrained_params(cfg.pretrained_checkpoint)
    # Decision head
    head = GeneralDecisionHead(d_model=model_cfg.d_model, d_head=model_cfg.d_model)

    init_rng = jax.random.PRNGKey(cfg.seed)

    dummy_loop_states = jnp.zeros((1, 1, 8, model_cfg.d_model), dtype=jnp.bfloat16)
    dummy_option_mask = jnp.zeros((1, cfg.max_options, 8), dtype=jnp.float32)
    dummy_option_valid = jnp.ones(
        (
            1,
            cfg.max_options,
        ),
        dtype=jnp.bool_,
    )
    dummy_decision_index = jnp.asarray([7], dtype=jnp.int32)

    head_params = head.init(
        init_rng,
        dummy_loop_states,
        dummy_option_mask,
        dummy_option_valid,
        dummy_decision_index,
    )["params"]

    # Optimizer
    warmup_steps = max(1, int(total_updates * cfg.warmup_ratio))
    schedule_steps = max(total_updates, warmup_steps + 1)
    backbone_schedule = warmup_cosine_schedule(
        peak_lr=cfg.backbone_lr,
        total_steps=schedule_steps,
        warmup_steps=warmup_steps,
        min_lr=cfg.backbone_lr * cfg.min_lr_ratio,
    )
    head_schedule = warmup_cosine_schedule(
        peak_lr=cfg.head_lr,
        total_steps=schedule_steps,
        warmup_steps=warmup_steps,
        min_lr=cfg.head_lr * cfg.min_lr_ratio,
    )
    backbone_optimizer = make_muon_adamw_optimizer(
        backbone_params,
        backbone_schedule,
        weight_decay=cfg.backbone_weight_decay,
        muon_momentum=cfg.muon_momentum,
        muon_ns_steps=cfg.muon_ns_steps,
    )
    head_optimizer = optax.adamw(
        head_schedule,
        b1=0.9,
        b2=0.95,
        weight_decay=cfg.head_weight_decay,
    )

    backbone_opt_state = backbone_optimizer.init(backbone_params)
    head_opt_state = head_optimizer.init(head_params)

    start_epoch = 0
    examples_seen = 0
    update_step = 0
    best_eval_nll = np.inf

    # Resume
    if cfg.resume_checkpoint is not None:
        target = {
            "backbone_params": backbone_params,
            "head_params": head_params,
            "backbone_opt_state": backbone_opt_state,
            "head_opt_state": head_opt_state,
            "next_epoch": np.asarray(0, dtype=np.int32),
            "examples_seen": np.asarray(0, dtype=np.int64),
            "update_step": np.asarray(0, dtype=np.int64),
            "best_eval_nll": np.asarray(np.inf, dtype=np.float32),
        }
        restored = restore_checkpoint(cfg.resume_checkpoint, target=target)
        backbone_params = restored["backbone_params"]
        head_params = restored["head_params"]
        backbone_opt_state = restored["backbone_opt_state"]
        head_opt_state = restored["head_opt_state"]
        start_epoch = int(restored["next_epoch"])
        examples_seen = int(restored["examples_seen"])
        update_step = int(restored["update_step"])
        best_eval_nll = float(restored["best_eval_nll"])
        print(f"resumed from epoch {start_epoch}")

    resume_examples_in_epoch = examples_seen - start_epoch * len(train_data)
    last_eval_step = -1

    # Loss + gradient
    router_aux_coef = float(metadata["config"].get("router_aux_loss_coef", 0.0))
    router_z_coef = float(metadata["config"].get("router_z_loss_coef", 0.0))

    @jax.jit
    def compute_grads(
        backbone_params: Any,
        head_params: Any,
        input_ids: jax.Array,
        token_mask: jax.Array,
        option_mask: jax.Array,
        option_valid: jax.Array,
        decision_index: jax.Array,
        targets: jax.Array,
        hard_labels: jax.Array,
        sample_weight: jax.Array,
    ) -> tuple[dict[str, jax.Array], PyTree, PyTree]:
        def loss_fn(bp: PyTree, hp: PyTree) -> tuple[jax.Array, dict[str, jax.Array]]:
            loop_states, router_stats = backbone.apply(
                {"params": bp},
                input_ids,
                return_loop_states=True,
                compute_logits=False,
                token_mask=token_mask,
            )

            logits = cast(
                jax.Array,
                head.apply(
                    {"params": hp},
                    loop_states,
                    option_mask,
                    option_valid,
                    decision_index,
                ),
            )
            base_loss, metrics, _ = decision_loss(logits, targets, hard_labels, sample_weight)

            router_aux = jnp.asarray(router_stats["router_aux_loss"], dtype=jnp.float32)  # type: ignore
            router_z = jnp.asarray(router_stats["router_z_loss"], dtype=jnp.float32)  # type: ignore

            total_loss = base_loss + router_aux_coef * router_aux + router_z_coef * router_z

            output_metrics = {
                **metrics,
                "total_loss": total_loss,
                "router_aux": router_aux,
                "router_z": router_z,
            }

            return total_loss, output_metrics

        (_, metrics), (backbone_grads, head_grads) = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
            backbone_params, head_params
        )

        return metrics, backbone_grads, head_grads

    @jax.jit
    def apply_grads(
        backbone_params: PyTree,
        head_params: PyTree,
        backbone_opt_state: PyTree,
        head_opt_state: PyTree,
        backbone_grads: PyTree,
        head_grads: PyTree,
    ) -> tuple[PyTree, PyTree, PyTree, PyTree, jax.Array, jax.Array]:
        grad_norm = global_norm(backbone_grads, head_grads)

        clip_scale = jnp.minimum(1.0, cfg.max_grad_norm / (grad_norm + 1e-6))
        backbone_grads = tree_scale(backbone_grads, clip_scale)
        head_grads = tree_scale(head_grads, clip_scale)

        (backbone_updates, backbone_opt_state) = backbone_optimizer.update(
            backbone_grads, backbone_opt_state, backbone_params
        )

        (head_updates, head_opt_state) = head_optimizer.update(head_grads, head_opt_state, head_params)

        backbone_params = optax.apply_updates(backbone_params, backbone_updates)
        head_params = optax.apply_updates(head_params, head_updates)

        return (
            backbone_params,
            head_params,
            backbone_opt_state,
            head_opt_state,
            grad_norm,
            clip_scale,
        )

    eval_fn = make_eval_function(backbone, head)

    # output metadata
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_logger = MetricsLogger(directory=str(output_dir), append=cfg.resume_checkpoint is not None)

    config_path = output_dir / "decision_config.json"

    config_path.write_text(
        json.dumps(
            {
                **asdict(cfg),
                "tokenizer_name": tokenizer_name,
                "n_loops": model_cfg.n_loops,
                "d_model": model_cfg.d_model,
                "decision_representation": "final_coda",
                "supervised_states": 1,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def save_evaluation_checkpoint(result: EvaluationResult, seen: int) -> None:
        nonlocal best_eval_nll
        eval_nll = float(result.metrics["nll"])
        is_best = eval_nll < best_eval_nll
        if is_best:
            best_eval_nll = eval_nll
        state = {
            "backbone_params": backbone_params,
            "head_params": head_params,
            "backbone_opt_state": backbone_opt_state,
            "head_opt_state": head_opt_state,
            "next_epoch": np.asarray(seen // len(train_data), dtype=np.int32),
            "examples_seen": np.asarray(seen, dtype=np.int64),
            "update_step": np.asarray(update_step, dtype=np.int64),
            "best_eval_nll": np.asarray(best_eval_nll, dtype=np.float32),
        }
        save_checkpoint(output_dir / f"step_{update_step:08d}", state, force=True)
        if is_best:
            save_checkpoint(output_dir / "best", state, force=True)
            print(" saved new best checkpoint")
        # Prune only after both the step and any improved best are safely saved.
        for removed in prune_step_checkpoints(output_dir, keep=2):
            print(f" removed old checkpoint: {removed.name}")

    # Train
    for epoch in range(start_epoch, cfg.epochs):
        epoch_number = epoch + 1
        epoch_start = time.perf_counter()
        accumulated_backbone: PyTree | None = None
        accumulated_head: PyTree | None = None
        accumulated_metrics: dict[str, jax.Array] | None = None
        accum_count = 0
        accumulated_examples = 0.0

        examples_since_log = 0
        log_start = time.perf_counter()

        def perform_update(current_epoch: int, current_examples_seen: int) -> None:
            nonlocal \
                backbone_params, \
                head_params, \
                backbone_opt_state, \
                head_opt_state, \
                accumulated_backbone, \
                accumulated_head, \
                accumulated_metrics, \
                accum_count, \
                accumulated_examples, \
                update_step, \
                examples_since_log, \
                log_start, \
                last_eval_step

            if (
                accumulated_backbone is None
                or accumulated_head is None
                or accumulated_metrics is None
                or accumulated_examples <= 0.0
            ):
                raise RuntimeError("cannot update without accumulated gradients")

            scale = 1.0 / accumulated_examples
            backbone_grads = tree_scale(accumulated_backbone, scale)
            head_grads = tree_scale(accumulated_head, scale)

            (
                backbone_params,
                head_params,
                backbone_opt_state,
                head_opt_state,
                grad_norm,
                clip_scale,
            ) = apply_grads(
                backbone_params,
                head_params,
                backbone_opt_state,
                head_opt_state,
                backbone_grads,
                head_grads,
            )

            grad_norm.block_until_ready()
            update_step += 1

            averaged_metrics = jax.device_get(tree_scale(accumulated_metrics, scale))
            if update_step % cfg.log_every == 0:
                elapsed = time.perf_counter() - log_start
                examples_per_second = examples_since_log / max(elapsed, 1e-9)
                schedule_step = max(update_step - 1, 0)
                lr_backbone = float(np.asarray(backbone_schedule(schedule_step)).item())
                lr_head = float(np.asarray(head_schedule(schedule_step)).item())

                logged_metrics = flatten_metrics(averaged_metrics)
                logged_metrics.update(
                    {
                        "epoch": float(current_epoch),
                        "grad_norm": float(grad_norm),
                        "clip_scale": float(clip_scale),
                        "backbone_learning_rate": lr_backbone,
                        "head_learning_rate": lr_head,
                        "examples_per_second": examples_per_second,
                    }
                )
                metrics_logger.log_examples("train", update_step, current_examples_seen, logged_metrics)

                print(
                    f"epoch={current_epoch} step={update_step:06d} "
                    f"loss={float(averaged_metrics['loss']):.4f} "
                    f"total={float(averaged_metrics['total_loss']):.4f} "
                    f"acc={float(averaged_metrics['accuracy']):.4f} "
                    f"nll={float(averaged_metrics['nll']):.4f} "
                    f"brier={float(averaged_metrics['brier']):.4f} "
                    f"grad={float(grad_norm):.3f} clip={float(clip_scale):.3f} "
                    f"lr={lr_backbone:.2e}/{lr_head:.2e} examples/s={examples_per_second:.1f}"
                )

                log_start = time.perf_counter()
                examples_since_log = 0

            if update_step % cfg.eval_every == 0:
                evaluations = evaluate_all(
                    eval_fn,
                    backbone_params,
                    head_params,
                    eval_datasets,
                    batch_size=cfg.eval_batch_size,
                )
                print_combined_evaluation(evaluations)
                combined_evaluation = combine_evaluations(evaluations)
                metrics_logger.log_examples(
                    "validation",
                    update_step,
                    current_examples_seen,
                    evaluation_log_metrics(combined_evaluation),
                )
                last_eval_step = update_step
                save_evaluation_checkpoint(combined_evaluation, current_examples_seen)

            accumulated_backbone = None
            accumulated_head = None
            accumulated_metrics = None
            accum_count = 0
            accumulated_examples = 0.0

        for batch_index, batch in enumerate(
            train_data.batches(cfg.microbatch_size, epoch=epoch, shuffle=True, shuffle_options=True)
        ):
            if epoch == start_epoch and batch_index * cfg.microbatch_size < resume_examples_in_epoch:
                continue
            d = to_device(batch)

            metrics, backbone_grads, head_grads = compute_grads(backbone_params, head_params, **d)
            batch_examples = float(np.sum(batch.sample_weight))
            weighted_backbone_grads = tree_scale(backbone_grads, batch_examples)
            weighted_head_grads = tree_scale(head_grads, batch_examples)
            weighted_metrics = tree_scale(metrics, batch_examples)

            if accumulated_backbone is None:
                accumulated_backbone = weighted_backbone_grads
                accumulated_head = weighted_head_grads
                accumulated_metrics = weighted_metrics

            else:
                accumulated_backbone = tree_add(accumulated_backbone, weighted_backbone_grads)
                accumulated_head = tree_add(accumulated_head, weighted_head_grads)
                assert accumulated_metrics is not None
                accumulated_metrics = tree_add(accumulated_metrics, weighted_metrics)

            accum_count += 1
            accumulated_examples += batch_examples
            examples_seen += int(batch_examples)
            examples_since_log += int(batch_examples)

            if accum_count == cfg.grad_accum_steps:
                perform_update(epoch_number, examples_seen)

        if accum_count > 0:
            perform_update(epoch_number, examples_seen)

        # End-of-epoch
        if last_eval_step != update_step:
            evaluations = evaluate_all(
                eval_fn,
                backbone_params,
                head_params,
                eval_datasets,
                batch_size=cfg.eval_batch_size,
            )
            combined_evaluation = combine_evaluations(evaluations)
            metrics_logger.log_examples(
                "validation",
                update_step,
                examples_seen,
                evaluation_log_metrics(combined_evaluation),
            )
            last_eval_step = update_step
            print_combined_evaluation(evaluations)
            save_evaluation_checkpoint(combined_evaluation, examples_seen)
        print()
        print(f"epoch {epoch_number} complete in {time.perf_counter() - epoch_start:.1f}s")

        print()


def main() -> None:
    train(load_decision_config("configs/decision.yaml"))


if __name__ == "__main__":
    main()
