import jax
import jax.numpy as jnp


def decision_loss(
    logits: jax.Array,
    targets: jax.Array,
    hard_labels: jax.Array,
    sample_weight: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array], jax.Array]:
    """
    logits:
        [R, B, O]

    targets:
        [B, O]
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [loops, batch, options]")
    if targets.shape != logits.shape[1:]:
        raise ValueError("targets must have shape [batch, options]")
    if hard_labels.shape != logits.shape[1:2]:
        raise ValueError("hard_labels must have shape [batch]")
    if sample_weight.shape != logits.shape[1:2]:
        raise ValueError("sample_weight must have shape [batch]")

    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    probs = jnp.exp(log_probs)

    # Soft-label CE for every loop
    ce = -jnp.sum(targets[None, ...] * log_probs, axis=-1)  # (R, B)

    # all loops receive supervision
    n_loops = logits.shape[0]

    loop_weights = jnp.arange(1, n_loops + 1, dtype=jnp.float32)

    loop_weights = loop_weights / jnp.sum(loop_weights)
    denom = jnp.maximum(jnp.sum(sample_weight), 1.0)

    loss_per_loop = jnp.sum(ce * sample_weight[None, :], axis=-1) / denom

    loss = jnp.sum(loop_weights * loss_per_loop)

    # Final-loop metrics
    final_probs = probs[-1]
    prediction = jnp.argmax(final_probs, axis=-1)

    correct = (prediction == hard_labels).astype(jnp.float32)
    accuracy = jnp.sum(correct * sample_weight) / denom

    final_nll = jnp.sum(ce[-1] * sample_weight) / denom

    # Multiclass Brier score
    brier_per_example = jnp.sum(jnp.square(final_probs - targets), axis=-1)
    brier = jnp.sum(brier_per_example * sample_weight) / denom
    confidence = jnp.max(final_probs, axis=-1)
    mean_confidence = jnp.sum(confidence * sample_weight) / denom

    metrics = {
        "loss": loss,
        "nll": final_nll,
        "accuracy": accuracy,
        "brier": brier,
        "mean_confidence": mean_confidence,
        "loss_per_loop": loss_per_loop,
    }

    return loss, metrics, final_probs
