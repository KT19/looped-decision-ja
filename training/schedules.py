import optax


def warmup_cosine_schedule(*, peak_lr: float, total_steps: int, warmup_steps: int, min_lr: float) -> optax.Schedule:
    if peak_lr <= 0.0:
        raise ValueError("peak_lr must be positive")
    if not 0.0 <= min_lr <= peak_lr:
        raise ValueError("min_lr must be between zero and peak_lr")
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be positive")
    if total_steps <= warmup_steps:
        raise ValueError("total_steps must be larger than warmup_steps")

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=min_lr,
    )
