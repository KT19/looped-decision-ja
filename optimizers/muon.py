from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


def zeropower_via_newton_schulz(x: jax.Array, steps: int = 5) -> jax.Array:
    """
    Approximate the orthogonal polar factor of a matrix.

    Input:
        x: [m, n] or [e, m, n]

    Output:
        approximately orthogonalized matrix [m, n]
    """
    assert x.ndim in (2, 3)

    original_dtype = x.dtype

    # ns is more convenient when wors <= columns
    transposed = x.shape[-2] > x.shape[-1]

    if transposed:
        x = jnp.swapaxes(x, -1, -2)

    x = x.astype(jnp.float32)

    # Normalize the matrix
    norm = jnp.linalg.norm(x, axis=(-2, -1), keepdims=True)

    x = x / (norm + 1e-7)

    # Polynomial coefficients used in Muon
    a = 3.4445
    b = -4.7750
    c = 2.0315

    def body(_, x):
        xx_t = x @ jnp.swapaxes(x, -1, -2)
        x = a * x + b * (xx_t @ x) + c * (xx_t @ xx_t @ x)
        return x

    x = jax.lax.fori_loop(0, steps, body, x)

    if transposed:
        x = jnp.swapaxes(x, -1, -2)

    return x.astype(original_dtype)


class MuonState(NamedTuple):
    momentum: optax.Updates
    count: jax.Array


def muon(
    learning_rate: Any, momentum: float = 0.95, ns_steps: int = 5
) -> optax.GradientTransformation:
    """
    Minimal Muon optimizer for 2D parameters
    """

    def init_fn(params):
        return MuonState(
            momentum=jax.tree_util.tree_map(
                jnp.zeros_like,
                params,
            ),
            count=jnp.asarray(0, dtype=jnp.int32),
        )

    def get_lr(count) -> jax.Array:
        if callable(learning_rate):
            return learning_rate(count)  # type: ignore

        return jnp.asarray(learning_rate, dtype=jnp.float32)

    def update_fn(updates, state, params=None):
        del params

        lr = get_lr(state.count)

        new_momentum = jax.tree_util.tree_map(
            lambda m, g: momentum * m + g,
            state.momentum,
            updates,
        )

        def transform(g, m):
            # Nesterov-style momentum
            update = g + momentum * m

            if update.ndim in (2, 3):
                update = zeropower_via_newton_schulz(update, steps=ns_steps)

                # Scale based on matrix shape
                update = update * jnp.sqrt(max(update.shape[-2], update.shape[-1]))

            return -lr * update

        new_updates = jax.tree_util.tree_map(transform, updates, new_momentum)

        return new_updates, MuonState(new_momentum, state.count + 1)

    return optax.GradientTransformation(
        init_fn,
        update_fn,
    )
