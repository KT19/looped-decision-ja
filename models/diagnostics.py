import jax
import jax.numpy as jnp


def tensor_rms(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    x = x.astype(jnp.float32)

    return jnp.sqrt(jnp.mean(jnp.square(x)) + eps)


def mean_cosine_similarity(x: jax.Array, y: jax.Array, eps: float = 1e-8) -> jax.Array:
    x = x.astype(jnp.float32)
    y = y.astype(jnp.float32)

    x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)
    y_norm = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + eps)

    cosine = jnp.sum(x_norm * y_norm, axis=-1)

    return jnp.mean(cosine)
