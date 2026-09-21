import flax.linen as nn
import jax
import jax.numpy as jnp


def rms_norm(x: jax.Array, weight: jax.Array, eps: float = 1e-6) -> jax.Array:
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(variance + eps) * weight


class RMSNorm(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        weight = self.param(
            "weight",
            nn.initializers.ones,
            (self.dim,),
        )

        return rms_norm(x, weight)
