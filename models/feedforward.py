import flax.linen as nn
import jax

from models.config import ModelConfig


class SwiGLU(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        cfg = self.config

        gate = nn.Dense(cfg.d_ff, use_bias=False, name="gate_proj")(x)
        up = nn.Dense(cfg.d_ff, use_bias=False, name="up_proj")(x)

        hidden = nn.silu(gate) * up

        return nn.Dense(cfg.d_model, use_bias=False, name="down_proj")(hidden)
