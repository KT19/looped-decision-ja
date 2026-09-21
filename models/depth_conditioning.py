import flax.linen as nn
import jax

from models.config import ModelConfig
from models.normalization import RMSNorm


class DepthConditioning(nn.Module):
    """
    Lightweight recurrent-depth conditioning
    """

    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array, depth_embedding: jax.Array) -> jax.Array:
        cfg = self.config

        x = RMSNorm(cfg.d_model, name="norm")(x)

        return x + cfg.depth_embedding_scale * depth_embedding[None, None, :]
