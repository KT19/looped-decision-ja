import flax.linen as nn
import jax

from models.attention import GQAAttention
from models.config import ModelConfig
from models.feedforward import SwiGLU
from models.normalization import RMSNorm


class DenseTransformerBlock(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        cfg = self.config

        # Attention
        attn_input = RMSNorm(cfg.d_model, name="attn_norm")(x)
        attn_out = GQAAttention(cfg, name="attention")(attn_input)

        x = x + attn_out

        # FFN
        ffn_input = RMSNorm(cfg.d_model, name="ffn_norm")(x)
        ffn_out = SwiGLU(cfg, name="ffn")(ffn_input)

        x = x + ffn_out

        return x
