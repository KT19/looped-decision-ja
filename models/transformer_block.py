import flax.linen as nn
import jax

from models.attention import GQAAttention
from models.config import ModelConfig
from models.moe import SparseMoE
from models.normalization import RMSNorm


class TransformerBlock(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, dict]:
        cfg = self.config

        x = x + GQAAttention(cfg, name="attention")(
            RMSNorm(cfg.d_model, name="attn_norm")(x)
        )

        moe_out, moe_stats = SparseMoE(
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            n_experts=cfg.n_experts,
            top_k=cfg.top_k,
            capacity_factor=cfg.capacity_factor,
            name="moe",
        )(RMSNorm(cfg.d_model, name="ffn_norm")(x))

        x = x + moe_out

        return x, moe_stats
