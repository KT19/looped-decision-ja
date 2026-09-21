import flax.linen as nn
import jax

from models.attention import GQAAttention
from models.config import ModelConfig
from models.depth_conditioning import DepthConditioning
from models.moe import SparseMoE


class RecurrentSparseBlock(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        depth_embedding: jax.Array,
        residual_scale: float,
        token_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        cfg = self.config

        # attention
        attn_input = DepthConditioning(cfg, name="attn_conditoning")(x, depth_embedding)
        attn_out = GQAAttention(cfg, name="attention")(attn_input)

        x = x + residual_scale * attn_out

        # Sparse MoE
        moe_input = DepthConditioning(cfg, name="moe_conditioning")(x, depth_embedding)

        moe_out, moe_stats = SparseMoE(
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            n_experts=cfg.n_experts,
            top_k=cfg.top_k,
            capacity_factor=cfg.capacity_factor,
            name="moe",
        )(moe_input, token_mask=token_mask)

        x = x + residual_scale * moe_out

        return x, moe_stats
