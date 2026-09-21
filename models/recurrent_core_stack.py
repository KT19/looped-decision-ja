import flax.linen as nn
import jax
import jax.numpy as jnp

from models.config import ModelConfig
from models.diagnostics import mean_cosine_similarity, tensor_rms
from models.recurrent_sparse_block import RecurrentSparseBlock


class RecurrentCoreStack(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        loop_embedding: jax.Array,
        residual_scale: float,
        token_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        cfg = self.config

        # stats
        router_aux_losses = []
        router_z_losses = []

        expert_fractions = []
        dropped_fractions = []
        router_entropies = []

        block_state_rms = []
        block_delta_rms = []
        block_state_cosine = []

        for block_idx in range(cfg.n_core_blocks):
            previous_x = x

            depth_embedding = loop_embedding
            x, moe_stats = RecurrentSparseBlock(cfg, name=f"block_{block_idx}")(
                x, depth_embedding, residual_scale, token_mask=token_mask
            )

            router_aux_losses.append(moe_stats["router_aux_loss"])
            router_z_losses.append(moe_stats["router_z_loss"])

            expert_fraction = moe_stats["expert_fraction"]
            dropped_fraction = moe_stats["dropped_fraction"]

            expert_fractions.append(expert_fraction)
            dropped_fractions.append(dropped_fraction)

            p = jnp.clip(expert_fraction, 1e-8, 1.0)
            router_entropy = -jnp.sum(p * jnp.log(p))
            router_entropies.append(router_entropy)

            delta = x - previous_x

            block_state_rms.append(tensor_rms(x))
            block_delta_rms.append(tensor_rms(delta))
            block_state_cosine.append(mean_cosine_similarity(previous_x, x))

        stats = {
            "router_aux_loss": jnp.mean(jnp.stack(router_aux_losses)),
            "router_z_loss": jnp.mean(jnp.stack(router_z_losses)),
            "expert_fraction": jnp.stack(expert_fractions),
            "dropped_fraction": jnp.stack(dropped_fractions),
            "router_entropy": jnp.stack(router_entropies),
            "block_state_rms": jnp.stack(block_state_rms),
            "block_delta_rms": jnp.stack(block_delta_rms),
            "block_state_cosine": jnp.stack(block_state_cosine),
        }

        return x, stats
