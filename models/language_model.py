import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from models.config import ModelConfig
from models.dense_transformer_block import DenseTransformerBlock
from models.diagnostics import mean_cosine_similarity, tensor_rms
from models.normalization import RMSNorm
from models.recurrent_core_stack import RecurrentCoreStack


class LanguageModel(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        tokens: jax.Array,
        return_loop_states: bool = False,
        compute_logits: bool = True,
        token_mask: jax.Array | None = None,
    ) -> Any:
        cfg = self.config

        batch_size, sequence_length = tokens.shape
        if token_mask is not None and token_mask.shape != (batch_size, sequence_length):
            raise ValueError("token_mask must have the same [batch, sequence] shape as tokens")

        # Token embedding
        token_embedding = nn.Embed(num_embeddings=cfg.vocab_size, features=cfg.d_model, name="token_embedding")

        x = token_embedding(tokens)

        # Learned positional embeddings
        pos = self.param(
            "position_embedding",
            nn.initializers.normal(stddev=0.02),
            (cfg.max_seq_len, cfg.d_model),
        )

        x = x + pos[None, :sequence_length, :]

        # Prelude
        x = DenseTransformerBlock(cfg, name="prelude")(x)

        # Loop embedding
        loop_embeddings = self.param(
            "loop_embeddings",
            nn.initializers.normal(stddev=0.02),
            (cfg.n_loops, cfg.d_model),
        )
        # stabilize
        if cfg.loop_residual_scale is None:
            residual_scale = 1.0 / math.sqrt(cfg.n_loops * cfg.n_core_blocks)
        else:
            residual_scale = cfg.loop_residual_scale

        # shared core
        recurrent_core = RecurrentCoreStack(cfg, name="recurrent_stack")

        # Diagnostics
        router_aux_losses = []
        router_z_losses = []

        expert_fractions = []
        dropped_fractions = []
        router_entropies = []

        state_rms_values = []
        delta_rms_values = []
        state_cosines = []

        loop_states = []

        # recurrent depth
        for loop_idx in range(cfg.n_loops):
            previous_x = x

            x, moe_stats = recurrent_core(x, loop_embeddings[loop_idx], residual_scale, token_mask=token_mask)

            # Routing
            router_aux_losses.append(moe_stats["router_aux_loss"])
            router_z_losses.append(moe_stats["router_z_loss"])

            expert_fraction = moe_stats["expert_fraction"]
            expert_fractions.append(expert_fraction)

            dropped_fractions.append(moe_stats["dropped_fraction"])

            # Aggregate expert-use entropy
            p = jnp.clip(expert_fraction, 1e-8, 1.0)
            router_entropy = -jnp.sum(p * jnp.log(p))
            router_entropies.append(router_entropy)

            # Recurrent dynamics
            delta = x - previous_x
            state_rms_values.append(tensor_rms(x))
            delta_rms_values.append(tensor_rms(delta))
            state_cosines.append(mean_cosine_similarity(previous_x, x))

            if return_loop_states:
                loop_states.append(x)

        # Reuse the pretrained output transformation for both training tasks.
        x = DenseTransformerBlock(cfg, name="coda")(x)
        x = RMSNorm(cfg.d_model, name="final_norm")(x)

        # Decision supervision uses only the final, post-coda representation.
        if not compute_logits:
            if not return_loop_states:
                raise ValueError("compute_logits=False requires return_loop_states=True")

            # Statistics
            stats = {
                "router_aux_loss": jnp.mean(jnp.stack(router_aux_losses)),
                "router_z_loss": jnp.mean(jnp.stack(router_z_losses)),
                "expert_fraction": jnp.stack(expert_fractions),
                "dropped_fraction": jnp.stack(dropped_fractions),
                "router_entropy": jnp.stack(router_entropies),
                "state_rms": jnp.stack(state_rms_values),
                "delta_rms": jnp.stack(delta_rms_values),
                "state_cosine": jnp.stack(state_cosines),
            }

            return x[None, ...], stats

        # LM head
        logits = token_embedding.attend(x)

        # Aggregate router losses
        router_aux_loss = jnp.mean(jnp.stack(router_aux_losses))
        router_z_loss = jnp.mean(jnp.stack(router_z_losses))

        # Statistics
        stats = {
            "router_aux_loss": router_aux_loss,
            "router_z_loss": router_z_loss,
            "expert_fraction": jnp.stack(expert_fractions),
            "dropped_fraction": jnp.stack(dropped_fractions),
            "router_entropy": jnp.stack(router_entropies),
            "state_rms": jnp.stack(state_rms_values),
            "delta_rms": jnp.stack(delta_rms_values),
            "state_cosine": jnp.stack(state_cosines),
        }

        if return_loop_states:
            stacked_loop_states = jnp.stack(loop_states, axis=0)

            return logits, stats, stacked_loop_states

        return logits, stats
