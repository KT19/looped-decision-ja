import math

import flax.linen as nn
import jax
import jax.numpy as jnp


class SparseMoE(nn.Module):
    d_model: int
    d_ff: int
    n_experts: int
    top_k: int
    capacity_factor: float = 1.2

    @nn.compact
    def __call__(self, x: jax.Array, token_mask: jax.Array | None = None) -> tuple[jax.Array, dict[str, jax.Array]]:
        """
        x: [B, T, D]

        Returns:
            output: [B, T, D]
            stats: router statistics / overflow statistics
        """
        batch_size, sequence_length, model_width = x.shape
        expert_count = self.n_experts
        top_k = self.top_k

        if not 1 <= top_k <= expert_count:
            raise ValueError("top_k must be between 1 and n_experts")
        if token_mask is None:
            token_mask = jnp.ones((batch_size, sequence_length), dtype=jnp.bool_)
        elif token_mask.shape != (batch_size, sequence_length):
            raise ValueError("token_mask must have shape [batch, sequence]")

        token_count = batch_size * sequence_length
        assignment_count = token_count * top_k
        # Average assignments per expert
        # Keep capacity static for JIT compilation. Masked tokens do not consume slots.
        capacity = math.ceil(self.capacity_factor * assignment_count / expert_count)

        # router
        router_logits = nn.Dense(expert_count, use_bias=False, name="router")(x)

        router_probs = jax.nn.softmax(router_logits.astype(jnp.float32), axis=-1)

        top_values, top_indices = jax.lax.top_k(router_probs, top_k)

        top_weights = top_values / (jnp.sum(top_values, axis=-1, keepdims=True) + 1e-9)

        # Expert parameters
        gate_w = self.param(
            "gate_w",
            nn.initializers.lecun_normal(),
            (expert_count, model_width, self.d_ff),
        )

        up_w = self.param(
            "up_w",
            nn.initializers.lecun_normal(),
            (expert_count, model_width, self.d_ff),
        )

        down_w = self.param(
            "down_w",
            nn.initializers.lecun_normal(),
            (expert_count, self.d_ff, model_width),
        )

        # Flatten
        x_flat = x.reshape(token_count, model_width)
        token_active = token_mask.reshape(token_count).astype(jnp.bool_)

        expert_ids = top_indices.reshape(assignment_count)
        routing_weights = top_weights.reshape(assignment_count)

        token_ids = jnp.repeat(jnp.arange(token_count, dtype=jnp.int32), top_k)
        assignment_active = token_active[token_ids]

        # Compute slot position inside each expert
        # one_hot: [A, E]
        expert_one_hot = jax.nn.one_hot(expert_ids, expert_count, dtype=jnp.int32)
        active_one_hot = expert_one_hot * assignment_active[:, None].astype(jnp.int32)
        cumulative = jnp.cumsum(active_one_hot, axis=0) - 1
        slot_ids = jnp.sum(cumulative * expert_one_hot, axis=-1)

        # capacity mask
        valid = assignment_active & (slot_ids < capacity)
        safe_slots = jnp.minimum(slot_ids, capacity - 1)
        valid_f = valid.astype(x.dtype)

        # pack
        # expert capacity E*capacity*D
        expert_inputs = jnp.zeros((expert_count, capacity, model_width), dtype=x.dtype)
        assignment_inputs = x_flat[token_ids] * valid_f[:, None]
        expert_inputs = expert_inputs.at[expert_ids, safe_slots].add(assignment_inputs)

        # Batched expert computation
        gate = jnp.einsum("ecd,edf->ecf", expert_inputs, gate_w)
        up = jnp.einsum("ecd,edf->ecf", expert_inputs, up_w)

        hidden = nn.silu(gate) * up

        expert_outputs = jnp.einsum("ecf,efd->ecd", hidden, down_w)

        # Gather expert results back
        selected_outputs = expert_outputs[expert_ids, safe_slots]
        weighted_outputs = selected_outputs * routing_weights[:, None].astype(x.dtype) * valid_f[:, None]

        output_flat = jnp.zeros((token_count, model_width), dtype=x.dtype)
        output_flat = output_flat.at[token_ids].add(weighted_outputs)
        output = output_flat.reshape(batch_size, sequence_length, model_width)

        # Router losses and statistics
        valid_token_count = jnp.maximum(jnp.sum(token_active), 1)
        valid_assignment_count = jnp.maximum(valid_token_count * top_k, 1)
        assignment_fraction = jnp.sum(active_one_hot, axis=0).astype(jnp.float32) / valid_assignment_count
        probability_fraction = (
            jnp.sum(router_probs * token_mask[..., None].astype(jnp.float32), axis=(0, 1)) / valid_token_count
        )

        aux_loss = expert_count * jnp.sum(assignment_fraction * probability_fraction)
        z_per_token = jnp.square(jax.nn.logsumexp(router_logits.astype(jnp.float32), axis=-1))
        z_loss = jnp.sum(z_per_token * token_mask.astype(jnp.float32)) / valid_token_count

        # Overflow stats
        dropped_fraction = 1.0 - jnp.sum(valid.astype(jnp.float32)) / valid_assignment_count
        expert_counts = jnp.sum(active_one_hot, axis=0)
        max_expert_load = jnp.max(expert_counts)

        stats = {
            "router_aux_loss": aux_loss,
            "router_z_loss": z_loss,
            "expert_fraction": assignment_fraction,
            "router_probability": probability_fraction,
            "dropped_fraction": dropped_fraction,
            "expert_counts": expert_counts,
            "capacity": jnp.asarray(capacity, dtype=jnp.int32),
            "max_expert_load": max_expert_load,
        }

        return output, stats
