import math

import flax.linen as nn
import jax
import jax.numpy as jnp


class DecisionRMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        scale = self.param(
            "scale",
            nn.initializers.ones,
            (x.shape[-1],),
        )

        rms = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(rms + self.eps)

        return x * scale


class GeneralDecisionHead(nn.Module):
    """
    General textual-option classifier.

    Inputs:
    ===
    loop_states:
        [R, B, T, D]

    option_mask:
        [B, O, T]

    option_valid:
        [B, O]

    decision_index:
        [B]

    Outputs:
    ===
    logits:
        [R, B, O]

    One set of logits for every recurrent loop
    """

    d_model: int
    d_head: int | None = None

    @nn.compact
    def __call__(
        self,
        loop_states: jax.Array,
        option_mask: jax.Array,
        option_valid: jax.Array,
        decision_index: jax.Array,
    ) -> jax.Array:
        d_head = self.d_head or self.d_model

        """
        Option representation
        Mean-pool tokens belonging to each textual option
        """
        option_mask_f = option_mask.astype(loop_states.dtype)
        option_count = jnp.sum(option_mask_f, axis=-1)
        option_count = jnp.maximum(option_count, 1.0)

        option_repr = jnp.einsum("rbtd,bot->rbod", loop_states, option_mask_f)
        option_repr = option_repr / option_count[None, ..., None]

        """
        Final decision-token representation
        """
        recurrent_steps, batch_size, _, model_width = loop_states.shape
        gather_index = decision_index[None, :, None, None]  # (1, B, 1, 1)
        gather_index = jnp.broadcast_to(
            gather_index, (recurrent_steps, batch_size, 1, model_width)
        )
        decision_repr = jnp.take_along_axis(loop_states, gather_index, axis=2)[
            :, :, 0, :
        ]  # R, B, D

        # shared normalization
        norm = DecisionRMSNorm(name="norm")
        decision_repr = norm(decision_repr)
        option_repr = norm(option_repr)

        """
        Shared query / option projections
        no task-specific classifier
        """
        query_proj = nn.Dense(d_head, use_bias=False, name="query_projection")
        option_proj = nn.Dense(d_head, use_bias=False, name="option_projection")

        q = query_proj(decision_repr)
        k = option_proj(option_repr)

        # All classes scored simultaneously
        logits = jnp.einsum("rbd,rbod->rbo", q, k)
        logits = logits / math.sqrt(d_head)

        # Mask padded option slots
        logits = jnp.where(
            option_valid[None, ...], logits, jnp.asarray(-1e30, dtype=logits.dtype)
        )

        return logits
