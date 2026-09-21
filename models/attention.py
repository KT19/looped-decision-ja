import flax.linen as nn
import jax
import jax.numpy as jnp

from models.config import ModelConfig


class GQAAttention(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        cfg = self.config

        B, T, D = x.shape

        assert cfg.n_heads % cfg.n_kv_heads == 0

        head_dim = D // cfg.n_heads

        q = nn.Dense(cfg.n_heads * head_dim, use_bias=False, name="q_proj")(x)
        k = nn.Dense(cfg.n_kv_heads * head_dim, use_bias=False, name="k_proj")(x)
        v = nn.Dense(cfg.n_kv_heads * head_dim, use_bias=False, name="v_proj")(x)

        q = q.reshape(B, T, cfg.n_heads, head_dim)
        k = k.reshape(B, T, cfg.n_kv_heads, head_dim)
        v = v.reshape(B, T, cfg.n_kv_heads, head_dim)

        # Repeat KV heads for grouped query attention
        repeat = cfg.n_heads // cfg.n_kv_heads

        k = jnp.repeat(k, repeat, axis=2)
        v = jnp.repeat(v, repeat, axis=2)

        # (B, H, T, Dh)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = head_dim**-0.5

        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale

        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        scores = jnp.where(
            causal_mask[None, None, :, :], scores, jnp.finfo(scores.dtype).min
        )

        attn = jax.nn.softmax(scores, axis=-1)

        out = jnp.einsum("bhts,bhsd->bhtd", attn, v)

        out = out.transpose(0, 2, 1, 3)
        out = out.reshape(B, T, D)

        return nn.Dense(D, use_bias=False, name="o_proj")(out)
