from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from models.language_model import LanguageModel


def sample_next_token(
    logits: jax.Array, key: jax.Array, temperature: float = 0.8, top_k: int = 40
) -> jax.Array:
    logits = logits.astype(jnp.float32)

    if temperature <= 0.0:
        return jnp.argmax(logits, axis=-1)

    logits = logits / temperature

    vocab_size = logits.shape[-1]

    if top_k > 0 and top_k < vocab_size:
        values, _ = jax.lax.top_k(logits, top_k)

        threshold = values[..., -1:]

        logits = jnp.where(logits < threshold, -jnp.inf, logits)

    return jax.random.categorical(key, logits, axis=-1)


class JaxGenerator:
    def __init__(
        self,
        model: LanguageModel,
        tokenizer: Any,
        max_seq_len: int,
        temperature: float = 0.8,
        top_k: int = 40,
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.top_k = top_k

        @jax.jit
        def forward(params: Any, tokens: jax.Array):
            logits, _ = model.apply(
                {"params": params},
                tokens,
            )

            return logits

        self.forward = forward

        @jax.jit
        def sample(logits: jax.Array, key: jax.Array):
            return sample_next_token(
                logits=logits, key=key, temperature=temperature, top_k=top_k
            )

        self.sample = sample

    def warmup(self, params: Any):
        dummy = jnp.zeros((1, self.max_seq_len), dtype=jnp.int32)
        logits = self.forward(params, dummy)

        logits.block_until_ready()

    def generate(
        self, params: Any, prompt: str, max_new_tokens: int = 128, seed: int = 0
    ):
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        if not token_ids:
            raise ValueError("Prompt tokenized to zero tokens.")

        key = jax.random.PRNGKey(seed)

        prompt_ids = list(token_ids)
        prompt_ids = prompt_ids[-self.max_seq_len :]

        prompt_len = len(prompt_ids)
        buffer = np.zeros(self.max_seq_len, dtype=np.int32)
        buffer[:prompt_len] = np.asarray(prompt_ids, dtype=np.int32)

        generated_len = prompt_len

        max_steps = min(max_new_tokens, self.max_seq_len - prompt_len)

        eos_id = self.tokenizer.eos_token_id

        for _ in range(max_steps):
            tokens = jnp.asarray(buffer[None, :])
            logits = self.forward(params, tokens)

            position = generated_len - 1
            next_logits = logits[0, position, :]

            key, sample_key = jax.random.split(key)
            next_token = self.sample(next_logits, sample_key)

            next_token = int(jax.device_get(next_token))
            buffer[generated_len] = next_token

            generated_len += 1

            if generated_len >= self.max_seq_len:
                break

            if next_token == eos_id:
                break

        result = buffer[:generated_len].tolist()

        return self.tokenizer.decode(result, skip_special_tokens=True)
