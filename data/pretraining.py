from dataclasses import dataclass
from typing import Any

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass
class DataState:
    documents_consumed: int
    token_buffer: list[int]


class PackedPretrainingDataset:
    """
    Stateful streaming + packing data loader.
    """

    def __init__(
        self,
        tokenizer_name: str,
        seq_len: int,
        batch_size: int,
        dataset_name: str = "hotchpotch/fineweb-2-edu-japanese",
        dataset_config: str = "sample_10BT",
        split: str = "train",
        shuffle: bool = True,
        shuffle_buffer: int = 10000,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token.")

        self.eos_id = self.tokenizer.eos_token_id

        self.dataset_name = dataset_name
        self.dataset_config = dataset_config

        self.split = split
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        # stateful fields
        self.documents_consumed = 0
        self.token_buffer: list[int] = []
        self._build_stream()

    def _build_stream(self) -> None:
        dataset = load_dataset(self.dataset_name, self.dataset_config, split=self.split, streaming=True)

        if self.shuffle:
            dataset = dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)

        self.dataset = dataset
        self.iterator = iter(dataset)

    def _next_document_tokens(self) -> list[int]:
        """
        Fetch the next usable document and tokenize it.
        """
        while True:
            example = next(self.iterator)

            # count source example
            self.documents_consumed += 1

            text = example.get("text", None)
            if not text:
                continue

            text = text.strip()
            if not text:
                continue

            token_ids = self.tokenizer.encode(text, add_special_tokens=False)

            if not token_ids:
                continue

            token_ids.append(self.eos_id)

            return token_ids

    def _fill_buffer(self, required_tokens: int) -> None:
        while len(self.token_buffer) < required_tokens:
            document_tokens = self._next_document_tokens()

            self.token_buffer.extend(document_tokens)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Produce:
            x: [B, T]
            y: [B, T]
        """

        sequence_length = self.seq_len + 1
        total_needed = self.batch_size * sequence_length

        self._fill_buffer(total_needed)
        flat = np.asarray(self.token_buffer[:total_needed], dtype=np.int32)

        # remove consumed tokens while retaining any remainder from the final soure document.
        del self.token_buffer[:total_needed]

        sequences = flat.reshape(self.batch_size, sequence_length)

        x = sequences[:, :-1]
        y = sequences[:, 1:]

        return x, y

    # checkpoint state
    def state_dict(self) -> dict[str, Any]:
        return {
            "documents_consumed": int(self.documents_consumed),
            "token_buffer": list(self.token_buffer),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        documents_consumed = int(state["documents_consumed"])
        token_buffer = [int(x) for x in state["token_buffer"]]

        print("restoring data stream...")

        print(f"replay documents: {documents_consumed}")
        print(f"buffered tokens: {len(token_buffer):,}")

        # Rebuild stream from exactly the same seed.
        self.documents_consumed = 0
        self.token_buffer = []

        self._build_stream()

        # Replay the exact source-example count
        for _ in range(documents_consumed):
            next(self.iterator)
            self.documents_consumed += 1

        self.token_buffer = token_buffer
        print("data stream restored.")
