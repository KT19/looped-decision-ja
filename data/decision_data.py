import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray


class Tokenizer(Protocol):
    eos_token_id: int | None
    pad_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class PreparedExample(TypedDict):
    raw: dict[str, Any]
    state_tokens: list[int]
    question_tokens: list[int]
    option_tokens: list[list[int]]


class SerializedDecision(TypedDict):
    input_ids: NDArray[np.int32]
    option_mask: NDArray[np.float32]
    option_valid: NDArray[np.bool_]
    decision_index: np.int32
    input_length: np.int32


class SerializedRow(SerializedDecision):
    targets: NDArray[np.float32]
    hard_label: np.int32


@dataclass(frozen=True)
class DecisionBatch:
    input_ids: NDArray[np.int32]
    option_mask: NDArray[np.float32]
    option_valid: NDArray[np.bool_]
    decision_index: NDArray[np.int32]
    targets: NDArray[np.float32]
    hard_labels: NDArray[np.int32]
    sample_weight: NDArray[np.float32]
    input_lengths: NDArray[np.int32]
    sources: tuple[str, ...]


def truncate_middle(tokens: list[int], max_length: int) -> list[int]:
    if len(tokens) <= max_length:
        return tokens
    if max_length <= 0:
        return []

    left = max_length // 2
    right = max_length - left
    return tokens[:left] + tokens[-right:]


def make_token_mask(input_lengths: NDArray[np.int32], max_seq_len: int) -> NDArray[np.bool_]:
    positions = np.arange(max_seq_len, dtype=np.int32)[None, :]
    return positions < np.asarray(input_lengths, dtype=np.int32)[:, None]


class DecisionSerializer:
    """Shared text format used by decision training, evaluation, and inference."""

    def __init__(self, tokenizer: Tokenizer, max_seq_len: int, max_options: int = 6) -> None:
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if max_options < 2:
            raise ValueError("max_options must be at least 2")
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs EOS.")

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_options = max_options
        self.eos_id = int(tokenizer.eos_token_id)
        self.pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else self.eos_id
        self.state_header = self.encode("状態:\n")
        self.question_header = self.encode("\n\n質問:\n")
        self.options_header = self.encode("\n\n選択肢:\n")
        self.decision_tail = self.encode("\n判断:\n") + [self.eos_id]
        self.newline = self.encode("\n")
        self.option_prefix = {index: self.encode(f"{index + 1}. ") for index in range(max_options)}

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def serialize(self, state: str, question: str, options: list[str]) -> SerializedDecision:
        return self.serialize_tokens(
            state_tokens=self.encode(state),
            question_tokens=self.encode(question)[:256],
            option_tokens=[self.encode(option)[:128] for option in options],
        )

    def serialize_tokens(
        self,
        state_tokens: list[int],
        question_tokens: list[int],
        option_tokens: list[list[int]],
    ) -> SerializedDecision:
        option_count = len(option_tokens)
        if not 2 <= option_count <= self.max_options:
            raise ValueError(f"decision must have between 2 and {self.max_options} options")

        fixed_length = (
            len(self.state_header)
            + len(self.question_header)
            + len(question_tokens)
            + len(self.options_header)
            + len(self.decision_tail)
        )
        for index, option in enumerate(option_tokens):
            fixed_length += len(self.option_prefix[index]) + len(option) + len(self.newline)

        state_budget = self.max_seq_len - fixed_length
        if state_budget < 0:
            raise ValueError("question/options exceed max_seq_len")

        tokens: list[int] = []
        option_mask = np.zeros((self.max_options, self.max_seq_len), dtype=np.float32)
        tokens.extend(self.state_header)
        tokens.extend(truncate_middle(state_tokens, state_budget))
        tokens.extend(self.question_header)
        tokens.extend(question_tokens)
        tokens.extend(self.options_header)

        for index, option in enumerate(option_tokens):
            tokens.extend(self.option_prefix[index])
            start = len(tokens)
            tokens.extend(option)
            option_mask[index, start : len(tokens)] = 1.0
            tokens.extend(self.newline)

        tokens.extend(self.decision_tail)
        input_length = len(tokens)
        if input_length > self.max_seq_len:
            raise RuntimeError("serialization overflow")

        input_ids = np.full(self.max_seq_len, self.pad_id, dtype=np.int32)
        input_ids[:input_length] = np.asarray(tokens, dtype=np.int32)
        option_valid = np.zeros(self.max_options, dtype=np.bool_)
        option_valid[:option_count] = True

        return {
            "input_ids": input_ids,
            "option_mask": option_mask,
            "option_valid": option_valid,
            "decision_index": np.int32(input_length - 1),
            "input_length": np.int32(input_length),
        }


class DecisionDataset:
    def __init__(
        self,
        path: str | Path,
        tokenizer: Tokenizer,
        max_seq_len: int,
        max_options: int = 6,
        seed: int = 42,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_options = max_options
        self.seed = seed

        self.serializer = DecisionSerializer(tokenizer, max_seq_len, max_options)

        self.examples: list[PreparedExample] = []
        with self.path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not (line := line.strip()):
                    continue

                try:
                    raw: dict[str, Any] = json.loads(line)
                    options = list(raw["options"])
                    targets = [float(value) for value in raw["target_distribution"]]
                    hard_label = int(raw["hard_label"])
                    source = str(raw["source"]).strip()
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(f"invalid decision example at {self.path}:{line_number}") from exc

                if not 2 <= len(options) <= max_options:
                    raise ValueError(f"{raw['id']} must have between 2 and {max_options} options")
                if len(targets) != len(options):
                    raise ValueError(f"{raw['id']} target length does not match its options")
                if not 0 <= hard_label < len(options):
                    raise ValueError(f"{raw['id']} hard label is outside its option range")
                if not source:
                    raise ValueError(f"{raw['id']} has an empty source")

                self.examples.append(
                    {
                        "raw": raw,
                        "state_tokens": self.serializer.encode(str(raw["state"])),
                        "question_tokens": self.serializer.encode(str(raw["question"]))[:256],
                        "option_tokens": [self.serializer.encode(str(option))[:128] for option in options],
                    }
                )

        if not self.examples:
            raise ValueError(f"decision dataset is empty: {self.path}")

        print(f"{self.path.name}: {len(self.examples)} examples")

    def __len__(self) -> int:
        return len(self.examples)

    def serialize(self, prepared: PreparedExample, rng: random.Random | None) -> SerializedRow:
        raw = prepared["raw"]
        option_count = len(raw["options"])
        permutation = list(range(option_count))

        if rng is not None and raw["type"] != "ordinal":
            rng.shuffle(permutation)

        option_tokens = [prepared["option_tokens"][index] for index in permutation]
        targets = [float(raw["target_distribution"][index]) for index in permutation]
        hard_label = permutation.index(int(raw["hard_label"]))

        serialized = self.serializer.serialize_tokens(
            prepared["state_tokens"], prepared["question_tokens"], option_tokens
        )

        target_array = np.zeros(self.max_options, dtype=np.float32)
        target_array[:option_count] = np.asarray(targets, dtype=np.float32)

        return {
            **serialized,
            "targets": target_array,
            "hard_label": np.int32(hard_label),
        }

    def batches(
        self,
        batch_size: int,
        *,
        epoch: int,
        shuffle: bool,
        shuffle_options: bool,
    ) -> Iterator[DecisionBatch]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        indices = list(range(len(self.examples)))
        if shuffle:
            random.Random(self.seed + epoch).shuffle(indices)

        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start : batch_start + batch_size]
            sample_weight = np.ones(len(batch_indices), dtype=np.float32)

            if len(batch_indices) < batch_size:
                missing = batch_size - len(batch_indices)
                sample_weight = np.concatenate([sample_weight, np.zeros(missing, dtype=np.float32)])
                batch_indices += [batch_indices[0]] * missing

            rows: list[SerializedRow] = []
            for local_index, example_index in enumerate(batch_indices):
                example_rng = None
                if shuffle_options:
                    example_rng = random.Random(self.seed + epoch * 1_000_003 + example_index * 97 + local_index)
                rows.append(self.serialize(self.examples[example_index], example_rng))

            yield DecisionBatch(
                input_ids=np.stack([row["input_ids"] for row in rows]),
                option_mask=np.stack([row["option_mask"] for row in rows]),
                option_valid=np.stack([row["option_valid"] for row in rows]),
                decision_index=np.asarray([row["decision_index"] for row in rows], dtype=np.int32),
                targets=np.stack([row["targets"] for row in rows]),
                hard_labels=np.asarray([row["hard_label"] for row in rows], dtype=np.int32),
                sample_weight=sample_weight,
                input_lengths=np.asarray([row["input_length"] for row in rows], dtype=np.int32),
                sources=tuple(str(self.examples[index]["raw"]["source"]) for index in batch_indices),
            )
