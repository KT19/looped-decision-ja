import json
from pathlib import Path
from typing import Any, ClassVar


class MetricsLogger:
    """Append training and validation metrics as one JSON object per line."""

    _VALID_SPLITS: ClassVar[set[str]] = {"train", "validation"}

    def __init__(self, directory: str, append: bool = False):
        self.path = Path(directory) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not append:
            self.path.write_text("", encoding="utf-8")

    def log(
        self,
        split: str,
        step: int,
        tokens_seen: int,
        metrics: dict[str, Any],
    ) -> None:
        if split not in self._VALID_SPLITS:
            raise ValueError(f"Unsupported metrics split: {split}")

        record = {
            "split": split,
            "step": int(step),
            "tokens_seen": int(tokens_seen),
            **{name: float(value) for name, value in metrics.items()},
        }

        with self.path.open("a", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")

    def log_examples(
        self,
        split: str,
        step: int,
        examples_seen: int,
        metrics: dict[str, Any],
    ) -> None:
        if split not in self._VALID_SPLITS:
            raise ValueError(f"Unsupported metrics split: {split}")

        record = {
            "split": split,
            "step": int(step),
            "examples_seen": int(examples_seen),
            **{name: float(value) for name, value in metrics.items()},
        }

        with self.path.open("a", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")
