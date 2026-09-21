import dataclasses
import json
from pathlib import Path
from typing import Any

from configs.config import Config


def config_to_dict(cfg: Config) -> dict[str, Any]:
    if dataclasses.is_dataclass(cfg):
        result = dataclasses.asdict(cfg)
    else:
        raise TypeError("Config must be a dataclass")

    return result


def build_experiment_metadata(cfg: Config, vocab_size: int) -> dict[str, Any]:
    effective_batch_size = cfg.batch_size * cfg.grad_accum_steps
    tokens_per_update = effective_batch_size * cfg.seq_len
    planned_tokens = cfg.steps * tokens_per_update

    metadata = {
        "config": config_to_dict(cfg),
        "tokenizer": {"name": cfg.tokenizer_name, "vocab_size": int(vocab_size)},
        "training": {
            "microbatch_size": int(cfg.batch_size),
            "grad_accum_steps": int(cfg.grad_accum_steps),
            "effective_batch_size": int(effective_batch_size),
            "sequence_length": int(cfg.seq_len),
            "tokens_per_update": int(tokens_per_update),
            "optimizer_steps": int(cfg.steps),
            "planned_tokens": int(planned_tokens),
        },
    }

    return metadata


def write_experiment_metadata(directory: str, metadata: dict[str, Any]) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    metadata_path = path / "experiment.json"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Experiment metadata: {metadata_path}")
