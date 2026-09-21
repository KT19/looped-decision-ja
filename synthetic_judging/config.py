from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class JudgeConfig:
    input_path: Path
    output_path: Path
    failed_path: Path
    stats_path: Path
    base_url: str
    api_key: str
    model: str
    votes: int = 5
    request_concurrency: int = 2
    temperature: float = 0.8
    think: bool = False
    seed: int = 12345
    max_retries: int = 3
    max_completion_tokens: int = 128
    request_timeout: float = 60.0


def load_judge_config(path: str | Path) -> JudgeConfig:
    with Path(path).open(encoding="utf-8") as file:
        document: dict[str, Any] = yaml.safe_load(file)

    config = document["judge"]
    return JudgeConfig(
        input_path=Path(config["input_path"]),
        output_path=Path(config["output_path"]),
        failed_path=Path(config["failed_path"]),
        stats_path=Path(config["stats_path"]),
        base_url=str(config["base_url"]),
        api_key=str(config["api_key"]),
        model=str(config["model"]),
        votes=int(config.get("votes", 5)),
        request_concurrency=int(config.get("request_concurrency", 2)),
        temperature=float(config.get("temperature", 0.8)),
        think=bool(config.get("think", False)),
        seed=int(config.get("seed", 12345)),
        max_retries=int(config.get("max_retries", 3)),
        max_completion_tokens=int(config.get("max_completion_tokens", 128)),
        request_timeout=float(config.get("request_timeout", 60.0)),
    )
