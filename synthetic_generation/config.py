from dataclasses import dataclass
from pathlib import Path

import yaml

DOMAINS = (
    "customer_support",
    "business_operations",
    "logistics",
    "education",
    "content_policy",
    "workplace",
    "public_services",
    "general_reasoning",
    "fact_assessment",
    "culture",
    "city",
    "daily_life",
    "interpersonal",
    "personal_finance",
    "health_and_safety",
    "planning",
    "privacy_and_security",
    "ethics_and_fairness",
)

SCENARIO_KINDS = ("clear", "borderline", "insufficient")
SCENARIO_WEIGHTS = (0.55, 0.30, 0.15)
QUESTION_TYPES = ("choice", "binary", "ordinal")


@dataclass(frozen=True)
class GeneratorConfig:
    base_url: str
    api_key: str
    model: str
    output_dir: str
    target_count: int = 1000
    concurrency: int = 4
    questions_per_state: int = 3
    temperature: float = 0.9
    think: bool = False
    max_tokens: int = 2500
    seed: int = 42
    max_attempt_multiplier: int = 4
    request_timeout: float = 120.0


def load_generator_config(path: str | Path) -> GeneratorConfig:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return GeneratorConfig(**config["generator"])
