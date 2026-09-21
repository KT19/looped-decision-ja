from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DecisionType = Literal["choice", "binary", "ordinal"]
DecisionSplit = Literal["train", "eval"]


class DecisionExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    split: DecisionSplit
    domain: str
    state: str = Field(min_length=1)
    question: str = Field(min_length=1)
    type: DecisionType
    options: list[str] = Field(min_length=2, max_length=6)
    target_distribution: list[float]
    hard_label: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_targets(self):
        option_count = len(self.options)

        if len(self.target_distribution) != option_count:
            raise ValueError("target_distribution length must match options")
        if not 0 <= self.hard_label < option_count:
            raise ValueError("hard_label out of range")
        if any(probability < 0.0 for probability in self.target_distribution):
            raise ValueError("negative probability")

        total = sum(self.target_distribution)
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"target_distribution must sum to 1, got {total}")

        return self

