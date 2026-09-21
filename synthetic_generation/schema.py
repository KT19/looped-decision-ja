from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Domain = Literal[
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
]

ScenarioKind = Literal["clear", "borderline", "insufficient"]
QuestionType = Literal["choice", "binary", "ordinal"]


class DecisionQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: QuestionType
    question: str = Field(min_length=4, max_length=500)
    options: list[str] = Field(min_length=2, max_length=6)
    answer_index: int = Field(ge=0, le=5)
    reference_reason: str = Field(min_length=4, max_length=1000)


class SyntheticScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: Domain
    scenario_kind: ScenarioKind
    difficulty: int = Field(ge=1, le=5)
    state: str = Field(min_length=40, max_length=6000)
    questions: list[DecisionQuestion] = Field(min_length=1, max_length=6)
