from dataclasses import asdict, dataclass, field
from typing import Any

from synthetic_generation.schema import Domain, QuestionType, ScenarioKind


@dataclass(frozen=True)
class GenerationProvenance:
    teacher_model: str
    seed: int
    generated_at: str
    generator_version: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationProvenance":
        return cls(
            teacher_model=str(data["teacher_model"]),
            seed=int(data["seed"]),
            generated_at=str(data["generated_at"]),
            generator_version=str(data["generator_version"]),
        )


@dataclass(frozen=True)
class QuestionJudgment:
    num_votes: int
    votes: list[int]
    counts: list[int]
    target_distribution: list[float]
    agreement: float
    normalized_entropy: float
    majority_indices: list[int]
    majority_index: int | None
    generator_answer_index: int
    generator_support: float
    generator_in_majority: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuestionJudgment":
        majority_index = data.get("majority_index", data.get("mojority_index"))
        return cls(
            num_votes=int(data["num_votes"]),
            votes=[int(value) for value in data["votes"]],
            counts=[int(value) for value in data["counts"]],
            target_distribution=[float(value) for value in data["target_distribution"]],
            agreement=float(data["agreement"]),
            normalized_entropy=float(data["normalized_entropy"]),
            majority_indices=[int(value) for value in data["majority_indices"]],
            majority_index=None if majority_index is None else int(majority_index),
            generator_answer_index=int(data["generator_answer_index"]),
            generator_support=float(data["generator_support"]),
            generator_in_majority=bool(data["generator_in_majority"]),
        )


@dataclass(frozen=True)
class SyntheticQuestion:
    type: QuestionType
    question: str
    options: list[str]
    answer_index: int
    reference_reason: str
    judge: QuestionJudgment | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SyntheticQuestion":
        judgment = data.get("judge")
        return cls(
            type=data["type"],
            question=str(data["question"]),
            options=[str(option) for option in data["options"]],
            answer_index=int(data["answer_index"]),
            reference_reason=str(data["reference_reason"]),
            judge=None if judgment is None else QuestionJudgment.from_dict(judgment),
        )


@dataclass(frozen=True)
class JudgingProvenance:
    judge_model: str
    votes_per_question: int
    temperature: float
    think: bool
    seed: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JudgingProvenance":
        return cls(
            judge_model=str(data["judge_model"]),
            votes_per_question=int(data["votes_per_question"]),
            temperature=float(data["temperature"]),
            think=bool(data.get("think", False)),
            seed=int(data["seed"]),
        )


@dataclass(frozen=True)
class SyntheticScenario:
    id: str
    generation_index: int
    domain: Domain
    scenario_kind: ScenarioKind
    difficulty: int
    state: str
    questions: list[SyntheticQuestion]
    provenance: GenerationProvenance
    judging_provenance: JudgingProvenance | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SyntheticScenario":
        judging = data.get("judging_provenance")
        return cls(
            id=str(data["id"]),
            generation_index=int(data["generation_index"]),
            domain=data["domain"],
            scenario_kind=data["scenario_kind"],
            difficulty=int(data["difficulty"]),
            state=str(data["state"]),
            questions=[SyntheticQuestion.from_dict(question) for question in data["questions"]],
            provenance=GenerationProvenance.from_dict(data["provenance"]),
            judging_provenance=None if judging is None else JudgingProvenance.from_dict(judging),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.judging_provenance is None:
            result.pop("judging_provenance")
        for question, serialized in zip(self.questions, result["questions"], strict=True):
            if question.judge is None:
                serialized.pop("judge")
        return result


@dataclass(frozen=True)
class VoteResponse:
    choice_index: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoteResponse":
        return cls(choice_index=int(data["choice_index"]))


@dataclass(frozen=True)
class FailedJudgment:
    id: str
    scenario_index: int
    error: str


@dataclass
class KindAccumulator:
    questions: int = 0
    agreement_sum: float = 0.0
    generator_support_sum: float = 0.0

    def add(self, judgment: QuestionJudgment) -> None:
        self.questions += 1
        self.agreement_sum += judgment.agreement
        self.generator_support_sum += judgment.generator_support


@dataclass(frozen=True)
class KindSummary:
    questions: int
    mean_agreement: float
    mean_generator_support: float


@dataclass(frozen=True)
class JudgeSummary:
    scenarios: int
    questions: int
    unanimous: int
    high_agreement: int
    medium_agreement: int
    low_agreement: int
    generator_in_majority: int
    generator_not_in_majority: int
    mean_generator_support: float
    mean_normalized_entropy: float
    by_scenario_kind: dict[str, KindSummary] = field(default_factory=dict)
    judge_model: str = ""
    votes_per_question: int = 0
    temperature: float = 0.0
    think: bool = False
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
