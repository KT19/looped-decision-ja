import hashlib
import json
import math
from pathlib import Path
from typing import Any

from decision_corpus.schema import DecisionExample, DecisionSplit

JNLI_OPTIONS = ["含意される", "矛盾する", "どちらとも断定できない"]
JNLI_LABELS = {"entailment": 0, "contradiction": 1, "neutral": 2}
JSTS_OPTIONS = [
    "0: 意味が全く異なる",
    "1: ほとんど意味が異なる",
    "2: 一部だけ意味が似ている",
    "3: ある程度意味が似ている",
    "4: ほぼ同じ意味である",
    "5: 意味が完全に一致する",
]


def one_hot(index: int, size: int) -> list[float]:
    result = [0.0 for _ in range(size)]
    result[index] = 1.0
    return result


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    value_int = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value_int / float(2**64)


def convert_jnli(records: list[dict[str, Any]], split: DecisionSplit) -> list[DecisionExample]:
    result = []
    for row in records:
        target = JNLI_LABELS[row["label"]]
        pair_id = str(row["sentence_pair_id"])
        result.append(
            DecisionExample(
                id=f"jnli:{split}:{pair_id}",
                source="jnli",
                split=split,
                domain="natural_language_inference",
                state=f"前提:\n{row['sentence1']}\n\n仮説:\n{row['sentence2']}",
                question="前提と仮説の関係として最も適切なものを選んでください。",
                type="choice",
                options=JNLI_OPTIONS.copy(),
                target_distribution=one_hot(target, len(JNLI_OPTIONS)),
                hard_label=target,
                metadata={"original_label": row["label"], "yjcaptions_id": row.get("yjcaptions_id")},
            )
        )
    return result


def convert_jcommonsenseqa(records: list[dict[str, Any]], split: DecisionSplit) -> list[DecisionExample]:
    result = []
    for row in records:
        options = [row[f"choice{index}"] for index in range(5)]
        target = int(row["label"])
        result.append(
            DecisionExample(
                id=f"jcommonsenseqa:{split}:{row['q_id']}",
                source="jcommonsenseqa",
                split=split,
                domain="commonsense_reasoning",
                state=row["question"],
                question="最も適切な答えを選んでください",
                type="choice",
                options=options,
                target_distribution=one_hot(target, len(options)),
                hard_label=target,
                metadata={},
            )
        )
    return result


def jsts_distribution(score: float) -> list[float]:
    score = float(max(0.0, min(5.0, score)))
    lower = math.floor(score)
    upper = math.ceil(score)
    probabilities = [0.0 for _ in range(6)]

    if lower == upper:
        probabilities[lower] = 1.0
        return probabilities

    probabilities[lower] = 1.0 - (score - lower)
    probabilities[upper] = score - lower
    return probabilities


def convert_jsts(records: list[dict[str, Any]], split: DecisionSplit) -> list[DecisionExample]:
    result = []
    for row in records:
        score = float(row["label"])
        hard_label = min(5, max(0, math.floor(score + 0.5)))
        result.append(
            DecisionExample(
                id=f"jsts:{split}:{row['sentence_pair_id']}",
                source="jsts",
                split=split,
                domain="semantic_similarity",
                state=f"文1:\n{row['sentence1']}\n\n文2:\n{row['sentence2']}",
                question="2つの文の意味の類似度として最も適切な値を選んでください。",
                type="ordinal",
                options=JSTS_OPTIONS.copy(),
                target_distribution=jsts_distribution(score),
                hard_label=hard_label,
                metadata={"ordinal_score": score, "yjcaptions_id": row.get("yjcaptions_id")},
            )
        )
    return result


def convert_jcola(records: list[dict[str, Any]], distribution_name: str) -> list[DecisionExample]:
    result = []
    for row in records:
        target = int(row["label"])
        result.append(
            DecisionExample(
                id=f"jcola:{distribution_name}:{row['uid']}",
                source="jcola",
                split="eval",
                domain="linguistic_acceptability",
                state=row["sentence"],
                question="この日本語文は自然ですか?",
                type="binary",
                options=["不自然である", "自然である"],
                target_distribution=one_hot(target, 2),
                hard_label=target,
                metadata={
                    "distribution": distribution_name,
                    "source_reference": row.get("source"),
                    "diacritic": row.get("diacritic"),
                },
            )
        )
    return result


def convert_synthetic(
    path: Path,
    eval_fraction: float,
    seed: int,
) -> tuple[list[DecisionExample], list[DecisionExample]]:
    train_examples = []
    eval_examples = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if not (line := line.strip()):
                continue

            scenario = json.loads(line)
            scenario_id = str(scenario["id"])
            judging_provenance = scenario.get("judging_provenance", {})
            is_eval = stable_fraction(scenario_id, seed) < eval_fraction
            split: DecisionSplit = "eval" if is_eval else "train"

            for question_index, question in enumerate(scenario["questions"]):
                options = list(question["options"])

                generator_target = int(question["answer_index"])
                judge = question.get("judge")
                if judge is not None:
                    target_distribution = [float(x) for x in judge["target_distribution"]]
                    majority_indices = [int(index) for index in judge["majority_indices"]]
                    majority_index = judge.get("majority_index")

                    if majority_index is not None:
                        hard_label = int(majority_index)
                    elif generator_target in majority_indices:
                        hard_label = generator_target
                    else:
                        hard_label = majority_indices[0]
                else:
                    raise RuntimeError("judge is missing")

                example = DecisionExample(
                    id=f"synthetic:{scenario_id}:q{question_index}",
                    source="synthetic_decisions",
                    split=split,
                    domain=scenario["domain"],
                    state=scenario["state"],
                    question=question["question"],
                    type=question["type"],
                    options=options,
                    target_distribution=target_distribution,
                    hard_label=hard_label,
                    metadata={
                        "scenario_id": scenario_id,
                        "scenario_kind": scenario["scenario_kind"],
                        "difficulty": scenario["difficulty"],
                        "generator_answer": generator_target,
                        "judge_agreement": float(judge["agreement"]),
                        "teacher_model": scenario.get("provenance", {}).get("teacher_model"),
                        "generation_seed": scenario.get("provenance", {}).get("seed"),
                        "judge_model": judging_provenance.get("judge_model"),
                        "judge_votes_per_question": judging_provenance.get("votes_per_question"),
                        "judge_temperature": judging_provenance.get("temperature"),
                        "judge_think": judging_provenance.get("think"),
                        "judge_seed": judging_provenance.get("seed"),
                    },
                )
                (eval_examples if is_eval else train_examples).append(example)

    return train_examples, eval_examples
