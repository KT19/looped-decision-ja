import asyncio
import hashlib
import json
import math
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from synthetic_judging.config import JudgeConfig
from synthetic_judging.models import (
    FailedJudgment,
    JudgeSummary,
    JudgingProvenance,
    KindAccumulator,
    KindSummary,
    QuestionJudgment,
    SyntheticQuestion,
    SyntheticScenario,
    VoteResponse,
)
from synthetic_judging.prompts import SYSTEM_PROMPT, make_judge_prompt


def stable_int(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31 - 1)


def prepare_options(question: SyntheticQuestion, seed: int) -> tuple[list[str], list[int]]:
    """Return displayed options and their displayed-to-canonical index mapping."""
    mapping = list(range(len(question.options)))
    if question.type != "ordinal":
        random.Random(seed).shuffle(mapping)

    displayed = [question.options[canonical_index] for canonical_index in mapping]
    return displayed, mapping


def response_schema(num_options: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "choice_index": {
                "type": "integer",
                "enum": list(range(num_options)),
            }
        },
        "required": ["choice_index"],
        "additionalProperties": False,
    }


async def request_vote(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: JudgeConfig,
    state: str,
    question: SyntheticQuestion,
    vote_seed: int,
) -> int:
    """Return the selected index in the original option order."""
    displayed_options, displayed_to_canonical = prepare_options(question, vote_seed)
    prompt = make_judge_prompt(state, question.question, displayed_options)
    schema = response_schema(len(displayed_options))
    last_error: Exception | None = None

    for attempt in range(config.max_retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=config.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=config.temperature,
                    seed=vote_seed + attempt,
                    max_completion_tokens=config.max_completion_tokens,
                    reasoning_effort="medium" if config.think else "none",
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "decision_vote",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                )

            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("empty judge response")

            vote = VoteResponse.from_dict(json.loads(content))
            if not 0 <= vote.choice_index < len(displayed_to_canonical):
                raise ValueError("choice index outside option range")
            return displayed_to_canonical[vote.choice_index]

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt + 1 < config.max_retries:
                await asyncio.sleep(min(2**attempt, 8))

    assert last_error is not None
    raise RuntimeError(f"judge vote failed after {config.max_retries} attempts") from last_error


def normalized_entropy(probabilities: list[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0

    entropy = -sum(probability * math.log(probability) for probability in probabilities if probability > 0.0)
    return entropy / math.log(len(probabilities))


async def judge_question(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: JudgeConfig,
    scenario_id: str,
    state: str,
    question_index: int,
    question: SyntheticQuestion,
) -> QuestionJudgment:
    vote_tasks = []
    for vote_index in range(config.votes):
        vote_seed = stable_int(f"{scenario_id}:q{question_index}:vote{vote_index}", config.seed)
        vote_tasks.append(request_vote(client, semaphore, config, state, question, vote_seed))

    results = await asyncio.gather(*vote_tasks, return_exceptions=True)
    votes: list[int] = []
    errors: list[str] = []

    for result in results:
        if isinstance(result, BaseException):
            errors.append(repr(result))
        else:
            votes.append(int(result))

    if len(votes) != config.votes:
        raise RuntimeError(f"not all judge votes completed {len(votes)}/{config.votes}; errors={errors}")

    counts = [0 for _ in question.options]
    for vote in votes:
        counts[vote] += 1

    distribution = [count / len(votes) for count in counts]
    maximum_count = max(counts)
    majority_indices = [index for index, count in enumerate(counts) if count == maximum_count]
    majority_index = majority_indices[0] if len(majority_indices) == 1 else None
    generator_support = distribution[question.answer_index]

    return QuestionJudgment(
        num_votes=len(votes),
        votes=votes,
        counts=counts,
        target_distribution=distribution,
        agreement=maximum_count / len(votes),
        normalized_entropy=normalized_entropy(distribution),
        majority_indices=majority_indices,
        majority_index=majority_index,
        generator_answer_index=question.answer_index,
        generator_support=generator_support,
        generator_in_majority=question.answer_index in majority_indices,
    )


async def judge_scenario(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: JudgeConfig,
    scenario: SyntheticScenario,
) -> SyntheticScenario:
    question_tasks = [
        judge_question(client, semaphore, config, scenario.id, scenario.state, index, question)
        for index, question in enumerate(scenario.questions)
    ]
    judgments = await asyncio.gather(*question_tasks)
    judged_questions = [
        replace(question, judge=judgment)
        for question, judgment in zip(scenario.questions, judgments, strict=True)
    ]

    return replace(
        scenario,
        questions=judged_questions,
        judging_provenance=JudgingProvenance(
            judge_model=config.model,
            votes_per_question=config.votes,
            temperature=config.temperature,
            think=config.think,
            seed=config.seed,
        ),
    )


def append_scenario(path: Path, scenario: SyntheticScenario) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(scenario.to_dict(), ensure_ascii=False))
        file.write("\n")


def append_failure(path: Path, failure: FailedJudgment) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(failure), ensure_ascii=False))
        file.write("\n")


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {scenario.id for scenario in load_scenarios(path)}


def load_scenarios(path: Path) -> list[SyntheticScenario]:
    scenarios = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not (line := line.strip()):
                continue
            try:
                scenarios.append(SyntheticScenario.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"invalid scenario at {path}:{line_number}") from exc
    return scenarios


def summarize(path: Path, config: JudgeConfig) -> JudgeSummary:
    scenarios = load_scenarios(path)
    questions = 0
    unanimous = 0
    high_agreement = 0
    medium_agreement = 0
    low_agreement = 0
    generator_majority = 0
    generator_not_majority = 0
    generator_support_sum = 0.0
    entropy_sum = 0.0
    by_kind: dict[str, KindAccumulator] = {}

    for scenario in scenarios:
        accumulator = by_kind.setdefault(scenario.scenario_kind, KindAccumulator())
        for question in scenario.questions:
            if question.judge is None:
                raise RuntimeError(f"scenario {scenario.id} contains an unjudged question")

            judgment = question.judge
            questions += 1
            generator_support_sum += judgment.generator_support
            entropy_sum += judgment.normalized_entropy

            if judgment.agreement >= 0.999:
                unanimous += 1
            elif judgment.agreement >= 0.8:
                high_agreement += 1
            elif judgment.agreement >= 0.6:
                medium_agreement += 1
            else:
                low_agreement += 1

            if judgment.generator_in_majority:
                generator_majority += 1
            else:
                generator_not_majority += 1

            accumulator.add(judgment)

    kind_summaries = {
        kind: KindSummary(
            questions=values.questions,
            mean_agreement=values.agreement_sum / values.questions,
            mean_generator_support=values.generator_support_sum / values.questions,
        )
        for kind, values in by_kind.items()
        if values.questions > 0
    }

    return JudgeSummary(
        scenarios=len(scenarios),
        questions=questions,
        unanimous=unanimous,
        high_agreement=high_agreement,
        medium_agreement=medium_agreement,
        low_agreement=low_agreement,
        generator_in_majority=generator_majority,
        generator_not_in_majority=generator_not_majority,
        mean_generator_support=generator_support_sum / max(questions, 1),
        mean_normalized_entropy=entropy_sum / max(questions, 1),
        by_scenario_kind=kind_summaries,
        judge_model=config.model,
        votes_per_question=config.votes,
        temperature=config.temperature,
        think=config.think,
        seed=config.seed,
    )


async def run_judging(config: JudgeConfig) -> None:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.failed_path.parent.mkdir(parents=True, exist_ok=True)
    config.stats_path.parent.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios(config.input_path)
    completed_ids = load_completed_ids(config.output_path)
    print(f"input scenarios: {len(scenarios)}")
    print(f"already judged: {len(completed_ids)}")
    print(f"votes / question: {config.votes}")
    print(f"request concurrency: {config.request_concurrency}")
    print(f"thinking: {config.think}")

    client = AsyncOpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.request_timeout,
    )
    semaphore = asyncio.Semaphore(config.request_concurrency)
    accepted = len(completed_ids)
    failed = 0

    try:
        for scenario_index, scenario in enumerate(scenarios):
            if scenario.id in completed_ids:
                continue

            try:
                judged = await judge_scenario(client, semaphore, config, scenario)
                append_scenario(config.output_path, judged)
                completed_ids.add(scenario.id)
                accepted += 1
            except Exception as exc:  # noqa: BLE001
                append_failure(
                    config.failed_path,
                    FailedJudgment(id=scenario.id, scenario_index=scenario_index, error=repr(exc)),
                )
                failed += 1

            print(f"\rjudged={accepted}/{len(scenarios)} failed={failed}", end="", flush=True)
    finally:
        await client.close()

    print()
    if config.output_path.exists():
        summary = summarize(config.output_path, config)
        config.stats_path.write_text(
            json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print()
        print(f"scenarios: {summary.scenarios} questions: {summary.questions}")
