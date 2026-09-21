import asyncio
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import ValidationError

from synthetic_generation.config import (
    DOMAINS,
    QUESTION_TYPES,
    SCENARIO_KINDS,
    SCENARIO_WEIGHTS,
    GeneratorConfig,
)
from synthetic_generation.prompts import SYSTEM_PROMPT, make_generation_prompt
from synthetic_generation.schema import SyntheticScenario

JAPANESE_RE = re.compile(r"[ぁ-んァ-ヶ一-龠々ー]")


@dataclass(frozen=True)
class RequestPlan:
    seed: int
    domain: str
    scenario_kind: str
    difficulty: int
    question_types: list[str]


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def japanese_char_count(text: str) -> int:
    return len(JAPANESE_RE.findall(text))


def scenario_hash(state: str) -> str:
    return hashlib.sha256(normalize_text(state).lower().encode("utf-8")).hexdigest()


def make_id(scenario: SyntheticScenario) -> str:
    payload = {
        "domain": scenario.domain,
        "state": normalize_text(scenario.state),
        "questions": [
            {
                "question": normalize_text(question.question),
                "options": [normalize_text(option) for option in question.options],
            }
            for question in scenario.questions
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_question_types(rng: random.Random, count: int) -> list[str]:
    """Keep question types approximately balanced."""
    result: list[str] = []
    while len(result) < count:
        block = list(QUESTION_TYPES)
        rng.shuffle(block)
        result.extend(block)
    return result[:count]


def validate_scenario(
    scenario: SyntheticScenario,
    requested_domain: str,
    requested_kind: str,
    requested_difficulty: int,
    requested_types: list[str],
) -> list[str]:
    errors: list[str] = []

    if scenario.domain != requested_domain:
        errors.append("domain_mismatch")
    if scenario.scenario_kind != requested_kind:
        errors.append("scenario_kind_mismatch")
    if scenario.difficulty != requested_difficulty:
        errors.append("difficulty_mismatch")
    if len(scenario.questions) != len(requested_types):
        errors.append("question_count_mismatch")
        return errors
    if japanese_char_count(scenario.state) < 20:
        errors.append("state_not_sufficiently_japanese")

    forbidden_state_markers = ["正解は", "答えは", "answer_index", "reference_reason"]
    state_lower = scenario.state.lower()
    if any(marker.lower() in state_lower for marker in forbidden_state_markers):
        errors.append("posible_answer_leak")

    for index, (question, expected_type) in enumerate(zip(scenario.questions, requested_types, strict=True)):
        prefix = f"q{index}"
        if question.type != expected_type:
            errors.append(f"{prefix}_type_mismatch")
        if japanese_char_count(question.question) < 2:
            errors.append(f"{prefix}_question_not_japanese")

        normalized_options = [normalize_text(option) for option in question.options]
        if len(set(normalized_options)) != len(normalized_options):
            errors.append(f"{prefix}_duplicate_options")
        if not 0 <= question.answer_index < len(question.options):
            errors.append(f"{prefix}_invalid_answer_index")
        if question.type == "binary" and len(question.options) != 2:
            errors.append(f"{prefix}_binary_not_two_options")
        if question.type == "choice" and not 3 <= len(question.options) <= 6:
            errors.append(f"{prefix}_choice_option_count")
        if question.type == "ordinal" and not 3 <= len(question.options) <= 5:
            errors.append(f"{prefix}_ordinal_option_count")
        if any(len(option) > 300 for option in question.options):
            errors.append(f"{prefix}_option_too_long")
        if japanese_char_count(question.reference_reason) < 2:
            errors.append(f"{prefix}_reason_not_japanese")

    return errors


def shuffle_unordered_options(scenario: SyntheticScenario, seed: int) -> SyntheticScenario:
    """Shuffle non-ordinal options while preserving the correct answer."""
    rng = random.Random(seed)
    shuffled = scenario.model_copy(deep=True)
    for question in shuffled.questions:
        if question.type == "ordinal":
            continue
        permutation = list(range(len(question.options)))
        rng.shuffle(permutation)
        old_answer = question.answer_index
        question.options = [question.options[index] for index in permutation]
        question.answer_index = permutation.index(old_answer)
    return shuffled


def make_request_plan(generation_index: int, master_seed: int, questions_per_state: int) -> RequestPlan:
    seed = master_seed + generation_index * 104729
    rng = random.Random(seed)
    return RequestPlan(
        seed=seed,
        domain=rng.choice(DOMAINS),
        scenario_kind=rng.choices(SCENARIO_KINDS, weights=SCENARIO_WEIGHTS, k=1)[0],
        difficulty=rng.randint(1, 5),
        question_types=build_question_types(rng, questions_per_state),
    )


async def request_scenario(
    client: AsyncOpenAI,
    config: GeneratorConfig,
    generation_index: int,
    semaphore: asyncio.Semaphore,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    plan = make_request_plan(generation_index, config.seed, config.questions_per_state)
    plan_data = asdict(plan)
    prompt = make_generation_prompt(plan.domain, plan.scenario_kind, plan.difficulty, plan.question_types)

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                seed=plan.seed,
                reasoning_effort="medium" if config.think else "none",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "synthetic_decision_scenario",
                        "strict": True,
                        "schema": SyntheticScenario.model_json_schema(),
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001
            return None, {
                "generation_index": generation_index,
                "stage": "request",
                "error": repr(exc),
                "plan": plan_data,
            }

    choice = response.choices[0]
    content = choice.message.content
    if content is None or not content.strip():
        reasoning = getattr(choice.message, "reasoning", None)
        return None, {
            "generation_index": generation_index,
            "stage": "empty_response",
            "finish_reason": choice.finish_reason,
            "reasoning_chars": len(reasoning or ""),
            "plan": plan_data,
        }

    try:
        scenario = SyntheticScenario.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValidationError) as exc:
        return None, {
            "generation_index": generation_index,
            "stage": "schema_parse",
            "error": str(exc),
            "raw": content,
            "plan": plan_data,
        }

    errors = validate_scenario(
        scenario,
        plan.domain,
        plan.scenario_kind,
        plan.difficulty,
        plan.question_types,
    )
    if errors:
        return None, {
            "generation_index": generation_index,
            "stage": "deterministic_validation",
            "errors": errors,
            "scenario": scenario.model_dump(),
            "plan": plan_data,
        }

    scenario = shuffle_unordered_options(scenario, seed=plan.seed + 1)
    record = {
        "id": make_id(scenario),
        "generation_index": generation_index,
        **scenario.model_dump(),
        "provenance": {
            "teacher_model": config.model,
            "seed": plan.seed,
            "generated_at": datetime.now(UTC).isoformat(),
            "generator_version": "decision-synth-v1",
        },
    }
    return record, None


def load_existing_state(accepted_path: Path, rejected_path: Path) -> tuple[int, set[str], int]:
    accepted_count = 0
    hashes: set[str] = set()
    max_index = -1

    if accepted_path.exists():
        with accepted_path.open(encoding="utf-8") as file:
            for line in file:
                if not (line := line.strip()):
                    continue
                item = json.loads(line)
                accepted_count += 1
                hashes.add(scenario_hash(item["state"]))
                max_index = max(max_index, int(item["generation_index"]))

    if rejected_path.exists():
        with rejected_path.open(encoding="utf-8") as file:
            for line in file:
                if not (line := line.strip()):
                    continue
                item = json.loads(line)
                if "generation_index" in item:
                    max_index = max(max_index, int(item["generation_index"]))

    return accepted_count, hashes, max_index + 1


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False))
        file.write("\n")


async def run_generation(config: GeneratorConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "raw.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    accepted, seen_hashes, generation_index = load_existing_state(accepted_path, rejected_path)

    print(f"existing accepted: {accepted:,}")
    print(f"resume generation index: {generation_index:,}")
    if accepted >= config.target_count:
        print("target already reaced")
        return

    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key, timeout=config.request_timeout)
    semaphore = asyncio.Semaphore(config.concurrency)
    maximum_attempts = config.target_count * config.max_attempt_multiplier
    total_attempts = generation_index
    rejected = 0
    duplicates = 0

    while accepted < config.target_count and total_attempts < maximum_attempts:
        batch_size = min(config.concurrency, config.target_count - accepted)
        indices = range(generation_index, generation_index + batch_size)
        tasks = [request_scenario(client, config, index, semaphore) for index in indices]
        results = await asyncio.gather(*tasks)
        generation_index += batch_size
        total_attempts += batch_size

        for record, rejection in results:
            if rejection is not None:
                append_jsonl(rejected_path, rejection)
                rejected += 1
                continue

            assert record is not None
            state_hash = scenario_hash(record["state"])
            if state_hash in seen_hashes:
                append_jsonl(
                    rejected_path,
                    {"generation_index": record["generation_index"], "stage": "duplidate", "id": record["id"]},
                )
                duplicates += 1
                continue

            seen_hashes.add(state_hash)
            append_jsonl(accepted_path, record)
            accepted += 1

        print(
            f"\raccepted={accepted:,}/{config.target_count:,} rejected={rejected:,} "
            f"duplidates={duplicates:,} attempts={total_attempts:,}",
            end="",
            flush=True,
        )

    print()
    await client.close()
    stats = {
        "accepted": accepted,
        "rejected_this_run": rejected,
        "duplicates_this_run": duplicates,
        "last_generation_index": generation_index - 1,
        "teacher_model": config.model,
    }
    stats_path = output_dir / "generation_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"dataset: {accepted_path}")
    print(f"rejections: {rejected_path}")
    print(f"stats: {stats_path}")
