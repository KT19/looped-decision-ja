import json
import random
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_corpus.converters import (
    convert_jcola,
    convert_jcommonsenseqa,
    convert_jnli,
    convert_jsts,
    convert_synthetic,
)
from decision_corpus.schema import DecisionExample
from decision_corpus.sources import JCOLA_REPO, JGLUE_REPO, JGLUE_VERSION, SourcePaths


@dataclass(frozen=True)
class CorpusBuildConfig:
    synthetic_path: Path
    external_dir: Path
    output_dir: Path
    synthetic_eval_fraction: float = 0.10
    seed: int = 42


def run_git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def ensure_repo(url: str, path: Path, branch: str | None = None) -> None:
    """Clone a missing repository and leave an existing clone untouched."""
    if path.exists():
        if not (path / ".git").exists():
            raise RuntimeError(f"{path} exists but is not a Git repository")
        print(f"using existing repository: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    command = ["clone", "--depth", "1"]
    if branch is not None:
        command.extend(["--branch", branch])
    command.extend([url, str(path)])
    print(f"cloning: {url}")
    run_git(*command)


def git_head_commit(path: Path) -> str:
    """Read the current commit hash without creating or pushing a commit."""
    return run_git("rev-parse", "HEAD", cwd=path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found:\n{path}")

    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not (line := line.strip()):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON:\n{path}:{line_number}") from exc
    return records


def write_jsonl(path: Path, examples: Iterable[DecisionExample]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example.model_dump(), ensure_ascii=False))
            file.write("\n")
            count += 1
    return count


def counts_by(examples: list[DecisionExample], attribute: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        key = str(getattr(example, attribute))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def check_paths(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        message = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required dataset files are missing:\n{message}")


def build_corpus(config: CorpusBuildConfig) -> None:
    sources = SourcePaths.from_external_dir(config.external_dir)
    ensure_repo(JGLUE_REPO, sources.jglue_root, branch=JGLUE_VERSION)
    ensure_repo(JCOLA_REPO, sources.jcola_root)
    check_paths([config.synthetic_path, *sources.dataset_files()])

    train: list[DecisionExample] = []
    train.extend(convert_jnli(read_jsonl(sources.jnli_train), split="train"))
    train.extend(convert_jcommonsenseqa(read_jsonl(sources.jcommonsenseqa_train), split="train"))
    train.extend(convert_jsts(read_jsonl(sources.jsts_train), split="train"))

    eval_in_domain: list[DecisionExample] = []
    eval_in_domain.extend(convert_jnli(read_jsonl(sources.jnli_valid), split="eval"))
    eval_in_domain.extend(convert_jcommonsenseqa(read_jsonl(sources.jcommonsenseqa_valid), split="eval"))
    eval_in_domain.extend(convert_jsts(read_jsonl(sources.jsts_valid), split="eval"))

    synthetic_train, synthetic_eval = convert_synthetic(
        config.synthetic_path,
        eval_fraction=config.synthetic_eval_fraction,
        seed=config.seed,
    )
    train.extend(synthetic_train)
    random.Random(config.seed).shuffle(train)

    eval_transfer = []
    eval_transfer.extend(convert_jcola(read_jsonl(sources.jcola_valid_in), distribution_name="in_domain"))
    eval_transfer.extend(convert_jcola(read_jsonl(sources.jcola_valid_out), distribution_name="out_of_domain"))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_count = write_jsonl(config.output_dir / "train.jsonl", train)
    eval_count = write_jsonl(config.output_dir / "eval_in_domain.jsonl", eval_in_domain)
    synthetic_eval_count = write_jsonl(config.output_dir / "eval_synthetic.jsonl", synthetic_eval)
    transfer_count = write_jsonl(config.output_dir / "eval_transfer_jcola.jsonl", eval_transfer)

    manifest = {
        "version": "decision-corpus-v1",
        "seed": config.seed,
        "synthetic_eval_fraction": config.synthetic_eval_fraction,
        "repositories": {
            "jglue": {
                "requested_version": JGLUE_VERSION,
                "commit": git_head_commit(sources.jglue_root),
                "path": str(sources.jglue_root),
            },
            "jcola": {
                "commit": git_head_commit(sources.jcola_root),
                "path": str(sources.jcola_root),
            },
        },
        "counts": {
            "train": train_count,
            "eval_in_domain": eval_count,
            "eval_synthetic": synthetic_eval_count,
            "eval_transfer_jcola": transfer_count,
        },
        "train_by_source": counts_by(train, "source"),
        "train_by_type": counts_by(train, "type"),
        "eval_in_domain_by_source": counts_by(eval_in_domain, "source"),
        "eval_synthetic_by_type": counts_by(synthetic_eval, "type"),
        "transfer_by_source": counts_by(eval_transfer, "source"),
        "synthetic_source": str(config.synthetic_path),
    }
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 64)
    print(f"train: {train_count}")
    print(f"in-domain eval: {eval_count}")
    print(f"synthetic eval: {synthetic_eval_count}")
    print(f"JCoLA transfer eval: {transfer_count}")
