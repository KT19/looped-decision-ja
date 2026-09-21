import argparse
from pathlib import Path

from decision_corpus.builder import CorpusBuildConfig, build_corpus
from decision_corpus.sources import (
    DEFAULT_EXTERNAL_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SYNTHETIC_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC_PATH)
    parser.add_argument("--external-dir", type=Path, default=DEFAULT_EXTERNAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--synthetic-eval-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_corpus(
        CorpusBuildConfig(
            synthetic_path=args.synthetic,
            external_dir=args.external_dir,
            output_dir=args.output_dir,
            synthetic_eval_fraction=args.synthetic_eval_fraction,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
