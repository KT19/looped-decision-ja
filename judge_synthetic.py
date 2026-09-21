import asyncio

from synthetic_judging.config import load_judge_config
from synthetic_judging.pipeline import run_judging


def main() -> None:
    config = load_judge_config("configs/judge_synthetic.yaml")
    asyncio.run(run_judging(config))


if __name__ == "__main__":
    main()
