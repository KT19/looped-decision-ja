import asyncio

from synthetic_generation.config import load_generator_config
from synthetic_generation.pipeline import run_generation


def main() -> None:
    config = load_generator_config("configs/synthetic.yaml")
    asyncio.run(run_generation(config))


if __name__ == "__main__":
    main()
