import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from data.decision_data import DecisionSerializer, make_token_mask
from models.decision_model import GeneralDecisionHead
from training.checkpoint import restore_checkpoint
from utils.load_backbone import build_backbone


def print_result(options: list[str], probabilities: np.ndarray, elapsed_ms: float) -> None:
    print()
    print("=" * 16)
    prediction = int(np.argmax(probabilities))

    for index, (option, probability) in enumerate(zip(options, probabilities, strict=True)):
        marker = ">" if index == prediction else " "
        print(f" {marker} {index + 1}. {probability:7.2%} {option}")

    print(f"model latency: {elapsed_ms:.2f} ms")
    print("=" * 16)
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="decision_checkpoints/best")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = checkpoint_path.parent / "decision_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    _, model_cfg, backbone = build_backbone(config["pretrained_checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"], use_fast=True)
    if len(tokenizer) != model_cfg.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match pretrained model.")

    head = GeneralDecisionHead(d_model=model_cfg.d_model, d_head=model_cfg.d_model)
    state = restore_checkpoint(checkpoint_path)
    backbone_params = state["backbone_params"]
    head_params = state["head_params"]

    @jax.jit
    def infer(
        input_ids: jax.Array,
        token_mask: jax.Array,
        option_mask: jax.Array,
        option_valid: jax.Array,
        decision_index: jax.Array,
    ) -> jax.Array:
        loop_states, _ = backbone.apply(
            {"params": backbone_params},
            input_ids,
            return_loop_states=True,
            compute_logits=False,
            token_mask=token_mask,
        )
        logits = head.apply(
            {"params": head_params},
            loop_states,
            option_mask,
            option_valid,
            decision_index,
        )
        return jax.nn.softmax(logits.astype(jnp.float32), axis=-1)  # type: ignore

    max_seq_len = int(config["max_seq_len"])
    max_options = int(config["max_options"])
    serializer = DecisionSerializer(tokenizer, max_seq_len, max_options)

    def prepare_inputs(state_text: str, question: str, options: list[str]) -> tuple[dict[str, jax.Array], int]:
        encoded = serializer.serialize(state_text, question, options)
        input_lengths = np.asarray([encoded["input_length"]], dtype=np.int32)
        inputs = {
            "input_ids": jax.device_put(encoded["input_ids"][None, :]),
            "token_mask": jax.device_put(make_token_mask(input_lengths, max_seq_len)),
            "option_mask": jax.device_put(encoded["option_mask"][None, ...]),
            "option_valid": jax.device_put(encoded["option_valid"][None, :]),
            "decision_index": jax.device_put(np.asarray([encoded["decision_index"]], dtype=np.int32)),
        }
        return inputs, int(encoded["input_length"])

    print("Compiling inference...")
    dummy_inputs, _ = prepare_inputs("テスト", "どっちが良い？", ["選択肢A", "選択肢B"])
    jax.block_until_ready(infer(**dummy_inputs))
    print("Ready.")
    print()
    print("-" * 10)
    print("Decision interactive demo")
    print()

    while True:
        try:
            print("=" * 16)
            print("STATE")
            print("(Finish with an empty line)")
            state_lines = []
            while line := input("> "):
                state_lines.append(line)

            state_text = "\n".join(state_lines).strip()
            if not state_text:
                continue

            print()
            question = input("QUESTION> ").strip()
            if not question:
                continue

            print()
            print("OPTIONS")
            print("Enter one option per line.")
            print("Empty line finishes options.")
            options = []
            while len(options) < max_options:
                option = input(f"{len(options) + 1}> ").strip()
                if not option:
                    break
                options.append(option)

            if len(options) < 2:
                print("Need at least 2 options.")
                continue

            inputs, input_length = prepare_inputs(state_text, question, options)
            start = time.perf_counter()
            probabilities = infer(**inputs)
            jax.block_until_ready(probabilities)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            option_probabilities = np.asarray(probabilities[-1, 0, : len(options)], dtype=np.float32)
            print_result(options, option_probabilities, elapsed_ms)
            print(f"input tokens: {input_length}")
            print()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Bye.")
            break


if __name__ == "__main__":
    main()
