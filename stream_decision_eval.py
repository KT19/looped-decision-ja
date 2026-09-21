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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="decision_checkpoints/best")
    parser.add_argument("--data", default="data/decision_corpus/eval_in_domain.jsonl")
    parser.add_argument(
        "--source",
        default="jcommonsenseqa",
        help="Source within the JSONL; empty means all",
    )
    parser.add_argument("--limit", type=int, default=20, help="Maximum examples; 0 means all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")

    checkpoint = Path(args.checkpoint).resolve()
    config = json.loads((checkpoint.parent / "decision_config.json").read_text(encoding="utf-8"))
    _, model_cfg, backbone = build_backbone(config["pretrained_checkpoint"])

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"], use_fast=True)
    if len(tokenizer) != model_cfg.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match pretrained model")

    serializer = DecisionSerializer(tokenizer, int(config["max_seq_len"]), int(config["max_options"]))
    head = GeneralDecisionHead(d_model=model_cfg.d_model, d_head=model_cfg.d_model)
    saved = restore_checkpoint(checkpoint)

    @jax.jit
    def predict(input_ids, token_mask, option_mask, option_valid, decision_index):
        states, _ = backbone.apply(
            {"params": saved["backbone_params"]},
            input_ids,
            return_loop_states=True,
            compute_logits=False,
            token_mask=token_mask,
        )
        logits = head.apply(
            {"params": saved["head_params"]},
            states,
            option_mask,
            option_valid,
            decision_index,
        )
        return jax.nn.softmax(logits[-1].astype(jnp.float32), axis=-1)  # type: ignore

    count = correct = 0
    inference_seconds = 0.0
    try:
        with Path(args.data).open(encoding="utf-8") as rows:
            for line in rows:
                if not line.strip():
                    continue
                example = json.loads(line)
                if args.source and example["source"] != args.source:
                    continue
                options = list(example["options"])
                encoded = serializer.serialize(example["state"], example["question"], options)
                lengths = np.asarray([encoded["input_length"]], dtype=np.int32)
                inputs = (
                    jax.device_put(encoded["input_ids"][None, :]),
                    jax.device_put(make_token_mask(lengths, serializer.max_seq_len)),
                    jax.device_put(encoded["option_mask"][None, ...]),
                    jax.device_put(encoded["option_valid"][None, :]),
                    jax.device_put(np.asarray([encoded["decision_index"]], dtype=np.int32)),
                )

                if count == 0:  # Compile before timing.
                    jax.block_until_ready(predict(*inputs))

                started = time.perf_counter()
                probabilities = predict(*inputs)
                jax.block_until_ready(probabilities)
                elapsed = time.perf_counter() - started

                scores = np.asarray(probabilities[0, : len(options)])
                answer = int(np.argmax(scores))
                gold = int(example["hard_label"])
                count += 1
                correct += int(answer == gold)
                inference_seconds += elapsed
                print(f"\n[{count}] {example['source']}  {'✓' if answer == gold else '✗'}")
                print(f"状態: {example['state']}")
                print(f"質問: {example['question']}")
                for index, (option, score) in enumerate(zip(options, scores, strict=True)):
                    marker = "→" if index == answer else " "
                    print(f" {marker} {index + 1}. {score:6.1%} {option}")
                print(
                    f"正解: {gold + 1}. {options[gold]}  |  推論: {elapsed * 1000:.1f} ms",
                    flush=True,
                )
                if args.limit and count >= args.limit:
                    break
    except KeyboardInterrupt:
        print("\n中断しました。")

    if not count:
        print("対象の評価例がありません。")
        return
    print(
        f"\n{count}件 | 正解率 {correct / count:.1%} | 平均推論 {inference_seconds / count * 1000:.1f} ms/件 | {count / inference_seconds:.2f} 件/秒"
    )


if __name__ == "__main__":
    main()
