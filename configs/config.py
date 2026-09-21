from dataclasses import dataclass
from pathlib import Path

from configs.loading import load_yaml_sections


@dataclass(frozen=True)
class Config:
    # Model
    vocab_size: int = 50570
    d_model: int = 768

    n_heads: int = 16
    n_kv_heads: int = 4

    d_ff: int = 3072
    max_seq_len: int = 1024

    # MoE
    n_experts: int = 4
    top_k: int = 2
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    capacity_factor: float = 1.25

    # Recurrent depth
    n_loops: int = 2
    n_core_blocks: int = 2
    loop_residual_scale: float | None = None
    depth_embedding_scale: float = 0.1

    # Training
    batch_size: int = 4
    seq_len: int = 1024
    steps: int = 50000
    grad_accum_steps: int = 4

    learning_rate: float = 3e-3
    weight_decay: float = 0.01

    # Muon
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5

    seed: int = 42

    # Tokenizer/data
    tokenizer_name: str = "llm-jp/llm-jp-13b-v1.0"
    dataset_name: str = "hotchpotch/fineweb-2-edu-japanese"
    dataset_config: str = "sample_10BT"
    shuffle_buffer: int = 10000

    # Validation
    val_split: str = "test"
    val_batches: int = 16
    eval_every: int = 100

    # checkpoint
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 2000
    keep_checkpoints: int = 2
    resume: bool = False

    # Optimization schedule
    warmup_steps: int = 500
    min_learning_rate: float = 3e-4
    max_grad_norm: float = 2.0  # since small scale

    # Generation / qualitative eval
    sample_every: int = 500
    sample_max_new_tokens: int = 128
    sample_temperature: float = 0.8
    sample_top_k: int = 40

    sample_prompts: tuple[str, ...] = (
        "これからの人工知能の未来は",
        "大規模言語モデルとは",
    )


def load_config(path: str) -> Config:
    """Load a feature-grouped YAML configuration into the training config."""
    values = load_yaml_sections(Path(path))

    if "sample_prompts" in values:
        values["sample_prompts"] = tuple(values["sample_prompts"])

    return Config(**values)
