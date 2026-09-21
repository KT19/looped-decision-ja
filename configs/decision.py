from dataclasses import dataclass
from pathlib import Path

from configs.loading import load_yaml_sections


@dataclass(frozen=True)
class DecisionConfig:
    train_data: str
    pretrained_checkpoint: str
    output_dir: str
    eval_datasets: dict[str, str]

    max_seq_len: int = 512
    max_options: int = 6
    microbatch_size: int = 4
    grad_accum_steps: int = 4
    epochs: int = 3

    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    backbone_weight_decay: float = 0.0
    head_weight_decay: float = 0.0
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0

    eval_batch_size: int = 8
    eval_every: int = 250
    log_every: int = 20
    seed: int = 42
    resume_checkpoint: str | None = None


def load_decision_config(path: str | Path) -> DecisionConfig:
    return DecisionConfig(**load_yaml_sections(path))
