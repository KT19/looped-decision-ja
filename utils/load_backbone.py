import json
from pathlib import Path
from typing import Any

from models.config import ModelConfig
from models.language_model import LanguageModel


def load_pretraining_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()

    root = checkpoint_path.parent

    experiment_path = root / "experiment.json"

    if not experiment_path.exists():
        raise FileNotFoundError(f"Missing experiment metadata: {experiment_path}")

    with experiment_path.open("r", encoding="utf-8") as f:
        metadata: dict[str, Any] = json.load(f)

    return metadata


def build_backbone(
    checkpoint_path: str | Path,
) -> tuple[dict[str, Any], ModelConfig, LanguageModel]:
    metadata = load_pretraining_metadata(checkpoint_path)
    cfg = metadata["config"]

    model_cfg = ModelConfig(
        vocab_size=int(cfg["vocab_size"]),
        max_seq_len=int(cfg["max_seq_len"]),
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        n_kv_heads=int(cfg["n_kv_heads"]),
        d_ff=int(cfg["d_ff"]),
        n_experts=int(cfg["n_experts"]),
        top_k=int(cfg["top_k"]),
        capacity_factor=float(cfg["capacity_factor"]),
        router_aux_loss_coef=float(cfg["router_aux_loss_coef"]),
        router_z_loss_coef=float(cfg["router_z_loss_coef"]),
        n_core_blocks=int(cfg["n_core_blocks"]),
        n_loops=int(cfg["n_loops"]),
        loop_residual_scale=cfg.get("loop_residual_scale"),
        depth_embedding_scale=float(cfg.get("depth_embedding_scale", 1.0)),
    )

    model = LanguageModel(model_cfg)

    return metadata, model_cfg, model
