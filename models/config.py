from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 128
    d_model: int = 256

    n_heads: int = 8
    n_kv_heads: int = 2

    d_ff: int = 768
    max_seq_len: int = 128

    n_experts: int = 4
    top_k: int = 2
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    capacity_factor: float = 1.25

    # Recurrent depth
    n_loops: int = 4
    n_core_blocks: int = 2

    loop_residual_scale: float | None = None
    depth_embedding_scale: float = 0.1
