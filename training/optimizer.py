from typing import Any

import jax
import optax

from optimizers.muon import muon
from training.pytree import PyTree


def make_muon_adamw_optimizer(
    params: PyTree,
    learning_rate: optax.ScalarOrSchedule,
    *,
    weight_decay: float,
    muon_momentum: float,
    muon_ns_steps: int,
) -> optax.GradientTransformation:
    """Use Muon for matrix weights and AdamW for all other parameters."""

    def label_fn(path: tuple[Any, ...], value: jax.Array) -> str:
        path_str = "/".join(str(entry) for entry in path)
        is_expert_matrix = value.ndim == 3 and any(name in path_str for name in ("gate_w", "up_w", "down_w"))
        is_dense_kernel = (
            value.ndim == 2 and "kernel" in path_str and "lm_head" not in path_str and "router" not in path_str
        )
        return "muon" if is_expert_matrix or is_dense_kernel else "adamw"

    labels = jax.tree_util.tree_map_with_path(label_fn, params)
    return optax.multi_transform(
        {
            "muon": muon(
                learning_rate=learning_rate,
                momentum=muon_momentum,
                ns_steps=muon_ns_steps,
            ),
            "adamw": optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
        },
        labels,
    )
