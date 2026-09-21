from dataclasses import dataclass
from typing import Any

import jax
import numpy as np


@dataclass
class ModelStats:
    total_params: int

    expert_params: int
    dense_params: int

    active_expert_params: float
    active_unique_params: float

    recurrent_total_params: int
    recurrent_expert_params: int
    recurrent_dense_params: int

    executed_param_equivalents: float


def _numel(x) -> int:
    return int(np.prod(x.shape))


def _path_to_str(path) -> str:
    parts = []

    for entry in path:
        if hasattr(entry, "key"):
            parts.append(str(entry.key))

        elif hasattr(entry, "name"):
            parts.append(str(entry.name))

        elif hasattr(entry, "idx"):
            parts.append(str(entry.idx))

        else:
            parts.append(str(entry))

    return "/".join(parts)


def calculate_model_stats(
    params: Any, n_experts: int, top_k: int, n_loops: int
) -> ModelStats:
    """
    Report parameter statistics
    """
    expert_names = {"gate_w", "up_w", "down_w"}

    recurrent_names = {"recurrent_stack", "recurrent_core"}

    total_params = 0
    expert_params = 0

    recurrent_total = 0
    recurrent_expert = 0

    outside_active = 0.0

    recurrent_active = 0.0

    expert_fraction = top_k / n_experts

    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(params)

    for path, value in leaves_with_paths:
        path_str = _path_to_str(path)

        count = _numel(value)

        total_params += count
        path_parts = set(path_str.split("/"))
        is_recurrent = bool(path_parts & recurrent_names)
        is_expert = bool(path_parts & expert_names)

        if is_expert:
            expert_params += count

            active_count = count * expert_fraction
        else:
            active_count = float(count)

        if is_recurrent:
            recurrent_total += count

            if is_expert:
                recurrent_expert += count

            recurrent_active += active_count

        else:
            outside_active += active_count

    dense_params = total_params - expert_params
    recurrent_dense = recurrent_total - recurrent_expert
    active_expert_params = expert_params * expert_fraction

    active_unique_params = dense_params + active_expert_params

    executed_param_equivalents = outside_active + n_loops * recurrent_active

    return ModelStats(
        total_params=total_params,
        expert_params=expert_params,
        dense_params=dense_params,
        active_expert_params=active_expert_params,
        active_unique_params=active_unique_params,
        recurrent_total_params=recurrent_total,
        recurrent_expert_params=recurrent_expert,
        recurrent_dense_params=recurrent_dense,
        executed_param_equivalents=executed_param_equivalents,
    )


def _millions(x) -> float:
    return float(x) / 1000000


def print_model_stats(
    params: Any, n_experts: int, top_k: int, n_loops: int
) -> ModelStats:
    stats = calculate_model_stats(
        params=params, n_experts=n_experts, top_k=top_k, n_loops=n_loops
    )

    active_ratio = stats.active_unique_params / stats.total_params

    print()

    print("=" * 10)
    print("MODEL PARAMETER REPORT")
    print("-" * 10)

    print(f"Physical parameters: {_millions(stats.total_params):8.2f} M")
    print(f"Dense/non-expert: {_millions(stats.dense_params):8.2f} M")
    print(f"Expert parameters: {_millions(stats.expert_params):8.2f} M")
    print()

    print(f"Active unique params/token: {_millions(stats.active_unique_params)} M")
    print(f"Active expert params: {_millions(stats.active_expert_params):8.2f} M")

    print(f"Active / physical ratio: {active_ratio * 100:8.2f} %")
    print()

    print(f"Recurrent-core physical: {_millions(stats.recurrent_total_params):8.2f} M")
    print(f"Recurrent dense: {_millions(stats.recurrent_dense_params):8.2f} M")
    print(f"Recurrent experts: {_millions(stats.recurrent_expert_params):8.2f} M")

    print()

    print(f"Recurrent loops: {n_loops}")
    print(
        f"Executed param-equivalents: {_millions(stats.executed_param_equivalents):8.2f} M"
    )

    print("-" * 10)
    print()

    return stats
