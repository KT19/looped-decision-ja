from typing import Any

import jax
import jax.numpy as jnp

PyTree = Any


def tree_add(left: PyTree, right: PyTree) -> PyTree:
    return jax.tree_util.tree_map(lambda x, y: x + y, left, right)


def tree_scale(tree: PyTree, scale: float | jax.Array) -> PyTree:
    return jax.tree_util.tree_map(lambda value: value * scale, tree)


def tree_squared_norm(tree: PyTree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return sum(
        (jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves),
        start=jnp.asarray(0.0, dtype=jnp.float32),
    )


def global_norm(*trees: PyTree) -> jax.Array:
    return jnp.sqrt(
        sum(
            (tree_squared_norm(tree) for tree in trees),
            start=jnp.asarray(0.0, dtype=jnp.float32),
        )
    )
