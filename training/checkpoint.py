import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import orbax.checkpoint as ocp

_STEP_PATTERN = re.compile(r"step_(\d+)$")


def prune_step_checkpoints(directory: str | Path, keep: int = 2) -> list[Path]:
    """Keep the newest numbered checkpoints; leave best and other files alone."""
    if keep < 1:
        raise ValueError("keep must be at least 1")
    directory = Path(directory).resolve()
    checkpoints = []
    for path in directory.iterdir():
        match = _STEP_PATTERN.fullmatch(path.name)
        if match and path.is_dir() and not path.is_symlink():
            checkpoints.append((int(match.group(1)), path))
    removed = []
    for _, path in sorted(checkpoints)[:-keep]:
        shutil.rmtree(path)
        removed.append(path)
    return removed


def make_checkpointer() -> ocp.Checkpointer:
    return ocp.Checkpointer(ocp.StandardCheckpointHandler())


def restore_checkpoint(path: str | Path, target: Any | None = None) -> Any:
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    args = ocp.args.StandardRestore() if target is None else ocp.args.StandardRestore(target)
    return make_checkpointer().restore(checkpoint_path, args=args)


def save_checkpoint(path: str | Path, state: Any, *, force: bool = False) -> None:
    checkpoint_path = Path(path).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    make_checkpointer().save(
        checkpoint_path,
        args=ocp.args.StandardSave(state),
        force=force,
    )


class CheckpointManager:
    def __init__(self, directory: str, keep: int = 2):
        self.directory = Path(directory).resolve()

        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep = keep

        self.checkpointer = make_checkpointer()

    def path_for_step(self, step: int) -> Path:
        return self.directory / f"step_{step:08d}"

    def available_steps(self) -> list[int]:
        steps = []

        for path in self.directory.iterdir():
            if not path.is_dir():
                continue

            match = _STEP_PATTERN.match(path.name)

            if match is None:
                continue

            steps.append(int(match.group(1)))

        return sorted(steps)

    def latest_step(self) -> int | None:
        steps = self.available_steps()

        if not steps:
            return None

        return steps[-1]

    def save(
        self,
        step: int,
        params: Any,
        opt_state: Any,
        tokens_seen: int,
        best_val_loss: float,
        data_state: dict,
        force: bool = False,
    ) -> None:
        path = self.path_for_step(step)

        state = {
            "params": params,
            "opt_state": opt_state,
            "step": np.asarray(step, dtype=np.int64),
            "tokens_seen": np.asarray(tokens_seen, dtype=np.int64),
            "best_val_loss": np.asarray(best_val_loss, dtype=np.float32),
        }

        self.checkpointer.save(path, args=ocp.args.StandardSave(state), force=force)

        # data-loader state is small json metadata
        data_state_path = path / "data_state.json"

        with data_state_path.open("w", encoding="utf-8") as f:
            json.dump(data_state, f, ensure_ascii=False)

        print(f"checkpoint saved: {path}")

        self._cleanup()

    def restore_latest(self, params: Any, opt_state: Any) -> Any:
        step = self.latest_step()

        if step is None:
            return None

        path = self.path_for_step(step)

        target = {
            "params": params,
            "opt_state": opt_state,
            "step": np.asarray(0, dtype=np.int64),
            "tokens_seen": np.asarray(0, dtype=np.int64),
            "best_val_loss": np.asarray(np.inf, dtype=np.float32),
        }

        restored = self.checkpointer.restore(path, args=ocp.args.StandardRestore(target))
        data_state_path = path / "data_state.json"

        if not data_state_path.exists():
            raise RuntimeError("Checkpoint does not contain 'data_state.json'")

        with data_state_path.open("r", encoding="utf-8") as f:
            data_state = json.load(f)

        restored["data_state"] = data_state

        print(f"checkpoint restored: {path}")

        return restored

    def _cleanup(self) -> None:
        if self.keep <= 0:
            return

        steps = self.available_steps()

        old_steps = steps[: -self.keep]

        for step in old_steps:
            path = self.path_for_step(step)

            shutil.rmtree(path)

            print(f"removed old checkpoint: {path}")
