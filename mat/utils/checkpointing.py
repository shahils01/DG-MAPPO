"""Utilities for locating and inspecting resumable training checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


LATEST_CHECKPOINT_NAME = "latest.pt"


def checkpoint_directory(args, run_dir) -> Path:
    """Return the stable checkpoint directory for a training run."""
    configured = getattr(args, "checkpoint_dir", None)
    if configured:
        return Path(configured).expanduser()
    return Path(run_dir) / "checkpoints" / f"seed{int(args.seed)}"


def resolve_resume_checkpoint(args, run_dir) -> Optional[Path]:
    """Resolve an explicit checkpoint or the stable ``latest.pt`` pointer."""
    explicit = getattr(args, "resume_checkpoint", None)
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
        return path

    if getattr(args, "auto_resume", False):
        latest = checkpoint_directory(args, run_dir) / LATEST_CHECKPOINT_NAME
        if latest.is_file():
            return latest
    return None


def load_training_checkpoint(path) -> Dict[str, Any]:
    """Load a trusted full-training checkpoint on CPU."""
    try:
        checkpoint = torch.load(
            Path(path), map_location="cpu", weights_only=False
        )
    except TypeError as error:
        # PyTorch versions predating the weights_only argument are still used
        # by some cluster environments.
        if "weights_only" not in str(error):
            raise
        checkpoint = torch.load(Path(path), map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"{path} is not a resumable training checkpoint. "
            "Use --model_dir for legacy transformer_*.pt weight files."
        )
    return checkpoint


def checkpoint_metadata(path) -> Dict[str, Any]:
    """Read the small fields needed before runner/W&B initialization."""
    checkpoint = load_training_checkpoint(path)
    return {
        "checkpoint_version": checkpoint.get("checkpoint_version"),
        "wandb_run_id": checkpoint.get("wandb_run_id"),
        "algorithm_name": checkpoint.get("algorithm_name"),
        "num_agents": checkpoint.get("num_agents"),
        "next_episode": checkpoint.get("next_episode", 0),
        "total_num_steps": checkpoint.get("total_num_steps", 0),
    }
