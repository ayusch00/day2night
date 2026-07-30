import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def load_cfg(path, section=None):
    """Load a YAML config and optionally select a named top-level section."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    if section is None:
        return cfg
    if section not in cfg:
        raise KeyError(
            f"Config is missing required top-level section '{section}': {path}"
        )
    selected = cfg[section]
    if not isinstance(selected, dict):
        raise ValueError(f"Config section '{section}' must be a mapping: {path}")
    return selected


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def make_run_dirs(cfg):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run = f"{cfg['exp_name']}_{ts}"
    exp_dir = os.path.join(cfg["logging"]["out_dir"], run)
    res_dir = cfg["logging"]["results_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    return exp_dir, res_dir

def resolve_encoder_checkpoint(
    spec,
    experiments_root="experiments",
    default_run_prefix="seg",
    filename="encoder_GE.pth",
):
    """
    Resolve the encoder checkpoint path.

    spec may be:
        - None/empty: returns None (no checkpoint)
        - Path to a .pth file
        - Directory that contains filename (defaults to encoder_GE.pth)
        - "latest" / "auto": pick the newest experiments/{default_run_prefix}_*/filename
        - "latest:<pattern>": glob pattern relative to experiments_root (e.g. "seg_cityscapes_*")
    """
    if not spec:
        return None
    spec = str(spec)
    base = Path(experiments_root)
    def latest(pattern):
        if not base.exists():
            raise FileNotFoundError(f"Experiments root not found: {base}")
        pat = pattern or f"{default_run_prefix}_*"
        candidates = sorted(
            (p for p in base.glob(f"{pat}/{filename}") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(f"No encoder checkpoints found for pattern '{pat}' under {base}")
        return candidates[-1]

    if spec in {"latest", "auto"}:
        return str(latest(None))
    if spec.startswith("latest:"):
        _, _, pattern = spec.partition(":")
        return str(latest(pattern.strip() or None))

    path = Path(spec).expanduser()
    if path.is_dir():
        path = path / filename
    if path.is_file():
        return str(path)
    raise FileNotFoundError(f"Encoder checkpoint not found: {path}")
