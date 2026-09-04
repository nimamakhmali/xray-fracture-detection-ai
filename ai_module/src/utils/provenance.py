"""src/utils/provenance.py — checkpoint / run identity helpers shared by evaluate, explain, benchmark."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "UNAVAILABLE"


def _experiment_from_path(weights: Path):
    w = Path(weights).resolve()
    return w.parent.parent.name if w.parent.name == "weights" else None


def derive_run_id(weights: Path, sha: str) -> str:
    """<experiment>_<sha8> — stable for a given checkpoint, distinct across checkpoints."""
    return f"{_experiment_from_path(weights) or Path(weights).stem}_{sha[:8]}"


def lookup_training_metadata(root: Path, weights: Path) -> Dict[str, str]:
    """Read reports/training/<experiment>_final.json (or production PROVENANCE.json). Never guesses."""
    out = {"experiment_id": "UNAVAILABLE", "dataset_version": "UNAVAILABLE",
           "training_status": "UNAVAILABLE", "run_kind": "UNAVAILABLE"}
    w = Path(weights).resolve()
    exp = _experiment_from_path(w)
    if exp is None and w.parent.name == "production" and (w.parent / "PROVENANCE.json").exists():
        exp = json.loads((w.parent / "PROVENANCE.json").read_text()).get("experiment_id")
    if exp:
        out["experiment_id"] = exp
        f = root / "reports" / "training" / f"{exp}_final.json"
        if f.exists():
            d = json.loads(f.read_text())
            out.update(dataset_version=d.get("dataset_version", "UNAVAILABLE"),
                       training_status=d.get("status", "UNAVAILABLE"),
                       run_kind=d.get("run_kind", "UNAVAILABLE"))
    return out


def current_freeze_version(root: Path) -> str:
    f = root / "reports" / "dataset" / "frozen_v1.json"
    return json.loads(f.read_text()).get("version", "UNAVAILABLE") if f.exists() else "UNAVAILABLE"