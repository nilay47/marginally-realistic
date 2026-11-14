"""Run-directory handling, config.json, git hash, atomic writes. No hyperparameters live here."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

DRIVE_ROOT = "/content/drive/MyDrive/physgen_cvpr27"
REPO = Path(__file__).resolve().parent.parent


def drive_root() -> Path:
    """Root for all outputs. PHYSGEN_ROOT overrides (used for local smoke tests)."""
    return Path(os.environ.get("PHYSGEN_ROOT", DRIVE_ROOT))


def git_info() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD") or "unknown",
            "dirty": bool(run("status", "--porcelain")),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD") or "unknown"}


def atomic_json(path: Path, obj) -> None:
    """Write JSON atomically. Refuses to overwrite an existing file (Rule 3)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)


def new_path(path: Path) -> Path:
    """Return path if free, else raise. Used for every artifact so nothing is overwritten."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    return path


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return time.time() - self.t0


def fmt_s(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"
