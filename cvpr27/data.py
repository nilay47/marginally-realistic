"""
Data cache + normalisation (spec 2.1, 2.2).

Cache layout (on Drive, shared by every notebook):
    <root>/cache/pendulum_T64_dt0.04_{train,heldout}.npz
        video  uint8   (n, T, 32, 32)
        omega  float32 (n, T)      gyro
        tau    float32 (n, T)      torque
        theta  float32 (n, T)      TRUE simulator angle (floors only, never a model input)
        meta   json string: seed, n, T, DT, constants
Train = 10,000 clips seed 0. Held-out real = 2,048 clips seed 1.

Normalisation: video -> [-1, 1]; gyro/torque standardised with TRAIN mean/std
(scalars per channel, over clips and time). Mean/std are written into every config.json.
Always de-normalise before computing a diagnostic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cvpr27 import physvid as pv
from cvpr27.common import drive_root, new_path

N_TRAIN, N_HELD = 10_000, 2_048
SEED_TRAIN, SEED_HELD = 0, 1


def cache_dir() -> Path:
    d = drive_root() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _meta(n, seed):
    return {"n": n, "seed": seed, "T": pv.T, "DT": pv.DT, "WIN": pv.WIN, "H": pv.H, "W": pv.W,
            "M": pv.M, "L": pv.L, "G": pv.G, "C": pv.C, "TAU_AMP": pv.TAU_AMP, "SUB": pv.SUB,
            "R_PX": pv.R_PX, "SIGMA": pv.SIGMA, "N_MODES": pv.N_MODES, "F_LO": pv.F_LO, "F_HI": pv.F_HI}


def cache_path(split: str) -> Path:
    return cache_dir() / f"pendulum_T{pv.T}_dt{pv.DT}_{split}.npz"


def build_split(split: str, n: int, seed: int, chunk: int = 1000):
    """Simulate + render in chunks. Chunks reuse the SAME rng stream: simulate() is run once for
    the whole split (cheap) and only rendering is chunked, so seed semantics match physvid."""
    path = cache_path(split)
    if path.exists():
        return path, False
    sim = pv.simulate(n, seed=seed)
    th = sim["theta"]
    vid = np.empty((n, pv.T, pv.H, pv.W), dtype=np.uint8)
    for s in range(0, n, chunk):
        vid[s:s + chunk] = pv.to_uint8(pv.render(th[s:s + chunk]))
    tmp = new_path(path)
    tmp = tmp.with_name(tmp.stem + ".tmp.npz")
    np.savez(tmp, video=vid, omega=sim["omega"].astype(np.float32),
             tau=sim["tau"].astype(np.float32), theta=th.astype(np.float32),
             meta=json.dumps(_meta(n, seed)))
    tmp.rename(path)
    return path, True


def load_split(split: str) -> dict:
    path = cache_path(split)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run nb0 / `python -m cvpr27.build_cache` first")
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in ("video", "omega", "tau", "theta")}
    d["meta"] = json.loads(str(z["meta"]))
    want = _meta(d["meta"]["n"], d["meta"]["seed"])
    diff = {k: (d["meta"].get(k), want[k]) for k in want if d["meta"].get(k) != want[k]}
    if diff:   # stale cache (e.g. built before the TAU_AMP retune): refuse rather than train on the wrong physics
        raise RuntimeError(f"cache {path.name} was built with different physics constants (cache, current): {diff}. "
                           "Delete/rename the cache and rerun nb0.")
    return d


def build_all():
    out = {}
    for split, n, seed in (("train", N_TRAIN, SEED_TRAIN), ("heldout", N_HELD, SEED_HELD)):
        path, built = build_split(split, n, seed)
        out[split] = {"path": str(path), "built_now": built}
    return out


# ---------------------------------------------------------------- normalisation
def sensor_stats(train: dict) -> dict:
    return {"gyro_mean": float(train["omega"].mean()), "gyro_std": float(train["omega"].std()),
            "torque_mean": float(train["tau"].mean()), "torque_std": float(train["tau"].std())}


class Norm:
    """video uint8 <-> [-1, 1]; sensors <-> standardised. Built from config stats only."""

    def __init__(self, stats: dict):
        self.mu = np.array([stats["gyro_mean"], stats["torque_mean"]], np.float32)
        self.sd = np.array([stats["gyro_std"], stats["torque_std"]], np.float32)
        self.stats = stats

    @staticmethod
    def video_to_model(v_u8):            # (n,T,H,W) uint8 -> (n,1,T,H,W) float32 in [-1,1]
        return (v_u8.astype(np.float32) / 127.5 - 1.0)[:, None]

    @staticmethod
    def video_to_u8(v):                  # (n,1,T,H,W) in [-1,1] -> (n,T,H,W) uint8
        x = np.clip((np.asarray(v, np.float32)[:, 0] + 1.0) * 127.5, 0, 255)
        return np.round(x).astype(np.uint8)

    def sens_to_model(self, omega, tau):  # (n,T),(n,T) -> (n,2,T)
        s = np.stack([omega, tau], 1).astype(np.float32)
        return (s - self.mu[None, :, None]) / self.sd[None, :, None]

    def sens_to_phys(self, s):           # (n,2,T) -> omega (n,T), tau (n,T)
        s = np.asarray(s, np.float32) * self.sd[None, :, None] + self.mu[None, :, None]
        return s[:, 0].astype(np.float64), s[:, 1].astype(np.float64)
