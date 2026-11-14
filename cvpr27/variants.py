"""Loading trained EMA models and drawing samples for every variant (spec 2.5)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from cvpr27 import physvid as pv
from cvpr27.common import drive_root, fmt_s, new_path
from cvpr27.data import Norm
from cvpr27.diffusion import Diffusion
from cvpr27.models import Variant


def load_ema(run_name: str, dev):
    run = drive_root() / run_name
    if not (run / "log.json").exists():
        raise FileNotFoundError(f"{run_name} has no log.json: training not finished")
    cfg = json.loads((run / "config.json").read_text())
    final = json.loads((run / "log.json").read_text())["final_ema"]
    ck = torch.load(run / final, map_location=dev)
    a = cfg["args"]
    variant = Variant(a["variant"], pv.T, pv.H, pv.W)
    model = variant.build(a["base"]).to(dev)
    model.load_state_dict(ck["ema"])
    model.eval().requires_grad_(False)
    return {"variant": variant, "model": model, "diff": Diffusion(a["diff_steps"], dev),
            "norm_stats": ck["norm_stats"], "cfg": cfg, "ckpt": final, "run": run_name}


def draw_dir(tag: str, d: int) -> Path:
    p = drive_root() / "eval_samples" / tag / f"draw{d}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def sample_job(name: str, d: int, m: dict, n: int, seed: int, dev, tag: str, norm: Norm, batch: int,
               cond_u8=None):
    """One (variant, draw) sampling job. Resumable: an existing .npz is loaded, never overwritten."""
    path = draw_dir(tag, d) / f"samples_{name}.npz"
    if path.exists():
        z = np.load(path)
        print(f"  [draw {d}] {name}: loaded existing samples")
        return {k: z[k] for k in z.files if k in ("video", "omega", "tau")}
    t0 = time.time()
    cond = torch.from_numpy(norm.video_to_model(cond_u8)) if cond_u8 is not None else None
    v, s = m["diff"].sample(m["variant"], m["model"], n, dev, mode="ancestral", seed=seed, batch=batch,
                            cond_video=cond)
    out = {}
    if v is not None:
        out["video"] = norm.video_to_u8(v.numpy())
    if s is not None:
        out["omega"], out["tau"] = (x.astype(np.float32) for x in norm.sens_to_phys(s.numpy()))
    np.savez_compressed(new_path(path), seed=seed, n=n, mode="ancestral", ckpt=m["ckpt"], run=m["run"], **out)
    print(f"  [draw {d}] {name}: {n} samples, {m['diff'].T}-step ancestral, {fmt_s(time.time() - t0)}")
    return out
