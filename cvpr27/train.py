"""
Train one variant (spec 2.4, 2.5).

    python -m cvpr27.train --variant joint --run_name joint_s0 --seed 0

Rules honoured:
  2  EMA weights ema_<step>.pt + samples_ckpt_<step>.npz at EVERY checkpoint, final samples .npz
  3  nothing overwritten: if log.json exists the run is skipped; every artifact is created with
     new_path/atomic_json; config.json holds all hyperparameters + git commit. Resume (after a
     Colab disconnect) restarts from the newest resume_<step>.pt; only superseded resume files
     (scratch optimiser state, not results) are deleted.
  4  every printed/logged number is computed in this run
  5  NaN/inf in loss or metrics, extractor invalid fraction > 1% on REAL data, or a floor above 1e-2
     -> raise and stop
  6  times the first 200 steps and prints the ETA
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from cvpr27 import physvid as pv
from cvpr27.common import atomic_json, drive_root, fmt_s, git_info, new_path
from cvpr27.data import Norm, load_split, sensor_stats
from cvpr27.diffusion import Diffusion
from cvpr27.models import Variant, count_params
from cvpr27.suite import Reference


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["joint", "video_only", "sensor_only"])
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--diff_steps", type=int, default=250)
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--n_ckpt", type=int, default=256)
    ap.add_argument("--n_final", type=int, default=1024)
    ap.add_argument("--sample_batch", type=int, default=64)
    ap.add_argument("--n_train", type=int, default=0, help="0 = all cached train clips")
    ap.add_argument("--keep_all_resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run: checks plumbing only")
    a = ap.parse_args(argv)
    if a.smoke:
        a.steps, a.batch, a.base, a.ckpt_every, a.n_ckpt, a.n_final = 20, 2, 16, 10, 4, 4
        a.diff_steps, a.ddim_steps, a.warmup, a.sample_batch, a.n_train = 10, 4, 5, 4, 64
    return a


def ema_step(ema, model, d):
    pe, pm = list(ema.parameters()), [p.detach() for p in model.parameters()]
    torch._foreach_mul_(pe, d)
    torch._foreach_add_(pe, pm, alpha=1 - d)


def to_phys(variant, v, s, norm):
    """Model-space sample tensors -> physical arrays (video uint8, omega, tau)."""
    vu8 = norm.video_to_u8(v.numpy()) if v is not None else None
    om, tau = norm.sens_to_phys(s.numpy()) if s is not None else (None, None)
    return vu8, om, tau


def save_samples(path, vu8, om, tau, **extra):
    d = dict(extra)
    if vu8 is not None:
        d["video"] = vu8
    if om is not None:
        d["omega"], d["tau"] = om.astype(np.float32), tau.astype(np.float32)
    np.savez_compressed(new_path(path), **d)


def main(argv=None):
    a = get_args(argv)
    run = drive_root() / a.run_name
    if (run / "log.json").exists():
        print(f"[{a.run_name}] log.json exists -> run already finished, skipping.")
        return
    run.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print(f"[{a.run_name}] device {dev}  T={pv.T} DT={pv.DT} WIN={pv.WIN}")

    # ---------------------------------------------------------------- data + normalisation
    train, held = load_split("train"), load_split("heldout")
    if a.n_train:
        train = {k: (v[:a.n_train] if hasattr(v, "shape") else v) for k, v in train.items()}
    stats = sensor_stats(train)
    norm = Norm(stats)
    V = torch.from_numpy(train["video"]).to(dev)                       # uint8 on device
    S = torch.from_numpy(norm.sens_to_model(train["omega"], train["tau"])).to(dev)
    N = V.shape[0]

    # ---------------------------------------------------------------- metric + reference (+ rule-5 gates)
    from cvpr27.videofeat import load_metric
    metric = None if a.smoke else load_metric(dev)      # smoke: plumbing only, video metric is tested in nb0 on GPU
    ref = Reference(held, norm, metric, dev)
    f = ref.floor
    bad = []
    if f["invalid_frame_frac"] > 0.01:
        bad.append(f"extractor invalid-frame fraction on REAL data = {f['invalid_frame_frac']:.4f} > 1%")
    for k in ("r_energy", "r_energy_win_max", "kin_rad_max"):
        if f[k]["median"] > 1e-2:
            bad.append(f"real-data floor {k} median = {f[k]['median']:.3e} > 1e-2")
    if bad:
        raise RuntimeError("RULE 5 STOP. " + " | ".join(bad))

    variant = Variant(a.variant, pv.T, pv.H, pv.W)
    model = variant.build(a.base).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    diff = Diffusion(a.diff_steps, dev)

    cfg_path = run / "config.json"
    if not cfg_path.exists():
        atomic_json(cfg_path, {
            "args": vars(a), "git": git_info(), "device": str(dev), "torch": torch.__version__,
            "n_params": count_params(model), "n_train_clips": int(N),
            "pv": {"T": pv.T, "DT": pv.DT, "WIN": pv.WIN, "STRIDE": pv.STRIDE, "H": pv.H, "W": pv.W,
                   "C": pv.C, "G": pv.G, "TAU_AMP": pv.TAU_AMP},
            "norm_stats": stats, "video_metric": None if a.smoke else json.loads((drive_root() / "cache" / "video_metric.json").read_text()),
            "choices_not_in_spec": {
                "lr_schedule": "linear warmup then constant (spec lists warmup, no decay)",
                "weight_decay": a.wd, "loss": "mse(video eps) + mse(sensor eps) with equal weight on the two means",
                "x0_clip": {"video": 1.0, "sensors": 5.0}, "attn_pos_emb": "learned, bottleneck only",
                "decoder_blocks_per_level": 3, "groupnorm_groups": 8, "heads": 4},
            "floor_real_heldout": f})
    else:
        print(f"[{a.run_name}] config.json exists (resuming); not rewritten")

    # ---------------------------------------------------------------- resume
    start = 0
    rs = sorted(glob.glob(str(run / "resume_*.pt")))
    if rs:
        st = torch.load(rs[-1], map_location=dev)
        model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
        start = st["step"]
        print(f"[{a.run_name}] RESUMED from step {start} ({Path(rs[-1]).name})")
        atomic_json(run / f"resume_event_{int(time.time())}.json", {"from_step": start, "git": git_info()})
    torch.manual_seed(a.seed * 1_000_003 + start)
    lr_at = lambda s: a.lr * min(1.0, s / max(1, a.warmup))
    accum = 1

    def do_checkpoint(step, last, train_loss):
        """Idempotent: every artifact is created once and never overwritten. The scratch resume state is
        written FIRST, so a crash anywhere below resumes at exactly this step and finishes what is missing."""
        st_path = run / f"resume_{step:06d}.pt"
        if not st_path.exists():
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step}, st_path)
            if not a.keep_all_resume:                       # superseded scratch optimiser state only
                for old in glob.glob(str(run / "resume_*.pt")):
                    if old != str(st_path):
                        Path(old).unlink()
        ema_p = run / f"ema_{step:06d}.pt"
        if not ema_p.exists():
            torch.save({"ema": ema.state_dict(), "step": step, "variant": a.variant, "base": a.base,
                        "norm_stats": stats}, ema_p)
        plan = ((("ancestral", a.n_final, 0),) if last else ()) + (("ddim", a.n_ckpt, 1),)
        for mode, n_gen, off in plan:
            mp = run / (f"metrics_final_{mode}.json" if mode == "ancestral" else f"metrics_{step:06d}.json")
            if mp.exists():
                print(f"[{a.run_name}] step {step} {mode}: metrics already on disk, skipping")
                continue
            gv, gs = diff.sample(variant, ema, n_gen, dev, mode=mode, ddim_steps=a.ddim_steps,
                                 seed=10_000 + step + off, batch=a.sample_batch)
            vu8, om, tau = to_phys(variant, gv, gs, norm)
            sp = run / ("samples_final.npz" if mode == "ancestral" else f"samples_ckpt_{step:06d}.npz")
            if not sp.exists():
                save_samples(sp, vu8, om, tau, step=step, mode=mode)
            met = ref.evaluate(vu8, om, tau)
            atomic_json(mp, {"step": step, "sampler": mode, "n": int(n_gen),
                             "ddim_steps": a.ddim_steps if mode == "ddim" else None,
                             "train_loss": train_loss, "metrics": met})
            phys = met.get("physical", {})
            print(f"[{a.run_name}] step {step} [{mode} n={n_gen}] "
                  + " ".join(f"{k}={met[k]:.4g}" for k in ("w1_mean", "psd_mean") if k in met)
                  + (f" {met['video_metric']['name']}={met['video_metric']['value']:.4g}" if "video_metric" in met else "")
                  + (f" r_energy={phys['r_energy']['median']:.3e} win_max={phys['r_energy_win_max']['median']:.3e}"
                     f" kin_max={phys['kin_rad_max']['median']:.3e}" if phys else "")
                  + f" invalid_frames={met.get('invalid_frame_frac', float('nan')):.4f}")

    # ---------------------------------------------------------------- loop
    t_wall, t_eval, run_loss, n_loss, t_ck = time.time(), 0.0, {}, 0, time.time()
    if start > 0:
        do_checkpoint(start, start == a.steps, None)     # finish an interrupted checkpoint
    eta_printed = False
    for step in range(start + 1, a.steps + 1):
        model.train()
        idx = torch.randint(0, N, (a.batch,), device=dev)
        v = (V[idx].float() / 127.5 - 1.0)[:, None] if variant.has_video else None
        s = S[idx] if variant.has_sens else None
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        while True:                                                     # OOM -> micro-batching
            try:
                opt.zero_grad(set_to_none=True)
                parts_acc = {}
                for mb in range(accum):
                    sl = slice(mb * a.batch // accum, (mb + 1) * a.batch // accum)
                    loss, parts = diff.loss(variant, model, v[sl] if v is not None else None,
                                            s[sl] if s is not None else None)
                    (loss / accum).backward()
                    for k, x in parts.items():
                        parts_acc[k] = parts_acc.get(k, 0.0) + x / accum
                break
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                if accum >= a.batch:
                    raise
                accum *= 2
                print(f"  OOM -> gradient accumulation x{accum}")
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        if not math.isfinite(float(gn)):
            raise FloatingPointError(f"RULE 5 STOP: non-finite grad norm at step {step}")
        opt.step()
        ema_step(ema, model, a.ema)
        for k, x in parts_acc.items():
            run_loss[k] = run_loss.get(k, 0.0) + x
        n_loss += 1
        tot = sum(parts_acc.values())
        if not math.isfinite(tot):
            raise FloatingPointError(f"RULE 5 STOP: non-finite loss at step {step}: {parts_acc}")

        if step - start == 200 and not eta_printed:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            per = (time.time() - t_wall) / 200
            n_ck = (a.steps - step) // a.ckpt_every + 1
            print(f"[{a.run_name}] first 200 steps: {per:.3f}s/step (accum x{accum}) -> "
                  f"training-only ETA for the remaining {a.steps - step} steps: {fmt_s(per * (a.steps - step))} "
                  f"(+ ~{n_ck} checkpoint evals, timed at the first one)")
            eta_printed = True
        elif step % 100 == 0 and step % a.ckpt_every != 0:
            print(f"  step {step:6d}  loss {tot:.4f}  " + " ".join(f"{k} {x:.4f}" for k, x in parts_acc.items()))

        if step % a.ckpt_every == 0 or step == a.steps:
            t_ck = time.time()
            do_checkpoint(step, step == a.steps, {k: x / n_loss for k, x in run_loss.items()})
            run_loss, n_loss = {}, 0
            if t_eval == 0.0:
                t_eval = time.time() - t_ck
                print(f"[{a.run_name}] first checkpoint (save+sample+eval) took {fmt_s(t_eval)}; "
                      f"~{fmt_s(t_eval * ((a.steps - step) // a.ckpt_every))} of eval remaining")

    # ---------------------------------------------------------------- log.json (marks the run finished)
    curve = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(run / "metrics_0*.json")))]
    final = json.loads((run / "metrics_final_ancestral.json").read_text())
    atomic_json(run / "log.json", {"run": a.run_name, "variant": a.variant, "curve_ddim": curve,
                                   "final_ancestral": final, "floor_real_heldout": f,
                                   "final_ema": f"ema_{a.steps:06d}.pt", "wall_s": time.time() - t_wall})
    print(f"[{a.run_name}] done in {fmt_s(time.time() - t_wall)}")


if __name__ == "__main__":
    main()
