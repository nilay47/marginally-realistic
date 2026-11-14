"""
nb0 backend (CPU is fine):  python -m cvpr27.benchmark

 1. builds the train / held-out caches (skipped if present)
 2. decides FVD vs KID once for every later run (videofeat.decide_metric -> video_metric.json)
 3. E1: floors (true theta; pixels->theta uint8; pixel noise 0.02/0.05/0.1), the corruption
    self-test (mispaired, lag 3 frames, torque corrupted-but-globally-balanced) and the noise sweep,
    ALL on the 2,048 held-out real clips
 4. writes <root>/benchmark/floors.json and selftest_table.md (never overwrites; skips if present)

Nothing here is hardcoded: every number is computed in this run. Rule 5 gates: stop if the extractor's
invalid-frame fraction on clean real data exceeds 1% or a clean floor exceeds 1e-2.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from cvpr27 import evalmetrics as em
from cvpr27 import physvid as pv
from cvpr27.common import atomic_json, drive_root, git_info, new_path
from cvpr27.data import build_all, load_split
from cvpr27.videofeat import decide_metric

KEYS = ["r_energy", "r_energy_win_max", "kin_rad_max"]
NOISE = (0.02, 0.05, 0.1)


def row(per):
    s = em.summarise(per)
    return {"n": int(len(per["r_energy"])), **{k: s[k] for k in em.DIAG_KEYS + ["win_over_glob"]},
            "invalid_frame_frac": s["invalid_frame_frac"], "invalid_clip_frac": s["invalid_clip_frac"],
            "ratio_of_medians_win_over_glob": s["ratio_of_medians_win_over_glob"]}


def run():
    out_dir = drive_root() / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "floors.json").exists():
        print("floors.json exists -> nb0 already done; printing it.")
        return json.loads((out_dir / "floors.json").read_text())
    t0 = time.time()
    print("cache:", json.dumps(build_all()))
    held = load_split("heldout")
    vid, th, om, tau = held["video"], held["theta"].astype(np.float64), held["omega"].astype(np.float64), \
        held["tau"].astype(np.float64)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_info = decide_metric(vid[:1024], dev)          # self-check on 2 x 512 clips
    print(f"video metric: {metric_info['kind'].upper()} flavor={metric_info['flavor']}")
    for att in metric_info["attempts"]:
        print("  attempt", att["flavor"], "ok" if att.get("ok") else f"FAILED: {att.get('reason') or att.get('check')}")

    # precompute + cache the real-reference features now, so parallel training notebooks only read them
    from cvpr27.videofeat import load_metric
    load_metric(dev).set_reference(vid)
    res = {"n_heldout": int(len(vid)), "pv": {"T": pv.T, "DT": pv.DT, "WIN": pv.WIN, "STRIDE": pv.STRIDE},
           "git": git_info(), "video_metric": metric_info, "conditions": {}}
    C = res["conditions"]

    # -- floors
    d = pv.diagnostics(th, om, tau)
    d["invalid_frac_clip"] = np.zeros(len(th))              # no rendering, so no invalid frames
    d["win_over_glob"] = d["r_energy_win_max"] / np.maximum(d["r_energy"], 1e-12)
    C["floor_true_theta"] = row(d)
    C["floor_pixels_uint8"] = row(em.score_clips(vid, om, tau))

    # -- corruptions (applied to REAL held-out clips)
    rng = np.random.default_rng(0)
    th8, _ = pv.extract_theta(vid)
    perm = rng.permutation(len(vid))
    C["corrupt_mispaired"] = row(em.score_clips(vid[perm], om, tau))
    lag = np.concatenate([np.repeat(vid[:, :1], 3, 1), vid[:, :-3]], 1)
    C["corrupt_lag3"] = row(em.score_clips(lag, om, tau))
    C["corrupt_orth_torque"] = row(em.score_clips(vid, om, pv.orthogonal_torque_corruption(om, tau, rng)))

    # -- noise-robustness sweep (noise added to the float render, then re-quantised to uint8)
    vf = pv.render(th)
    for s in NOISE:
        noisy = pv.to_uint8(np.clip(vf + rng.normal(0, s, vf.shape).astype(np.float32), 0, 1))
        C[f"pixel_noise_{s}"] = row(em.score_clips(noisy, om, tau))

    # -- Rule 5 gates
    f = C["floor_pixels_uint8"]
    bad = []
    if f["invalid_frame_frac"] > 0.01:
        bad.append(f"extractor invalid-frame fraction on real data {f['invalid_frame_frac']:.4f} > 1%")
    for k in KEYS:
        if f[k]["median"] > 1e-2:
            bad.append(f"pixels->theta floor {k} median {f[k]['median']:.3e} > 1e-2")
    res["rule5_gate_failures"] = bad
    res["seconds"] = round(time.time() - t0, 1)
    atomic_json(out_dir / "floors.json", res)
    if bad:
        raise RuntimeError("RULE 5 STOP (floors.json written for inspection). " + " | ".join(bad))

    # -- markdown table
    names = list(C)
    md = ["| condition | " + " | ".join(f"{k} (median / p95)" for k in KEYS) + " | invalid frames |",
          "|---|" + "---|" * (len(KEYS) + 1)]
    for c in names:
        md.append(f"| {c} | " + " | ".join(f"{C[c][k]['median']:.2e} / {C[c][k]['p95']:.2e}" for k in KEYS)
                  + f" | {C[c]['invalid_frame_frac']:.4f} |")
    new_path(out_dir / "selftest_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return res


if __name__ == "__main__":
    run()
