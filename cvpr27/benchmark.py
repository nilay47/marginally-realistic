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


# ------------------------------------------------------------------ E8: time reversal
E8_DIAG = KEYS + ["r_energy_win_med", "kin_rad_med"]


def time_reverse(vid, om, tau, video_only=False):
    """Reverse the clip in time. Full reversal: frames reversed, gyro reversed AND negated (omega = dtheta/dt flips
    sign), torque reversed. video_only=True: reverse only the video and leave the sensors forward."""
    v = np.ascontiguousarray(vid[:, ::-1])
    if video_only:
        return v, om, tau
    return v, -np.ascontiguousarray(om[:, ::-1]), np.ascontiguousarray(tau[:, ::-1])


def _ci(fn, n, B, rng):
    v = np.array([fn(rng.integers(0, n, n)) for _ in range(B)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def e8_time_reversal(held, norm, metric=None, B=200, seed=8, device="cpu", n=1024, m_sub=512) -> dict:
    """E8: does each metric detect CONSISTENT time reversal of real clips? Needs real data only (no trained models).

    Split the held-out set into disjoint halves A (n clips, source of the test clips) and R (reference, n clips).
    Conditions: real_A (baseline: real vs real), time_reversed, video_reversed_only. All compared with R, so every
    number has the same n and no clip is on both sides. FVD/KID, W1, MMD and the physics medians get 95% bootstrap
    CIs over the test clips (B draws); set-level metrics (video metric, MMD) use SUBSAMPLING WITHOUT REPLACEMENT instead
    (B subsets of m_sub from A and from R, the same m_sub for every condition incl. the baseline; the full-n point
    estimate is reported separately); AUROC is real R (negatives) vs the condition (positives). The video metric is
    skipped when metric is None (CPU smoke test). The reference of `metric` is replaced here and restored on exit."""
    rng = np.random.default_rng(seed)
    perm = np.random.default_rng(seed).permutation(len(held["video"]))
    n = min(n, len(perm) // 2)
    m_sub = min(m_sub, n)
    iA, iR = perm[:n], perm[n:2 * n]
    g = lambda i: (held["video"][i], held["omega"][i].astype(np.float64), held["tau"][i].astype(np.float64))
    real_R, real_A = g(iR), g(iA)
    conds = {"real_vs_real": real_A, "time_reversed": time_reverse(*real_A),
             "video_reversed_only": time_reverse(*real_A, video_only=True)}
    per_R = em.score_clips(*real_R)
    mmd_R = em.mmd_features(*real_R, norm)
    old_ref = None
    if metric is not None:
        old_ref = metric.ref
        metric.set_reference_nocache(real_R[0])
    out = {"n": int(n), "B": B, "m_subsample": int(m_sub), "set_level_ci_method": em.SUBSAMPLE_METHOD,
           "per_sample_ci_method": "percentile bootstrap over test clips (physics medians, W1)", "split": {"reference_R": "held-out half R", "test": "disjoint held-out half A (reversed or not)"},
           "video_metric": None if metric is None else {"name": metric.name, "kind": metric.kind, "flavor": metric.flavor},
           "conditions": {}}
    try:
        for name, (v, om, tau) in conds.items():
            r = {}
            per = em.score_clips(v, om, tau)
            r["physical"] = em.summarise(per)
            r["physical_ci"] = {k: [float(np.median(per[k]))] + _ci(lambda i, k=k: np.median(per[k][i]), n, B, rng)
                                for k in E8_DIAG}
            r["invalid_frame_frac"] = r["physical"]["invalid_frame_frac"]
            w = em.sensor_w1(real_R[1], real_R[2], om, tau)
            r["w1_mean"] = w["w1_mean"]
            r["w1_mean_ci"] = _ci(lambda i: em.sensor_w1(real_R[1], real_R[2], om[i], tau[i])["w1_mean"], n, B, rng)
            pool = em.MMDPool(mmd_R, em.mmd_features(v, om, tau, norm), device)
            ar, ref_idx = np.arange(n), np.arange(n)
            r["mmd"] = em.subsample_stat(lambda gi, ri: pool.mmd2(gi, ri), n, n, m_sub, B, rng, pool.mmd2(ar, ref_idx))
            if metric is not None:
                p = metric.prep(v)
                r["video_metric"] = em.subsample_stat(lambda gi, ri: metric.value_pair(p, gi, ri), n, n, m_sub, B, rng,
                                                      metric.value(p))
                em.check_finite("video_metric", r["video_metric"]["full_n_point"])
            if name != "real_vs_real":
                r["auroc_vs_real"] = {k: em.auroc(per_R[k], per[k]) for k in E8_DIAG}
            out["conditions"][name] = r
    finally:
        if metric is not None:
            metric.ref = old_ref
    base = out["conditions"]["real_vs_real"]
    if metric is not None:
        for name in ("time_reversed", "video_reversed_only"):
            c = out["conditions"][name]
            cv, bv = c["video_metric"], base["video_metric"]
            c["video_metric_over_real_baseline_at_m"] = cv["subsample_median"] / max(bv["subsample_median"], 1e-12)
            c["video_metric_ci_disjoint_from_baseline_at_m"] = bool(cv["subsample_ci95"][0] > bv["subsample_ci95"][1])
    return out


def e8_markdown(r: dict) -> list:
    f = lambda x: "n/a" if x is None else (f"{x:.3g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}")
    ci = lambda x, c: f"{f(x)} [{f(c[0])}, {f(c[1])}]"
    vm = r["video_metric"]["name"] if r["video_metric"] else None
    md = ["## E8: consistent time reversal of real clips", "",
          f"Test clips: held-out half A (n={r['n']}), compared with the disjoint held-out half R. `real_vs_real` is the baseline "
          "(half A unmodified). `time_reversed`: frames reversed, gyro reversed and negated, torque reversed (all modalities consistent). "
          f"`video_reversed_only`: video reversed, sensors forward. Values: point [95% percentile-bootstrap CI over test clips, B={r['B']}] for W1 and the physics diagnostics; "
          f"for {vm or 'the video metric'} and MMD² the value is the median [2.5-97.5 percentile] over {r['B']} random subsets of m={r['m_subsample']} "
          "drawn WITHOUT replacement from the test set and from the reference (same m for every row, incl. the baseline); "
          "full-n point estimates are in a separate table.", "",
          "| condition | " + (f"{vm} | " if vm else "") + "W1 | MMD² | r_energy | win_max | kin_max (rad) |", "|---|" + "---|" * (5 + bool(vm))]
    for name, c in r["conditions"].items():
        pc = c["physical_ci"]
        sub = lambda d: ci(d["subsample_median"], d["subsample_ci95"])
        cells = ([sub(c["video_metric"])] if vm else []) + [ci(c["w1_mean"], c["w1_mean_ci"]), sub(c["mmd"])] + \
                [ci(*pc[k][:1], pc[k][1:]) for k in KEYS]
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    md += ["", f"Full-n point estimates (n={r['n']} vs n={r['n']}; NOT comparable with the m={r['m_subsample']} subsample values above, "
           "FVD is biased upward at smaller n):", "", "| condition | " + (f"{vm} (full n) | " if vm else "") + "MMD² (full n) |", "|---|" + "---|" * (1 + bool(vm))]
    md += [f"| {n} | " + (f"{f(c['video_metric']['full_n_point'])} | " if vm else "") + f"{f(c['mmd']['full_n_point'])} |" for n, c in r["conditions"].items()]
    if vm:
        md += [""] + [f"- {n}: {vm} median = {f(c['video_metric_over_real_baseline_at_m'])}x the real-vs-real baseline at m={r['m_subsample']}; "
                      f"subsample ranges disjoint from baseline: {c['video_metric_ci_disjoint_from_baseline_at_m']}"
                      for n, c in r["conditions"].items() if "video_metric_over_real_baseline_at_m" in c]
    md += ["", "Per-sample AUROC (negatives = real half R, positives = condition; 0.5 = chance, larger score = more impossible):", "",
           "| condition | " + " | ".join(E8_DIAG) + " |", "|---|" + "---|" * len(E8_DIAG)]
    md += [f"| {n} | " + " | ".join(f(c["auroc_vs_real"][k]) for k in E8_DIAG) + " |" for n, c in r["conditions"].items() if "auroc_vs_real" in c]
    return md


def run_e8(tag="e8", B=200, smoke=False):
    """Standalone E8 (real data only):  python -m cvpr27.benchmark --e8 [--tag e8]"""
    from cvpr27.data import Norm, sensor_stats
    out_dir = drive_root() / f"eval_{tag}"
    if (out_dir / "e8.json").exists():
        print(f"{out_dir}/e8.json exists: use a new --tag (nothing is overwritten)."); return json.loads((out_dir / "e8.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = None
    if not smoke:
        from cvpr27.videofeat import load_metric
        metric = load_metric(dev)
    held, norm = load_split("heldout"), Norm(sensor_stats(load_split("train")))
    if smoke:
        held = {k: (v[:256] if hasattr(v, "shape") else v) for k, v in held.items()}
    t0 = time.time()
    res = e8_time_reversal(held, norm, metric, B=10 if smoke else B, device=dev, n=64 if smoke else 1024)
    res.update(git=git_info(), seconds=round(time.time() - t0, 1), device=str(dev))
    atomic_json(out_dir / "e8.json", res)
    md = e8_markdown(res)
    new_path(out_dir / "e8.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return res


if __name__ == "__main__":
    import sys
    if "--e8" in sys.argv:
        a = sys.argv
        run_e8(a[a.index("--tag") + 1] if "--tag" in a else "e8", smoke="--smoke" in a)
    else:
        run()
