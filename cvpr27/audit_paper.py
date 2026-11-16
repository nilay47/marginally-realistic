"""
Audit experiments on REAL simulated data, run LOCALLY (MPS/CPU), no trained models.

    export PHYSGEN_ROOT=~/physgen_local PV_T=32 PV_DT=0.04 PV_WIN=8
    python -m cvpr27.audit_paper --time_only      # checks MPS, builds data, times I3D on 64 clips, prints the projected runtime, stops
    python -m cvpr27.audit_paper                  # everything -> $PHYGEN_ROOT/audit/results_audit.{json,md}, fig_audit_A/B.{pdf,png}

R2  time reversal of real clips (re-uses benchmark.e8_time_reversal): real (half A vs half R), time_reversed (video reversed, gyro reversed
    and negated, torque reversed), video_reversed_only.
R3  wrong-physics worlds: the clips are simulated + rendered with ONE parameter changed (same random forcing / initial state / seed as the
    real test set, so only that parameter differs) and audited with NOMINAL physics: gravity x{0.8,0.9,0.95,1.05,1.1,1.25}; damping
    x{0,0.5,2,4}; torque-sensor gain x{0.9,1.1,1.25} (reported torque = gain * true torque; video and gyro stay true); video-sensor lag
    {1,2,4} frames (video delayed, first frame repeated, same convention as the E7 lag corruption).
Two tiers (decided from the timed I3D cost, 1.57 s/clip on MPS): EVERY condition, n=512 test clips vs a fixed 512-clip real reference R (disjoint seed):
W1 (gyro, torque), MMD, the three physics diagnostics (median + IQR, bootstrap CI of the median), per-sample AUROC (real R vs condition).
FVD (I3D-R50 via videofeat, pytorchvideo flavour = the one nb0 selected, full 224x224 protocol) ONLY on the trimmed set FVD_SET and the time-reversal
conditions, at n_fvd=256 (the first 256 test clips vs the first 256 reference clips), with the real-vs-real baseline at the same n. Set-level metrics
(FVD, MMD) use subsampling without replacement with m=128 for EVERY condition including the baseline; per-sample metrics use the percentile bootstrap.
R2 is computed with the same code as R3 (benchmark.time_reverse on the real test clips), so the real_vs_real baseline and the I3D features are shared.
Data are deterministic (seeds recorded) and cached under $PHYSGEN_ROOT/audit_cache; outputs are new files only.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from cvpr27 import benchmark as bm
from cvpr27 import evalmetrics as em
from cvpr27 import physvid as pv
from cvpr27.common import atomic_json, drive_root, fmt_s, git_info, new_path
from cvpr27.data import Norm, sensor_stats

FVD_ORDER = ["real_vs_real", "time_reversed", "video_reversed_only", "gravity=1.1", "damping=0.0", "lag=4", "torque_gain=1.25", "gravity=1.25",
             "damping=4.0", "gravity=0.9", "lag=1", "damping=2.0", "torque_gain=1.1"]
FVD_SET = {"gravity=0.9", "gravity=1.1", "gravity=1.25", "damping=0.0", "damping=2.0", "damping=4.0", "torque_gain=1.1", "torque_gain=1.25",
           "lag=1", "lag=4", "real_vs_real", "time_reversed", "video_reversed_only"}
SEED_REF, SEED_TEST, SEED_NORM, N_NORM = 1, 2, 0, 10_000
KEYS = ["r_energy", "r_energy_win_max", "kin_rad_max"]
SWEEPS = {  # name -> (axis label, [(condition name, value)])
    "gravity": ("gravity g x factor", [0.8, 0.9, 0.95, 1.05, 1.1, 1.25]),
    "damping": ("damping c x factor", [0.0, 0.5, 2.0, 4.0]),
    "torque_gain": ("torque-sensor gain", [0.9, 1.1, 1.25]),
    "lag": ("video-sensor lag (frames)", [1, 2, 4]),
}


@contextlib.contextmanager
def world(g=1.0, c=1.0):
    """Temporarily change gravity / damping in the SIMULATOR only (the audit always uses nominal physvid.G, physvid.C)."""
    g0, c0 = pv.G, pv.C
    pv.G, pv.C = g0 * g, c0 * c
    try:
        yield
    finally:
        pv.G, pv.C = g0, c0


def lag_video(v, k):
    return np.concatenate([np.repeat(v[:, :1], k, 1), v[:, :-k]], 1)


def make(seed, n, g=1.0, c=1.0, gain=1.0, lag=0):
    with world(g, c):
        sim = pv.simulate(n, seed=seed)
    v = pv.to_uint8(pv.render(sim["theta"]))
    if lag:
        v = lag_video(v, lag)
    return {"video": v, "omega": sim["omega"].astype(np.float64), "tau": (gain * sim["tau"]).astype(np.float64)}


def cached(root: Path, name, fn):
    p = root / f"{name}_T{pv.T}.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in ("video", "omega", "tau")}
    d = fn()
    np.savez(p.with_suffix(".tmp.npz"), **d)
    p.with_suffix(".tmp.npz").rename(p)
    return d


def conditions(n):
    out = {"real_vs_real": dict()}
    for name, (_, vals) in SWEEPS.items():
        for v in vals:
            out[f"{name}={v}"] = {"gravity": dict(g=v), "damping": dict(c=v), "torque_gain": dict(gain=v), "lag": dict(lag=int(v))}[name]
    return out


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def i3d_metric(dev):
    from cvpr27.videofeat import VideoMetric
    return VideoMetric("fvd", dev, None, "pytorchvideo")


def ci(fn, n, B, rng):
    v = np.array([fn(rng.integers(0, n, n)) for _ in range(B)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def eval_cond(name, d, ref, a, rng):
    v, om, tau = d["video"], d["omega"], d["tau"]
    n = len(v)
    per = em.score_clips(v, om, tau)
    r = {"n": n, "invalid_frame_frac": float((per["invalid_frac_clip"]).mean())}
    r["physical"] = {k: {"median": float(np.median(per[k])), "q25": float(np.percentile(per[k], 25)), "q75": float(np.percentile(per[k], 75)),
                         "median_ci95": ci(lambda i, k=k: np.median(per[k][i]), n, a.B, rng)} for k in KEYS}
    w = em.sensor_w1(ref["om"], ref["tau"], om, tau)
    r["w1"] = {**w, "w1_gyro_ci95": ci(lambda i: em.sensor_w1(ref["om"], ref["tau"], om[i], tau[i])["w1_gyro"], n, a.B, rng),
               "w1_torque_ci95": ci(lambda i: em.sensor_w1(ref["om"], ref["tau"], om[i], tau[i])["w1_torque"], n, a.B, rng)}
    pool = em.MMDPool(ref["mmd_feats"], em.mmd_features(v, om, tau, ref["norm"]), "cpu")
    ar = np.arange(n)
    r["mmd"] = em.subsample_stat(lambda gi, ri: pool.mmd2(gi, ri), n, n, a.m_sub, a.B, rng, pool.mmd2(ar, ar))
    r["fvd"] = None
    if getattr(a, "with_fvd", False) and name in FVD_SET:
        nf = a.n_fvd
        p = ref["metric"].prep(v[:nf])
        r["fvd"] = em.subsample_stat(lambda gi, ri: ref["metric"].value_pair(p, gi, ri), nf, nf, a.m_sub, a.B, rng, ref["metric"].value(p))
        r["fvd"]["n_fvd"] = int(nf)
        em.check_finite("fvd", r["fvd"]["full_n_point"])
    r["auroc_vs_real"] = {k: em.auroc(ref["per"][k], per[k]) for k in KEYS}
    return r


ALL_EXTRA = ["time_reversed", "video_reversed_only", "mispaired", "orth_torque"]


def cond_data(name, n, root):
    """Deterministic data for any condition (cached for the simulated worlds)."""
    A = lambda: cached(root, f"real_vs_real_s{SEED_TEST}_n{n}", lambda: make(SEED_TEST, n))
    if name in ("time_reversed", "video_reversed_only"):
        v, om, tau = bm.time_reverse(A()["video"], A()["omega"], A()["tau"], video_only=name == "video_reversed_only")
        return {"video": v, "omega": om, "tau": tau}
    if name in ("mispaired", "orth_torque"):
        rng = np.random.default_rng(77)
        perm = rng.permutation(n)
        perm = np.where(perm == np.arange(n), (perm + 1) % n, perm)               # never the clip's own video
        d = A()
        if name == "mispaired":
            return {"video": d["video"][perm], "omega": d["omega"], "tau": d["tau"]}
        return {"video": d["video"], "omega": d["omega"], "tau": pv.orthogonal_torque_corruption(d["omega"], d["tau"], rng)}
    kw = conditions(n)[name]
    return cached(root, f"{name.replace('=', '')}_s{SEED_TEST}_n{n}", lambda: make(SEED_TEST, n, **kw))


def setup(a):
    assert pv.T == 32 and abs(pv.DT - 0.04) < 1e-12 and pv.WIN == 8 and pv.TAU_AMP == 8.0, "export PV_T=32 PV_DT=0.04 PV_WIN=8 first"
    root = drive_root() / "audit_cache"
    root.mkdir(parents=True, exist_ok=True)
    R = cached(root, f"ref_s{SEED_REF}_n{a.n}", lambda: make(SEED_REF, a.n))
    return root, R


def out_dir_of(a):
    d = drive_root() / ("audit" if a.tag == "audit" else f"audit_{a.tag}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_fast(a):
    """All NON-FVD results, all conditions (full grid + time reversal + mispaired + orthogonal torque) at n: results_fast.json/.md + draft figures."""
    t0 = time.time()
    root, R = setup(a)
    out = out_dir_of(a)
    if (out / "results_fast.json").exists():
        print("results_fast.json exists"); return
    train_sim = pv.simulate(2000 if a.smoke else N_NORM, seed=SEED_NORM)
    norm = Norm(sensor_stats({"omega": train_sim["omega"], "tau": train_sim["tau"]}))
    per_R = em.score_clips(R["video"], R["omega"], R["tau"])
    floor = em.summarise(per_R)
    if floor["r_energy"]["median"] > 1e-2 or floor["invalid_frame_frac"] > 0.01:
        raise RuntimeError(f"STOP: real floor check failed: r_energy={floor['r_energy']['median']:.3g}, invalid={floor['invalid_frame_frac']:.3g}")
    ref = {"om": R["omega"], "tau": R["tau"], "norm": norm, "per": per_R, "metric": None,
           "mmd_feats": em.mmd_features(R["video"], R["omega"], R["tau"], norm)}
    a.with_fvd = False
    rng = np.random.default_rng(31)
    res = {"args": vars(a), "git": git_info(), "seeds": {"reference": SEED_REF, "test": SEED_TEST, "norm_stats": SEED_NORM},
           "physvid": {"T": pv.T, "DT": pv.DT, "WIN": pv.WIN, "TAU_AMP": pv.TAU_AMP, "G": pv.G, "C": pv.C}, "video_metric": None,
           "ci_methods": {"per_sample": f"percentile bootstrap B={a.B} (physics medians, W1)", "set_level": em.SUBSAMPLE_METHOD, "m": a.m_sub,
                          "n_physics_w1_mmd": a.n, "n_fvd": a.n_fvd, "fvd_conditions": FVD_ORDER},
           "real_reference_floor": floor, "R3": {"sweeps": {k: {"label": l, "values": v} for k, (l, v) in SWEEPS.items()}, "conditions": {}},
           "R2": {"conditions": {}}, "extra": {"conditions": {}}}
    names = list(conditions(a.n)) + ALL_EXTRA
    for i, name in enumerate(names):
        grp = "R2" if name in ("time_reversed", "video_reversed_only") else ("extra" if name in ("mispaired", "orth_torque") else "R3")
        c = {"params": conditions(a.n).get(name), **eval_cond(name, cond_data(name, a.n, root), ref, a, rng)}
        res[grp]["conditions"][name] = c
        print(f"[{i+1}/{len(names)}] {name}: W1 {c['w1']['w1_mean']:.3g} MMD {c['mmd']['subsample_median']:.3g} r_energy {c['physical']['r_energy']['median']:.3g} "
              f"kin {c['physical']['kin_rad_max']['median']:.3g} AUROC(r_energy) {c['auroc_vs_real']['r_energy']:.3f} ({fmt_s(time.time()-t0)})", flush=True)
    res["seconds"] = time.time() - t0
    atomic_json(out / "results_fast.json", res)
    new_path(out / "results_fast.md").write_text("\n".join(markdown(res)) + "\n")
    figures(res, out, "fast_")
    print(f"wrote results_fast.* and draft figures ({fmt_s(time.time()-t0)})", flush=True)


def run_fvd(a):
    """FVD, one condition at a time in FVD_ORDER; every condition is written to <out>/fvd/<name>.json the moment it finishes (resumable;
    reference and condition features are cached on disk). n_fvd clips per condition vs the first n_fvd reference clips, m_sub subsets."""
    t0 = time.time()
    root, R = setup(a)
    out = out_dir_of(a)
    fd = out / "fvd"
    fd.mkdir(exist_ok=True)
    dev = get_device()
    metric = i3d_metric(dev)
    metric.device = torch.device("cpu")
    feats_dir = root / "i3d_feats"
    feats_dir.mkdir(exist_ok=True)

    def feats(name, v):
        p = feats_dir / f"{name.replace('=', '')}_n{len(v)}_T{pv.T}.npy"
        if p.exists():
            return np.load(p)
        f = metric.ext.features(v)
        np.save(p.with_suffix(".tmp.npy"), f); p.with_suffix(".tmp.npy").rename(p)
        return f
    metric._install_ref(feats("ref", R["video"][:a.n_fvd]))
    print(f"reference features ready ({fmt_s(time.time()-t0)})", flush=True)
    rng = np.random.default_rng(41)
    for i, name in enumerate(FVD_ORDER):
        p = fd / f"{name.replace('=', '')}.json"
        if p.exists():
            print(f"[{i+1}/{len(FVD_ORDER)}] {name}: exists"); continue
        v = cond_data(name, a.n, root)["video"][:a.n_fvd]
        pr = {"feats": feats(name, v)}
        st = em.subsample_stat(lambda gi, ri: metric.value_pair(pr, gi, ri), a.n_fvd, a.n_fvd, a.m_sub, a.B, rng, metric.value(pr))
        st.update(n_fvd=int(a.n_fvd), condition=name, flavor="pytorchvideo I3D-R50 224x224", seconds=time.time() - t0)
        em.check_finite("fvd", st["full_n_point"])
        atomic_json(p, st)
        print(f"[{i+1}/{len(FVD_ORDER)}] {name}: FVD median {st['subsample_median']:.4g} [{st['subsample_ci95'][0]:.4g}, {st['subsample_ci95'][1]:.4g}] "
              f"full-n {st['full_n_point']:.4g} ({fmt_s(time.time()-t0)})", flush=True)


def run_report(a):
    """Merge results_fast.json + whatever FVD files exist -> results_audit.json/.md + Fig A/B (missing FVD marked). Re-runnable: writes a
    timestamped copy if the files already exist (nothing is overwritten)."""
    out = out_dir_of(a)
    res = json.loads((out / "results_fast.json").read_text())
    have = {}
    for p in (out / "fvd").glob("*.json"):
        st = json.loads(p.read_text()); have[st["condition"]] = st
    for grp in ("R3", "R2", "extra"):
        for k, c in res[grp]["conditions"].items():
            c["fvd"] = have.get(k)
    res["video_metric"] = {"name": "FVD", "flavor": "pytorchvideo I3D-R50 (Kinetics-400), 224x224; NOT comparable to published FVD"}
    res["fvd_available"] = sorted(have)
    res["fvd_missing"] = [k for k in FVD_ORDER if k not in have]
    sfx = "" if not (out / "results_audit.json").exists() else f"_{int(time.time())}"
    atomic_json(out / f"results_audit{sfx}.json", res)
    new_path(out / f"results_audit{sfx}.md").write_text("\n".join(markdown(res)) + "\n")
    figures(res, out, sfx.lstrip("_") + ("_" if sfx else ""))
    print(f"report written (FVD available: {len(have)}/{len(FVD_ORDER)}; missing: {res['fvd_missing']})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fast", "fvd", "report", "time"], help="fast: all non-FVD results; fvd: resumable FVD in priority order; report: merge + figures; time: I3D timing only")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--n_fvd", type=int, default=256)
    ap.add_argument("--m_sub", type=int, default=128)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--tag", default="audit")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.n, a.n_fvd, a.m_sub, a.B = 16, 8, 4, 6
    print(f"device: {get_device()} | PHYSGEN_ROOT={drive_root()}", flush=True)
    {"fast": run_fast, "fvd": run_fvd, "report": run_report}.get(a.mode, lambda a: None)(a)


# ------------------------------------------------------------------ report
def _f(x):
    return "n/a" if x is None else (f"{x:.3g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}")


def markdown(r):
    n, m, B, nf = r["args"]["n"], r["ci_methods"]["m"], r["args"]["B"], r["args"]["n_fvd"]
    md = ["# Audit on real simulated data (R2 time reversal, R3 wrong-physics worlds)", "",
          f"Physics, W1, MMD² and AUROC: n={n} test clips per condition vs one fixed real reference of n={n} (disjoint seed). "
          f"FVD ({(r['video_metric'] or {}).get('flavor', 'not computed in this file')}) is computed ONLY on the conditions marked, at n={nf} (first {nf} test clips vs first {nf} reference clips) "
          "because I3D feature extraction cost 1.57 s/clip locally; '—' = FVD not computed (yet). Audit physics is NOMINAL (g=9.81, c=0.4); each R3 world changes one simulator/sensor "
          "parameter with the same forcing and initial states as the real test set. FVD and MMD² cells: median [2.5-97.5 percentile] over "
          f"{B} subsets of m={m} drawn without replacement from test and reference (same m for the baseline, so ratios are at equal n; FVD and MMD² are at different n: {nf} and {n}); "
          f"W1 and physics: point [95% bootstrap CI, B={B}]; physics also shows the IQR. Real floor (reference clips): "
          f"r_energy {_f(r['real_reference_floor']['r_energy']['median'])}, win_max {_f(r['real_reference_floor']['r_energy_win_max']['median'])}, "
          f"kin {_f(r['real_reference_floor']['kin_rad_max']['median'])} rad.", ""]
    s = lambda d: "—" if d is None else f"{_f(d['subsample_median'])} [{_f(d['subsample_ci95'][0])}, {_f(d['subsample_ci95'][1])}]"
    head = ["| condition | FVD | W1 gyro | W1 torque | MMD² | r_energy med [IQR] | win_max med | kin_max med [IQR] (rad) | invalid frames | AUROC r_energy | AUROC win_max | AUROC kin |", "|" + "---|" * 12]
    def row(name, c):
        p, w = c["physical"], c["w1"]
        return (f"| {name} | {s(c['fvd'])} | {_f(w['w1_gyro'])} [{_f(w['w1_gyro_ci95'][0])}, {_f(w['w1_gyro_ci95'][1])}] | {_f(w['w1_torque'])} [{_f(w['w1_torque_ci95'][0])}, {_f(w['w1_torque_ci95'][1])}] | "
                f"{s(c['mmd'])} | {_f(p['r_energy']['median'])} [{_f(p['r_energy']['q25'])}, {_f(p['r_energy']['q75'])}] | {_f(p['r_energy_win_max']['median'])} | "
                f"{_f(p['kin_rad_max']['median'])} [{_f(p['kin_rad_max']['q25'])}, {_f(p['kin_rad_max']['q75'])}] | {_f(c['invalid_frame_frac'])} | " +
                " | ".join(_f(c["auroc_vs_real"][k]) for k in KEYS) + " |")
    md += ["## R3 wrong-physics worlds", ""] + head + [row(k, c) for k, c in r["R3"]["conditions"].items()]
    base = r["R3"]["conditions"]["real_vs_real"]
    md += ["", "## R2 consistent time reversal of real clips (baseline = real_vs_real above)", ""] + head + \
          [row("real_vs_real", base)] + [row(k, c) for k, c in r["R2"]["conditions"].items()]
    if r.get("extra", {}).get("conditions"):
        md += ["", "## Extra corruptions (same setup; FVD not computed)", ""] + head + [row(k, c) for k, c in r["extra"]["conditions"].items()]
    bf = base["fvd"]["subsample_median"] if base["fvd"] else None
    if bf:
        md += ["", "FVD / real-vs-real baseline at equal m: " + "; ".join(
            f"{k}: {_f(c['fvd']['subsample_median'] / max(bf, 1e-12))}x (subsample ranges disjoint from baseline: {c['fvd']['subsample_ci95'][0] > base['fvd']['subsample_ci95'][1]})"
            for k, c in {**r["R3"]["conditions"], **r["R2"]["conditions"]}.items() if c["fvd"] and k != "real_vs_real")]
        if r.get("fvd_missing"):
            md += ["", "FVD still missing for: " + ", ".join(r["fvd_missing"])]
        md += ["", f"Full-n FVD point estimates (n={nf} vs n={nf}, not comparable with the m={m} cells): " + "; ".join(
            f"{k}: {_f(c['fvd']['full_n_point'])}" for k, c in {**r["R3"]["conditions"], **r["R2"]["conditions"]}.items() if c["fvd"])]
    return md


def figures(res, d, prefix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cvpr27.figures import C, INK2, save, style
    style()
    conds = res["R3"]["conditions"]
    base = conds["real_vs_real"]
    series = [("FVD", lambda c: c["fvd"]["subsample_median"] if c["fvd"] else float("nan"), "#2a78d6", "o"), ("W1 (mean)", lambda c: c["w1"]["w1_mean"], "#eb6834", "s"),
              ("MMD²", lambda c: c["mmd"]["subsample_median"], "#1baf7a", "^"),
              ("r_energy", lambda c: c["physical"]["r_energy"]["median"], "#6250d6", "D"),
              ("kin mismatch", lambda c: c["physical"]["kin_rad_max"]["median"], "#e87ba4", "v")]
    fig, axs = plt.subplots(1, 4, figsize=(9.0, 2.7), sharey=True)
    for ax, (name, (label, vals)) in zip(axs, SWEEPS.items()):
        xs = [v - 1 if name != "lag" else v for v in vals]
        order = np.argsort(xs)
        xs = [xs[i] for i in order]
        vs = [vals[i] for i in order]
        x0 = 0.0
        for lab, f, col, mk in series:
            if lab == "FVD" and base["fvd"] is None:
                continue
            b = max(f(base), 1e-12)
            ok = [(x, v) for x, v in zip(xs, vs) if lab != "FVD" or conds[f"{name}={v}"]["fvd"] is not None]
            xx = [x0] + [x for x, _ in ok]
            yy = [1.0] + [max(f(conds[f"{name}={v}"]) / b, 1e-3) for _, v in ok]
            o = np.argsort(xx)
            ax.plot(np.array(xx)[o], np.array(yy)[o], color=col, marker=mk, label=lab + (" (n=%d)" % res["args"]["n_fvd"] if lab == "FVD" else ""))
        ax.axhline(1, color=INK2, lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel("lag (frames)" if name == "lag" else f"{label.split(' x')[0]} error (factor - 1)")
        ax.set_title(name.replace("_", " "))
    axs[0].set_ylabel("metric / real-vs-real baseline (log)")
    axs[0].legend(fontsize=6.5, loc="upper center", ncol=1)
    if base["fvd"] is None:
        axs[0].text(0.02, 0.02, "FVD not computed yet", transform=axs[0].transAxes, fontsize=6.5, color=INK2)
    save(fig, d, f"{prefix}fig_audit_A")
    e8 = res["R2"]["conditions"]
    names = list(e8)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    bf = base["fvd"]["subsample_median"] if base["fvd"] else None
    for xi, k in enumerate(names):
        if bf and e8[k]["fvd"]:
            ax[0].bar(xi, e8[k]["fvd"]["subsample_median"] / max(bf, 1e-12), color="#2a78d6")
        else:
            ax[0].text(xi, 1.0, "FVD not computed yet", ha="center", va="bottom", fontsize=6.5, color=INK2, rotation=90)
    ax[0].set_xticks(range(len(names)), names)
    ax[0].axhline(1, color=INK2, ls=":", lw=0.7)
    ax[0].set_yscale("log")
    ax[0].set(title="FVD / real-vs-real baseline", ylabel="ratio (log; 1 = undetected)")
    w = 0.27
    for j, (k, col) in enumerate(zip(KEYS, ["#6250d6", "#1baf7a", "#e87ba4"])):
        ax[1].bar(np.arange(len(names)) + (j - 1) * w, [e8[nm]["auroc_vs_real"][k] for nm in names], w, color=col, label=k)
    ax[1].axhline(0.5, color=INK2, ls=":", lw=0.7)
    ax[1].set_xticks(range(len(names)), names)
    ax[1].set(title="physics AUROC (real vs reversed)", ylim=(0.4, 1.02))
    ax[1].legend(fontsize=6.5, loc="lower right")
    for x in ax:
        x.tick_params(axis="x", labelsize=7)
    save(fig, d, f"{prefix}fig_audit_B")


if __name__ == "__main__":
    main()
