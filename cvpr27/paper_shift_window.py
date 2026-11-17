"""
Three CPU-only experiments (pendulum, T=32, DT=0.04, WIN=8; reference R = 512 clips seed 1, test clips seed 2, NEW calibration set = 512 nominal clips, seed 3).
    python -m cvpr27.paper_shift_window   ->  $PHYSGEN_ROOT/paper_figs/shift_window.{md,json}       (no FVD; new files only)

E1  valid shifts (correct physics, different distribution): (a) forcing amplitude x1.5, (b) initial omega std x1.5, (c) forcing frequency band [1,12] rad/s.
    Nominal g, c, renderer, dt, T. Same protocol as results_fast (ap.eval_cond: W1, MMD^2 m=128 subsets with real-vs-real range, medians of r_E / r_E^win / r_K, invalid
    frames, AUROC vs R) plus: fraction of clips over the top (true max|theta| > pi) and the FALSE-POSITIVE RATE of each check at the threshold = 95th percentile of the
    calibration scores (5% FPR on nominal data).
E2  motion-feature MMD baseline: per clip from pixel-extracted theta_hat and the gyro: [std theta, max|theta|, std omega, autocorr(theta) at lags 1..8, dominant frequency and
    spectral centroid of theta (FFT, rad/s), mean|dtheta|] (14 features), standardised on R, RBF MMD^2 (median heuristic on R), subsets m=128 without replacement (B=200),
    with the real-vs-real range (A vs R).
E3  window length L in {4,8,12,16} frames (L+1 Simpson points, stride 1): AUROC and TPR at 5% FPR (calibration threshold) of r_E^win(L); r_E (global) and r_K (window 8) too.
"""
from __future__ import annotations

import json

import numpy as np

from cvpr27 import audit_paper as ap
from cvpr27 import evalmetrics as em
from cvpr27 import physvid as pv
from cvpr27.common import atomic_json, drive_root, git_info, new_path
from cvpr27.data import Norm
from scipy.integrate import simpson

OUT = drive_root() / "paper_figs"
N, SEED_CAL = 512, 3


def sim_custom(n, seed, amp=1.0, om_std=1.0, flo=None, fhi=None):
    """physvid.simulate with an optional forcing amplitude factor, initial-omega std and forcing band; identical to pv.simulate at the defaults (same rng stream)."""
    old = (pv.TAU_AMP, pv.F_LO, pv.F_HI)
    pv.TAU_AMP, pv.F_LO, pv.F_HI = old[0] * amp, flo if flo is not None else old[1], fhi if fhi is not None else old[2]
    try:
        rng = np.random.default_rng(seed)
        w, a, p = pv._forcing_params(rng, n)
        th = rng.normal(0, 0.3, size=n)
        om = rng.normal(0, om_std, size=n)
        h = pv.DT / pv.SUB

        def f(t, th, om):
            acc = (pv._tau(t, w, a, p) - pv.C * om - pv.M * pv.G * pv.L * np.sin(th)) / (pv.M * pv.L * pv.L)
            return om, acc
        TH, OM = np.empty((n, pv.T)), np.empty((n, pv.T))
        t = 0.0
        for k in range(pv.T):
            TH[:, k], OM[:, k] = th, om
            for _ in range(pv.SUB):
                k1 = f(t, th, om); k2 = f(t + h / 2, th + h / 2 * k1[0], om + h / 2 * k1[1]); k3 = f(t + h / 2, th + h / 2 * k2[0], om + h / 2 * k2[1]); k4 = f(t + h, th + h * k3[0], om + h * k3[1])
                th = th + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); om = om + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]); t += h
        tau = np.stack([pv._tau(tt, w, a, p) for tt in np.arange(pv.T) * pv.DT], 1)
        return {"theta": TH, "omega": OM, "tau": tau}
    finally:
        pv.TAU_AMP, pv.F_LO, pv.F_HI = old


def clip_from(sim):
    return {"video": pv.to_uint8(pv.render(sim["theta"])), "omega": sim["omega"].astype(np.float64), "tau": sim["tau"].astype(np.float64),
            "over_top": float((np.abs(sim["theta"]).max(1) > np.pi).mean())}


def scores_all(video, om, tau, Ls=(4, 8, 12, 16)):
    """Per-clip: r_E (global), r_E^win(L) for each L, r_K (window 8). Pixel-extracted theta, audit formulas."""
    th, ok = pv.extract_theta(video)
    scale = pv._scale(th, om)
    out = {"r_E": pv.energy_residual(th, om, tau) / scale, "invalid": (~ok).mean(1)}
    for L in Ls:
        ws = [(a, a + L + 1) for a in range(0, pv.T - L)]
        out[f"r_Ewin{L}"] = np.stack([pv.energy_residual(th, om, tau, a, b) for a, b in ws], 1).max(1) / scale
    ws8 = [(a, a + 9) for a in range(0, pv.T - 8)]
    out["r_K"] = np.stack([np.abs(th[:, b - 1] - th[:, a] - simpson(om[:, a:b], dx=pv.DT, axis=1)) for a, b in ws8], 1).max(1)
    return out, th


def motion_features(th, om):
    n, T = th.shape
    x = th - th.mean(1, keepdims=True)
    den = (x ** 2).sum(1) + 1e-12
    ac = np.stack([(x[:, :-k] * x[:, k:]).sum(1) / den for k in range(1, 9)], 1)
    spec = np.abs(np.fft.rfft(x, axis=1))[:, 1:]                        # drop DC
    freqs = 2 * np.pi * np.arange(1, T // 2 + 1) / (T * pv.DT)
    dom = freqs[spec.argmax(1)]
    cen = (spec * freqs).sum(1) / (spec.sum(1) + 1e-12)
    return np.concatenate([th.std(1, keepdims=True), np.abs(th).max(1, keepdims=True), om.std(1, keepdims=True), ac, dom[:, None], cen[:, None],
                           np.abs(np.diff(th, axis=1)).mean(1, keepdims=True)], 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    root = drive_root() / "audit_cache"
    rng = np.random.default_rng(404)
    # sanity: custom simulator == physvid.simulate at defaults
    s0, s1 = pv.simulate(64, seed=ap.SEED_TEST), sim_custom(64, ap.SEED_TEST)
    assert all(np.allclose(s0[k], s1[k]) for k in s0), "sim_custom differs from pv.simulate at the defaults"

    def cached(name, fn):
        p = root / f"{name}_n{N}_T{pv.T}.npz"
        if p.exists():
            z = np.load(p); return {k: z[k] for k in z.files}
        d = fn(); np.savez(p, **d); return d
    R = ap.cached(root, f"ref_s1_n{N}", None)
    A = ap.cond_data("real_vs_real", N, root)
    cal = cached(f"cal_s{SEED_CAL}", lambda: clip_from(sim_custom(N, SEED_CAL)))
    shifts = {"shift_a_forcing_amp_x1.5": dict(amp=1.5), "shift_b_omega0_std_x1.5": dict(om_std=1.5), "shift_c_force_band_1_12": dict(flo=1.0, fhi=12.0)}
    SH = {k: cached(k, lambda kw=kw: clip_from(sim_custom(N, ap.SEED_TEST, **kw))) for k, kw in shifts.items()}
    over = {"real_vs_real": clip_from(sim_custom(N, ap.SEED_TEST))["over_top"], "reference_R": clip_from(sim_custom(N, 1))["over_top"], **{k: float(SH[k]["over_top"]) for k in SH}}
    train = pv.simulate(ap.N_NORM, seed=ap.SEED_NORM)
    norm = Norm({"gyro_mean": float(train["omega"].mean()), "gyro_std": float(train["omega"].std()), "torque_mean": float(train["tau"].mean()), "torque_std": float(train["tau"].std())})
    perR = em.score_clips(R["video"], R["omega"], R["tau"])
    fl = em.summarise(perR)
    if fl["r_energy"]["median"] > 1e-2 or fl["invalid_frame_frac"] > 0.01:
        raise RuntimeError("STOP: real floor check failed")
    ref = {"om": R["omega"], "tau": R["tau"], "norm": norm, "per": perR, "metric": None, "mmd_feats": em.mmd_features(R["video"], R["omega"], R["tau"], norm)}

    class Args: B, m_sub, with_fvd, n_fvd = 200, 128, False, 0
    # ---------------- calibration thresholds (5% FPR on nominal data)
    sc_cal, _ = scores_all(cal["video"], cal["omega"], cal["tau"])
    thr = {k: float(np.percentile(v, 95)) for k, v in sc_cal.items() if k != "invalid"}
    sc_R, th_R = scores_all(R["video"], R["omega"], R["tau"])
    sc_A, th_A = scores_all(A["video"], A["omega"], A["tau"])
    res = {"git": git_info(), "setup": {"T": pv.T, "DT": pv.DT, "WIN": pv.WIN, "n": N, "seeds": {"reference": 1, "test": 2, "calibration": SEED_CAL, "norm_stats": 0}},
           "thresholds_5pct_FPR_calibration_p95": thr, "calibration_floor_median": {k: float(np.median(v)) for k, v in sc_cal.items() if k != "invalid"}}
    # ---------------- E1
    e1 = {}
    for name, d in (("real_vs_real", A), *SH.items()):
        c = ap.eval_cond(name, d, ref, Args, rng)
        sc, _ = scores_all(d["video"], d["omega"], d["tau"])
        c["fpr_at_5pct_calibration"] = {k: float((sc[k] > thr[k]).mean()) for k in ("r_E", "r_Ewin8", "r_K")}
        c["over_the_top_frac"] = over[name]
        e1[name] = c
        print("E1", name, {k: round(v, 3) for k, v in c["fpr_at_5pct_calibration"].items()}, "AUROC", {k: round(v, 3) for k, v in c["auroc_vs_real"].items()}, flush=True)
    res["E1_valid_shifts"] = {"conditions": e1, "over_the_top_frac": over}
    # ---------------- E2 motion-feature MMD
    fR = motion_features(th_R, R["omega"])
    mu, sd = fR.mean(0), fR.std(0) + 1e-9
    z = lambda f: ((f - mu) / sd).astype(np.float32)
    conds = {"real_vs_real": A}
    for g in (0.95, 1.05, 1.1, 1.25):
        conds[f"gravity={g}"] = ap.cond_data(f"gravity={g}", N, root)
    for c_ in (0.0, 2.0, 4.0):
        conds[f"damping={c_}"] = ap.cond_data(f"damping={c_}", N, root)
    conds["lag=1"] = ap.cond_data("lag=1", N, root)
    conds["time_reversed"] = ap.cond_data("time_reversed", N, root)
    conds["cancelling_torque"] = ap.cond_data("orth_torque", N, root)
    conds.update(SH)
    pool0, e2, TH = None, {}, {}
    for name, d in conds.items():
        th, _ = pv.extract_theta(d["video"])
        TH[name] = th
        pool = em.MMDPool(z(fR), z(motion_features(th, d["omega"])), "cpu", None if pool0 is None else pool0.real_cache())
        pool0 = pool0 or pool
        ar = np.arange(N)
        e2[name] = em.subsample_stat(lambda gi, ri: pool.mmd2(gi, ri), N, N, 128, 200, rng, pool.mmd2(ar, ar))
    hi = e2["real_vs_real"]["subsample_ci95"][1]
    for name, s in e2.items():
        s["leaves_real_range"] = bool(s["subsample_median"] > hi)
        s["range_disjoint_from_real"] = bool(s["subsample_ci95"][0] > hi)
        s["median_over_real_upper"] = float(s["subsample_median"] / hi) if hi > 0 else None
        print("E2", name, round(s["subsample_median"], 5), s["leaves_real_range"], flush=True)
    res["E2_motion_feature_mmd"] = {"n_features": int(fR.shape[1]), "features": "std theta, max|theta|, std omega, autocorr lags 1-8, dominant freq, spectral centroid, mean|dtheta|", "real_vs_real_upper_97.5": hi, "conditions": e2}
    # ---------------- E3 windows + fixed FPR
    e3, Ls = {}, (4, 8, 12, 16)
    e3conds = {"gravity=0.95": conds["gravity=0.95"], "gravity=1.05": conds["gravity=1.05"], "gravity=1.25": conds["gravity=1.25"], "damping=2.0": conds["damping=2.0"],
               "lag=1": conds["lag=1"], "cancelling_torque": conds["cancelling_torque"], "time_reversed": conds["time_reversed"]}
    e3["real_A_fpr"] = {k: float((sc_A[k] > thr[k]).mean()) for k in thr}
    for name, d in e3conds.items():
        sc, _ = scores_all(d["video"], d["omega"], d["tau"])
        e3[name] = {k: {"auroc": em.auroc(sc_R[k], sc[k]), "tpr_at_5pct_fpr": float((sc[k] > thr[k]).mean())} for k in ["r_E"] + [f"r_Ewin{L}" for L in Ls] + ["r_K"]}
    res["E3_window_fixed_fpr"] = e3
    atomic_json(OUT / "shift_window.json", res)
    write_md(res, Ls)


def write_md(r, Ls):
    f = ap._f
    s = lambda d: f"{f(d['subsample_median'])} [{f(d['subsample_ci95'][0])}, {f(d['subsample_ci95'][1])}]"
    E1 = r["E1_valid_shifts"]["conditions"]
    md = ["# Valid shifts, motion-feature MMD baseline, window length and fixed-FPR detection (pendulum, T=32, n=512 per condition)", "",
          "Reference R: 512 real clips (seed 1). Test clips: seed 2 (same random numbers as the audit; shifts change only the named property). Calibration set: 512 nominal clips, seed 3 (independent). "
          f"Thresholds at 5% FPR = 95th percentile of the calibration scores: r_E {f(r['thresholds_5pct_FPR_calibration_p95']['r_E'])}, r_E^win(8) {f(r['thresholds_5pct_FPR_calibration_p95']['r_Ewin8'])}, r_K {f(r['thresholds_5pct_FPR_calibration_p95']['r_K'])} rad.", "",
          "## E1: valid shifts (CORRECT physics, different distribution): does the audit false-alarm?", "",
          "FPR = fraction of the condition's clips above the 5%-FPR threshold (5% expected for real). MMD² cell: median [2.5-97.5] over 200 subsets of m=128; W1/physics: point [95% bootstrap CI]. AUROC: reference = negatives (0.5 = indistinguishable from real).", "",
          "| condition | W1 gyro | W1 torque | MMD² | r_E med [IQR] | r_E^win med | r_K med [IQR] (rad) | invalid frames | over the top | AUROC r_E / win / r_K | FPR@5% r_E / win / r_K |", "|" + "---|" * 12]
    ot = r["E1_valid_shifts"]["over_the_top_frac"]
    for name, c in E1.items():
        p, w = c["physical"], c["w1"]
        md.append(f"| {name} | {f(w['w1_gyro'])} [{f(w['w1_gyro_ci95'][0])}, {f(w['w1_gyro_ci95'][1])}] | {f(w['w1_torque'])} [{f(w['w1_torque_ci95'][0])}, {f(w['w1_torque_ci95'][1])}] | {s(c['mmd'])} | "
                  f"{f(p['r_energy']['median'])} [{f(p['r_energy']['q25'])}, {f(p['r_energy']['q75'])}] | {f(p['r_energy_win_max']['median'])} | {f(p['kin_rad_max']['median'])} [{f(p['kin_rad_max']['q25'])}, {f(p['kin_rad_max']['q75'])}] | "
                  f"{f(c['invalid_frame_frac'])} | {f(ot[name])} | " + " / ".join(f(c['auroc_vs_real'][k]) for k in ap.KEYS) + " | " + " / ".join(f(c["fpr_at_5pct_calibration"][k]) for k in ("r_E", "r_Ewin8", "r_K")) + " |")
    e2 = r["E2_motion_feature_mmd"]
    md += ["", "## E2: motion-feature MMD² baseline (14 features from pixel-extracted θ and the gyro, standardised on R)", "",
           f"Features: {e2['features']}. RBF MMD² with median heuristic on R; median [2.5-97.5] over 200 subsets of m=128. Two criteria for leaving the real range, both against the real-vs-real 97.5th percentile ({f(e2['real_vs_real_upper_97.5'])}): the condition's MEDIAN above it (loose), and its whole 2.5-97.5 range above it (strict).", "",
           "| condition | MMD² median [range] | full-n | × real upper | median above real upper | range disjoint from real (lower end > real upper) |", "|---|---|---|---|---|---|"]
    for name, x in e2["conditions"].items():
        md.append(f"| {name} | {s(x)} | {f(x['full_n_point'])} | {f(x['median_over_real_upper'])} | {'**yes**' if x['leaves_real_range'] else 'no'} | {'**yes**' if x['range_disjoint_from_real'] else 'no'} |")
    md += ["", "Note: the features use θ̂ and the gyro only, never the torque, so a torque-only error (cancelling_torque) is invisible to this baseline by construction.", "",
           "## E3: window length and TPR at a fixed 5% FPR", "", f"False-positive rate of the calibration thresholds on real test clips (A, expected ≈5%): " +
           ", ".join(f"{k} {f(v)}" for k, v in r["E3_window_fixed_fpr"]["real_A_fpr"].items()), "",
           "Cells: AUROC / TPR@5%FPR for r_E^win(L), L = window length in frames (L+1 points, stride 1).", "",
           "| condition | " + " | ".join(f"L={L}" for L in Ls) + " |", "|" + "---|" * (len(Ls) + 1)]
    names = [k for k in r["E3_window_fixed_fpr"] if k != "real_A_fpr"]
    for k in names:
        md.append(f"| {k} | " + " | ".join(f"{f(r['E3_window_fixed_fpr'][k][f'r_Ewin{L}']['auroc'])} / {f(r['E3_window_fixed_fpr'][k][f'r_Ewin{L}']['tpr_at_5pct_fpr'])}" for L in Ls) + " |")
    md += ["", "Global r_E and window-8 r_K (AUROC / TPR@5%FPR):", "", "| condition | r_E | r_K (window 8) |", "|---|---|---|"]
    for k in names:
        x = r["E3_window_fixed_fpr"][k]
        md.append(f"| {k} | {f(x['r_E']['auroc'])} / {f(x['r_E']['tpr_at_5pct_fpr'])} | {f(x['r_K']['auroc'])} / {f(x['r_K']['tpr_at_5pct_fpr'])} |")
    new_path(OUT / "shift_window.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
