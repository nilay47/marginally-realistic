"""
Two CPU-only analyses for the paper (T=32 v2 setup, $PHYSGEN_ROOT=~/physgen_local); outputs in $PHYSGEN_ROOT/paper_figs (new files only).

  python -m cvpr27.paper_analyses valid   -> valid_only_physics.{md,json}
  python -m cvpr27.paper_analyses deriv   -> derivative_baseline.{md,json}

valid : joint_v2 final samples (gen_samples/joint_v2/samples_final.npz, 1024 ancestral) and real mispaired clips, vs the 512 real reference clips (seed 1).
        How the current code treats invalid frames: NOT dropped and NOT interpolated. evalmetrics.score_clips uses `th, ok = extract_theta(video)` and
        feeds th to physvid.diagnostics unchanged; `ok` is only used for the invalid-frame fraction. An invalid frame (peak < 0.3 or empty window) still
        yields the centroid around the brightest pixel (a fixed -135 deg when the window is empty); th is then np.unwrap'ed over time.
        Variants: (a) all clips, (b) clips with ZERO invalid frames, (c) every 9-frame window (WIN+1 Simpson points, stride 1) whose frames are all valid,
        pooled over clips (window-level r_E^win and r_K; no whole-clip r_E).
deriv : finite-difference ODE residual |theta_ddot + c*omega + g*sin(theta_hat) - tau| / std(real tau), median over frames per clip. raw: second
        difference of theta_hat. smoothed: Savitzky-Golay (window 7, order 3, deriv=2). gyro: omega_dot from the first (central) difference of the gyro,
        with sin(theta_hat). Real vs gravity x1.05 clips with the SAME pixel noise sigma on both sides, same clips/seeds/noise draws as noise_table.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.signal import savgol_filter
from scipy.integrate import simpson

from cvpr27 import audit_paper as ap
from cvpr27 import evalmetrics as em
from cvpr27 import physvid as pv
from cvpr27.common import atomic_json, drive_root, git_info, new_path

OUT = drive_root() / "paper_figs"


def q3(x):
    return {"median": float(np.median(x)), "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75)), "n": int(len(x))}


def window_arrays(th, om, tau):
    """Per-window scaled energy residual and kinematic mismatch, shape (n, n_windows); same formulas as physvid.diagnostics."""
    scale = pv._scale(th, om)
    ws = pv.windows()
    e = np.stack([pv.energy_residual(th, om, tau, a, b) for a, b in ws], 1) / scale[:, None]
    k = np.stack([np.abs(th[:, b - 1] - th[:, a] - simpson(om[:, a:b], dx=pv.DT, axis=1)) for a, b in ws], 1)
    return e, k, ws


def block(video, om, tau, per_ref, win_ref):
    th, ok = pv.extract_theta(video)
    per = pv.diagnostics(th, om, tau)
    n = len(video)
    clip_ok = ok.all(1)
    e, k, ws = window_arrays(th, om, tau)
    w_ok = np.stack([ok[:, a:b].all(1) for a, b in ws], 1)
    out = {"n_clips": n, "invalid_frame_frac": float((~ok).mean()), "clips_zero_invalid": int(clip_ok.sum()), "clips_zero_invalid_frac": float(clip_ok.mean()),
           "windows_total": int(w_ok.size), "windows_valid": int(w_ok.sum()), "windows_valid_frac": float(w_ok.mean())}
    keys = {"r_E": "r_energy", "r_E^win (per-clip worst window)": "r_energy_win_max", "r_K (per-clip worst window, rad)": "kin_rad_max"}
    for name, sel in (("a_all_clips", np.ones(n, bool)), ("b_zero_invalid_clips", clip_ok)):
        r = {"retained": int(sel.sum()), "retained_frac": float(sel.mean())}
        for lab, kk in keys.items():
            if sel.sum() >= 2:
                r[lab] = {**q3(per[kk][sel]), "auroc_vs_real": em.auroc(per_ref[kk], per[kk][sel])}
        out[name] = r
    c = {"retained": int(w_ok.sum()), "retained_frac": float(w_ok.mean())}
    if w_ok.sum() >= 2:
        c["r_E^win (window level)"] = {**q3(e[w_ok]), "auroc_vs_real_windows": em.auroc(win_ref["e"], e[w_ok])}
        c["r_K (window level, rad)"] = {**q3(k[w_ok]), "auroc_vs_real_windows": em.auroc(win_ref["k"], k[w_ok])}
    out["c_valid_windows_pooled"] = c
    return out


def valid():
    OUT.mkdir(parents=True, exist_ok=True)
    root = drive_root() / "audit_cache"
    R = ap.cached(root, "ref_s1_n512", None)
    per_ref = em.score_clips(R["video"], R["omega"], R["tau"])
    th_r, ok_r = pv.extract_theta(R["video"])
    e_r, k_r, _ = window_arrays(th_r, R["omega"], R["tau"])
    win_ref = {"e": e_r.ravel(), "k": k_r.ravel()}
    z = np.load(drive_root() / "gen_samples" / "joint_v2" / "samples_final.npz")
    assert z["video"].shape == (1024, pv.T, pv.H, pv.W)
    res = {"git": git_info(), "setup": {"T": pv.T, "DT": pv.DT, "WIN": pv.WIN, "window_points": pv.WIN + 1, "stride": pv.STRIDE},
           "reference": {"n_clips": 512, "invalid_frame_frac": float((~ok_r).mean()), "windows": int(e_r.size), "r_E": q3(per_ref["r_energy"]),
                         "r_E^win window level": q3(e_r.ravel()), "r_K window level": q3(k_r.ravel())},
           "current_code_invalid_frame_handling": "NOT dropped, NOT interpolated: score_clips -> extract_theta -> diagnostics(th) ignores the validity mask (used only to count invalid frames); invalid frames keep the argmax centroid (fixed -135 deg if the window is empty) and np.unwrap runs over time.",
           "joint_v2": block(z["video"], z["omega"].astype(np.float64), z["tau"].astype(np.float64), per_ref, win_ref)}
    A = ap.cond_data("mispaired", 512, root)
    res["real_mispaired"] = block(A["video"], A["omega"], A["tau"], per_ref, win_ref)
    atomic_json(OUT / "valid_only_physics.json", res)
    f = lambda x: f"{x:.3g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"
    cell = lambda d: "—" if d is None else f"{f(d['median'])} [{f(d['q25'])}, {f(d['q75'])}] (AUROC {f(d.get('auroc_vs_real', d.get('auroc_vs_real_windows')))})"
    md = ["# Valid-only physics (T=32; joint_v2 final samples n=1024 ancestral, real mispaired n=512; AUROC vs the 512 real reference clips)", "",
          "**Invalid-frame handling in the current code:** " + res["current_code_invalid_frame_handling"], "",
          f"Reference (real, 512 clips): invalid frame fraction {f(res['reference']['invalid_frame_frac'])}; r_E median {f(res['reference']['r_E']['median'])}; window-level r_E^win median {f(res['reference']['r_E^win window level']['median'])}; r_K {f(res['reference']['r_K window level']['median'])} rad.", "",
          "Cells: median [IQR] (AUROC, real reference = negatives). (a),(b) are clip-level (r_E^win and r_K = worst window of the clip). (c) pools every valid 9-frame window (WIN=8, stride 1) over clips and compares with the pooled windows of the real reference; there is no whole-clip r_E.", ""]
    for name in ("joint_v2", "real_mispaired"):
        r = res[name]
        md += [f"## {name}", "", f"invalid frame fraction {f(r['invalid_frame_frac'])}; clips with zero invalid frames {r['clips_zero_invalid']}/{r['n_clips']} ({f(r['clips_zero_invalid_frac'])}); valid windows {r['windows_valid']}/{r['windows_total']} ({f(r['windows_valid_frac'])})", "",
               "| variant | retained | r_E | r_E^win | r_K (rad) |", "|---|---|---|---|---|"]
        for nm, lab in (("a_all_clips", "(a) all clips"), ("b_zero_invalid_clips", "(b) zero-invalid clips")):
            x = r[nm]
            md.append(f"| {lab} | {x['retained']}/{r['n_clips']} ({f(x['retained_frac'])}) | {cell(x.get('r_E'))} | {cell(x.get('r_E^win (per-clip worst window)'))} | {cell(x.get('r_K (per-clip worst window, rad)'))} |")
        c = r["c_valid_windows_pooled"]
        md.append(f"| (c) valid windows pooled | {c['retained']}/{r['windows_total']} ({f(c['retained_frac'])}) | — | {cell(c.get('r_E^win (window level)'))} | {cell(c.get('r_K (window level, rad)'))} |")
        md.append("")
    new_path(OUT / "valid_only_physics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ------------------------------------------------------------------ derivative baseline
def deriv_scores(video, om, tau, tau_std):
    th, ok = pv.extract_theta(video)
    dt = pv.DT
    base = lambda thdd, om_, th_, ta: np.abs(thdd + pv.C * om_ + pv.G * np.sin(th_) - ta) / tau_std
    thdd = (th[:, 2:] - 2 * th[:, 1:-1] + th[:, :-2]) / dt ** 2
    raw = np.median(base(thdd, om[:, 1:-1], th[:, 1:-1], tau[:, 1:-1]), 1)
    thdd_s = savgol_filter(th, 7, 3, deriv=2, delta=dt, axis=1)
    smooth = np.median(base(thdd_s, om, th, tau), 1)
    omd = (om[:, 2:] - om[:, :-2]) / (2 * dt)
    gyro = np.median(base(omd, om[:, 1:-1], th[:, 1:-1], tau[:, 1:-1]), 1)
    return {"deriv_raw": raw, "deriv_smooth": smooth, "deriv_gyro": gyro}


def deriv(n=512, sigmas=(0.0, 0.02, 0.05, 0.1)):
    OUT.mkdir(parents=True, exist_ok=True)
    real, grav = ap.pv.simulate(n, seed=ap.SEED_TEST), None
    with ap.world(1.05, 1.0):
        grav = pv.simulate(n, seed=ap.SEED_TEST)
    rend = {"real": pv.render(real["theta"]), "grav": pv.render(grav["theta"])}
    tau_std = float(real["tau"].std())                       # real std of tau
    rows = []
    for si, s in enumerate(sigmas):
        rng = np.random.default_rng(1000 + si)               # identical noise draws to paper_figs.noise
        q = lambda x: pv.to_uint8(np.clip(x + rng.normal(0, s, x.shape), 0, 1)) if s > 0 else pv.to_uint8(x)
        vr, vg = q(rend["real"]), q(rend["grav"])
        sr = {**deriv_scores(vr, real["omega"], real["tau"], tau_std), **{k: v for k, v in em.score_clips(vr, real["omega"], real["tau"]).items()
                                                                           if k in ("r_energy", "r_energy_win_max")}}
        sg = {**deriv_scores(vg, grav["omega"], grav["tau"], tau_std), **{k: v for k, v in em.score_clips(vg, grav["omega"], grav["tau"]).items()
                                                                           if k in ("r_energy", "r_energy_win_max")}}
        r = {"sigma": s, "auroc": {k: em.auroc(sr[k], sg[k]) for k in sr}, "median_real": {k: float(np.median(v)) for k, v in sr.items()},
             "median_grav105": {k: float(np.median(v)) for k, v in sg.items()}}
        rows.append(r)
        print(s, {k: round(v, 3) for k, v in r["auroc"].items()}, flush=True)
    atomic_json(OUT / "derivative_baseline.json", {"git": git_info(), "setup": {"T": pv.T, "DT": pv.DT, "n": n, "seed": ap.SEED_TEST, "c": pv.C, "g": pv.G, "tau_std_real": tau_std,
                                                    "savgol": {"window": 7, "order": 3}}, "rows": rows})
    f = lambda x: f"{x:.3g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"
    names = [("deriv_raw", "derivative raw (θ̈ 2nd diff)"), ("deriv_smooth", "derivative smoothed (SG 7/3)"), ("deriv_gyro", "derivative gyro (ω̇ 1st diff)"),
             ("r_energy", "r_E"), ("r_energy_win_max", "r_E^win")]
    md = ["# Derivative baseline vs energy audit (T=32, n=512; real vs gravity ×1.05, same pixel noise σ on both sides)", "",
          "Per-clip score = median over frames of |θ̈ + c·ω + g·sin(θ̂) − τ| / std(real τ) with nominal c, g (θ̂ from pixels; τ and ω from the sensors). raw: θ̈ by second difference of θ̂; smoothed: Savitzky–Golay (window 7, order 3) second derivative of θ̂; gyro: ω̇ from the central first difference of the gyro (extra variant). "
          "r_E and r_E^win are the energy audit's scores. Same clips, seeds and noise draws as noise_table. Cells: AUROC (real σ = negatives, gravity ×1.05 σ = positives; 0.5 = chance) and, in brackets, median score real → gravity.", "",
          "| σ | " + " | ".join(n for _, n in names) + " |", "|" + "---|" * (len(names) + 1)]
    for r in rows:
        md.append(f"| {r['sigma']} | " + " | ".join(f"{f(r['auroc'][k])} [{f(r['median_real'][k])} → {f(r['median_grav105'][k])}]" for k, _ in names) + " |")
    new_path(OUT / "derivative_baseline.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def sensor_noise(n=512, sig_pix=0.02, fracs=(0.0, 0.01, 0.02, 0.05, 0.10)):
    """(1) pixel noise fixed at sig_pix (the SAME noisy videos as noise_table's sigma=0.02 row), Gaussian noise on gyro AND torque at frac x the real std of each
    sensor; one noise draw per level, applied identically to the real and the gravity x1.05 sides. (2) at sigma_pix = sig_pix and no sensor noise: AUROC of the
    scores for lag 1, time_reversed and video_reversed_only (positives) against the same noisy real clips (negatives)."""
    OUT.mkdir(parents=True, exist_ok=True)
    real = pv.simulate(n, seed=ap.SEED_TEST)
    with ap.world(1.05, 1.0):
        grav = pv.simulate(n, seed=ap.SEED_TEST)
    rend = {"real": pv.render(real["theta"]), "grav": pv.render(grav["theta"])}
    tau_std, om_std = float(real["tau"].std()), float(real["omega"].std())
    si = (0.0, 0.02, 0.05, 0.1).index(sig_pix)
    rng = np.random.default_rng(1000 + si)                  # identical draws to paper_figs.noise / deriv (real first, then gravity)
    q = lambda x, r: pv.to_uint8(np.clip(x + r.normal(0, sig_pix, x.shape), 0, 1))
    vr, vg = q(rend["real"], rng), q(rend["grav"], rng)
    scores = lambda v, om, ta: {**deriv_scores(v, om, ta, tau_std), **{k: x for k, x in em.score_clips(v, om, ta).items() if k in ("r_energy", "r_energy_win_max")}}
    names = [("deriv_gyro", "derivative gyro"), ("deriv_raw", "derivative raw"), ("r_energy", "r_E"), ("r_energy_win_max", "r_E^win")]
    rows = []
    for li, fr in enumerate(fracs):
        r2 = np.random.default_rng(5000 + li)
        ng, nt = r2.normal(0, 1, real["omega"].shape), r2.normal(0, 1, real["tau"].shape)      # one draw per level, same on both sides
        add = lambda d: (d["omega"] + fr * om_std * ng, d["tau"] + fr * tau_std * nt)
        sr, sg = scores(vr, *add(real)), scores(vg, *add(grav))
        rows.append({"sensor_noise_frac": fr, "auroc": {k: em.auroc(sr[k], sg[k]) for k, _ in names}, "median_real": {k: float(np.median(sr[k])) for k, _ in names},
                     "median_grav105": {k: float(np.median(sg[k])) for k, _ in names}})
        print(fr, {k: round(v, 3) for k, v in rows[-1]["auroc"].items()}, flush=True)
    # (2) failure types, no sensor noise
    base = scores(vr, real["omega"], real["tau"])
    clean = rend["real"]
    conds = {"lag=1": (ap.lag_video(clean, 1), real["omega"], real["tau"]),
             "time_reversed": (clean[:, ::-1], -real["omega"][:, ::-1], real["tau"][:, ::-1]),
             "video_reversed_only": (clean[:, ::-1], real["omega"], real["tau"])}
    fail = {}
    for ci, (k, (v, om, ta)) in enumerate(conds.items()):
        r3 = np.random.default_rng(2000 + ci)
        sc = scores(q(np.ascontiguousarray(v), r3), np.ascontiguousarray(om), np.ascontiguousarray(ta))
        fail[k] = {"auroc": {m: em.auroc(base[m], sc[m]) for m, _ in names}, "median_cond": {m: float(np.median(sc[m])) for m, _ in names}}
        print(k, {m: round(x, 3) for m, x in fail[k]["auroc"].items()}, flush=True)
    atomic_json(OUT / "sensor_noise_baseline.json", {"git": git_info(), "setup": {"T": pv.T, "DT": pv.DT, "n": n, "seed": ap.SEED_TEST, "sigma_pix": sig_pix, "gyro_std_real": om_std,
                "tau_std_real": tau_std, "sensor_noise": "Gaussian, frac x real std of each sensor, one draw per level, added identically to real and gravity x1.05 sides"},
                "sensor_noise_rows": rows, "failure_types_sigma_pix_0.02_no_sensor_noise": {"note": "negatives = noisy real clips (same as the gravity test's real side)", **fail}})
    f = lambda x: f"{x:.3g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"
    md = [f"# Sensor-noise robustness of the derivative-gyro baseline vs the energy audit (T=32, n={n}; pixel noise σ={sig_pix})", "",
          f"Real vs gravity ×1.05. Pixel noise σ={sig_pix} on both sides (the same noisy videos as the noise table's σ=0.02 row). Gyro and torque Gaussian noise at the stated fraction of each sensor's real std "
          f"(gyro std {f(om_std)}, torque std {f(tau_std)}); one draw per level, added identically to both sides. Cells: AUROC (real = negatives, gravity ×1.05 = positives) [median real → median gravity]. "
          "Scores as in the derivative baseline: derivative gyro = median |ω̇ + c·ω + g·sin θ̂ − τ|/std τ with ω̇ from the gyro's central difference; derivative raw = same with θ̈ from the second difference of θ̂.", "",
          "| sensor noise | " + " | ".join(n_ for _, n_ in names) + " |", "|" + "---|" * (len(names) + 1)]
    for r in rows:
        md.append(f"| {int(round(r['sensor_noise_frac'] * 100))}% | " + " | ".join(f"{f(r['auroc'][k])} [{f(r['median_real'][k])} → {f(r['median_grav105'][k])}]" for k, _ in names) + " |")
    md += ["", f"## Failure types covered (σ_pix={sig_pix}, no sensor noise; positives = the condition, negatives = the noisy real clips)", "",
           "| condition | " + " | ".join(n_ for _, n_ in names) + " |", "|" + "---|" * (len(names) + 1)]
    for k, v in fail.items():
        md.append(f"| {k} | " + " | ".join(f"{f(v['auroc'][m])} [{f(v['median_cond'][m])}]" for m, _ in names) + " |")
    md.append(f"| (real, median score) | " + " | ".join(f"[{f(float(np.median(base[m])))}]" for m, _ in names) + " |")
    new_path(OUT / "sensor_noise_baseline.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["valid", "deriv"]):
        {"valid": valid, "deriv": deriv, "sensor": sensor_noise}[w]()
