"""
JOB 1: exact residual formulas on the R3 clips (T=32). CPU only.   python -m cvpr27.paper_residual_check
For single-parameter changes of the simulator the audit residual has a closed form (M=L=1): with true dynamics (g, c) and the audit using nominal (g0, c0),
  dE_nom/dt = w*tau - c*w^2 - (g-g0)*sin(th)*w   =>   |dE - (work - c0*int w^2)| = |(c0-c)*int w^2 + (g-g0)*(cos th_b - cos th_a)|.
Torque-sensor gain k (reported tau' = k*tau): |(1-k) * int tau*w|.
Predicted (normalised by the same per-clip mean|E| scale as r_E, from the TRUE theta) vs measured (the existing pixel-based r_E / r_E^win_max on the
stored clips). Window version: the same formulas on every 9-point window (stride 1), max over windows. Writes residual_formula_check.{md,json}.
"floor-equivalent": a clip's predicted residual is below the real floor if predicted/scale < the median (or p95) per-clip r_E of the REAL reference clips
(measured, same pixel extractor); for gravity this is the threshold |cos th_b - cos th_a| < floor * scale / |g-g0|.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.integrate import simpson
from scipy.stats import pearsonr, spearmanr

from cvpr27 import audit_paper as ap
from cvpr27 import evalmetrics as em
from cvpr27 import physvid as pv
from cvpr27.common import atomic_json, drive_root, git_info, new_path

OUT = drive_root() / "paper_figs"


def predicted(kind, val, th, om, tau):
    """Return (global, windows (n,nw)) unnormalised predicted residuals from TRUE arrays."""
    ws = pv.windows() + [(0, pv.T)]
    out = []
    for a, b in ws:
        if kind == "gravity":
            r = (val - 1.0) * pv.G * (np.cos(th[:, b - 1]) - np.cos(th[:, a]))
        elif kind == "damping":
            r = (1.0 - val) * pv.C * simpson(om[:, a:b] ** 2, dx=pv.DT, axis=1)
        else:
            r = (1.0 - val) * simpson(tau[:, a:b] * om[:, a:b], dx=pv.DT, axis=1)
        out.append(np.abs(r))
    out = np.stack(out, 1)
    return out[:, -1], out[:, :-1]


def corr(x, y):
    lx, ly = np.log10(np.maximum(x, 1e-12)), np.log10(np.maximum(y, 1e-12))
    return {"spearman": float(spearmanr(x, y)[0]), "pearson": float(pearsonr(x, y)[0]), "pearson_log10": float(pearsonr(lx, ly)[0])}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    root = drive_root() / "audit_cache"
    n = 512
    real = em.score_clips(*(lambda d: (d["video"], d["omega"], d["tau"]))(ap.cond_data("real_vs_real", n, root)))
    fl = {"r_E": {"median": float(np.median(real["r_energy"])), "p95": float(np.percentile(real["r_energy"], 95))},
          "r_E^win_max": {"median": float(np.median(real["r_energy_win_max"])), "p95": float(np.percentile(real["r_energy_win_max"], 95))}}
    res = {"git": git_info(), "n": n, "real_floor_measured": fl, "formula": "see module docstring", "conditions": {}}
    sweeps = [("gravity", [0.9, 0.95, 1.05, 1.1, 1.25]), ("damping", [0.0, 0.5, 2.0, 4.0]), ("torque_gain", [0.9, 1.1, 1.25])]
    for kind, vals in sweeps:
        for v in vals:
            name = f"{kind}={v}"
            d = ap.cond_data(name, n, root)
            kw = {"gravity": dict(g=v), "damping": dict(c=v), "torque_gain": {}}[kind]
            with ap.world(kw.get("g", 1.0), kw.get("c", 1.0)):
                sim = pv.simulate(n, seed=ap.SEED_TEST)                    # TRUE arrays of the same clips
            th, om, tau = sim["theta"], sim["omega"], sim["tau"]
            assert np.allclose(d["omega"], om) and np.allclose(d["tau"], tau * (v if kind == "torque_gain" else 1.0))
            meas = em.score_clips(d["video"], d["omega"], d["tau"])
            scale = pv._scale(th, om)
            pg, pw = predicted("torque_gain" if kind == "torque_gain" else kind, v, th, om, tau)
            Pg, Pw = pg / scale, pw.max(1) / scale
            Mg, Mw = meas["r_energy"], meas["r_energy_win_max"]
            r = {"kind": kind, "value": v, "global": {**corr(Pg, Mg), "median_predicted": float(np.median(Pg)), "median_measured": float(np.median(Mg)),
                                                       "median_ratio_measured_over_predicted": float(np.median(Mg / np.maximum(Pg, 1e-12)))},
                 "windowed": {**corr(Pw, Mw), "median_predicted": float(np.median(Pw)), "median_measured": float(np.median(Mw)),
                              "median_ratio_measured_over_predicted": float(np.median(Mw / np.maximum(Pw, 1e-12)))}}
            for lab, P, f in (("global", Pg, fl["r_E"]), ("windowed", Pw, fl["r_E^win_max"])):
                r[lab]["frac_predicted_below_real_floor_median"] = float((P < f["median"]).mean())
                r[lab]["frac_predicted_below_real_floor_p95"] = float((P < f["p95"]).mean())
            if kind == "gravity":                                              # the clips the GLOBAL check should miss: |dcos| below the floor-equivalent
                dcos = np.abs(np.cos(th[:, -1]) - np.cos(th[:, 0]))
                thr = fl["r_E"]["median"] * scale / (abs(v - 1.0) * pv.G)
                r["global"]["frac_dcos_below_floor_equivalent"] = float((dcos < thr).mean())
                r["global"]["median_abs_dcos"] = float(np.median(dcos))
                thr95 = fl["r_E"]["p95"] * scale / (abs(v - 1.0) * pv.G)
                r["global"]["frac_dcos_below_floor_equivalent_p95"] = float((dcos < thr95).mean())
                r["global"]["median_floor_equivalent_dcos"] = float(np.median(thr))
            res["conditions"][name] = r
            print(name, {k: round(r["global"][k], 3) for k in ("spearman", "pearson", "median_predicted", "median_measured")}, flush=True)
    atomic_json(OUT / "residual_formula_check.json", res)
    f = lambda x: f"{x:.3g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"
    md = ["# Exact residual formulas vs measured pixel-based r_E (T=32, n=512 clips per condition)", "",
          "Predicted = closed-form residual from the TRUE simulator θ, ω, τ (gravity: |(g−g₀)(cos θ_b − cos θ_a)|; damping: |(c₀−c)∫ω²|; torque gain: |(1−k)∫τω|), normalised by the same per-clip mean|E| as r_E. "
          "Measured = the existing pixel-based r_E (global) / r_E^win (worst 9-point window) on the stored clips. Correlations are per clip (n=512): Spearman, Pearson, and Pearson on log10. "
          f"Real floor (measured on real clips): r_E median {f(fl['r_E']['median'])}, p95 {f(fl['r_E']['p95'])}; r_E^win median {f(fl['r_E^win_max']['median'])}, p95 {f(fl['r_E^win_max']['p95'])}.", ""]
    for lab, title in (("global", "Global (whole clip)"), ("windowed", "Windowed (max over 9-point windows)")):
        md += [f"## {title}", "", "| condition | Spearman | Pearson | Pearson log10 | median predicted | median measured | measured/predicted | frac predicted < real floor (median / p95) |" + (" frac |Δcos| < floor-equivalent (median / p95); median |Δcos| |" if lab == "global" else ""),
               "|" + "---|" * (8 + (lab == "global"))]
        for k, r in res["conditions"].items():
            x = r[lab]
            row = f"| {k} | {f(x['spearman'])} | {f(x['pearson'])} | {f(x['pearson_log10'])} | {f(x['median_predicted'])} | {f(x['median_measured'])} | {f(x['median_ratio_measured_over_predicted'])} | {f(x['frac_predicted_below_real_floor_median'])} / {f(x['frac_predicted_below_real_floor_p95'])} |"
            if lab == "global":
                row += (f" {f(x['frac_dcos_below_floor_equivalent'])} / {f(x['frac_dcos_below_floor_equivalent_p95'])}; {f(x['median_abs_dcos'])} |" if "frac_dcos_below_floor_equivalent" in x else " — |")
            md.append(row)
        md.append("")
    new_path(OUT / "residual_formula_check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
