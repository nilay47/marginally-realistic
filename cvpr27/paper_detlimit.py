"""
Detection limit for gravity errors: measured vs Proposition-1 predicted AUROC (pendulum, T=32, n=512 per condition, clean uint8). CPU only.
    python -m cvpr27.paper_detlimit        -> $PHYSGEN_ROOT/paper_figs/fig_detlimit.{pdf,png}, detlimit.{md,json}
Grid: gravity factor 1 +- {0.05,0.1,0.2,0.3,0.5,1,1.5,2,2.5,3,4,5,7,10}% (contains the requested 0.97,0.98,0.99,1.01,1.02,1.03,1.05,1.07,1.10). Same clips (seed 2, same forcing and
initial states; only g differs), same reference R (seed 1, 512 clips) as the audit.
Measured: AUROC (real reference = negatives) of the pixel-based per-clip r_E, r_E^win (worst 9-point window) and r_K, with a 95% bootstrap CI over test clips (B=200).
Predicted (Proposition 1): per clip, predicted unnormalised residual from the TRUE states, |(g-g0)(cos th_b - cos th_a)| (global: whole clip; windowed: max over 9-point windows),
normalised by the clip's mean|E| scale (true theta); then ADD a floor draw resampled (with replacement) from the per-clip measured residuals of REAL test clips (seed 2),
and compute the AUROC of (predicted + floor) vs the real reference's measured scores; mean over 200 resamples ("additive", as specified). r_K has no gravity term: its prediction is the floor alone (AUROC 0.5).
CAUTION: adding the floor MAGNITUDE inflates the positives (|P + e| <= P + |e|), so the additive prediction is an OPTIMISTIC bound on AUROC. We therefore also report a "random-sign"
variant, |P + s*f| with s = +-1 drawn per clip (the real floor is signed error), which is closer to the measured behaviour.
Detection limit = smallest |dg| (g0 = 9.81) at which AUROC >= 0.9, by linear interpolation between grid points, per sign and the smaller of the two.
"""
from __future__ import annotations

import json

import numpy as np

from cvpr27 import audit_paper as ap
from cvpr27 import evalmetrics as em
from cvpr27 import physvid as pv
from cvpr27.common import atomic_json, drive_root, git_info, new_path
from cvpr27.paper_residual_check import predicted

OUT = drive_root() / "paper_figs"
PCTS = [0.05, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10]
REQUIRED = [0.97, 0.98, 0.99, 1.01, 1.02, 1.03, 1.05, 1.07, 1.10]
KEYS = {"r_E": "r_energy", "r_E^win": "r_energy_win_max", "r_K": "kin_rad_max"}


def limit(xs, ys, thr=0.9):
    """smallest x>0 (sorted ascending) where y crosses thr, linear interpolation; None if never."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    for i in range(len(xs)):
        if ys[i] >= thr:
            if i == 0:
                return float(xs[0])
            return float(xs[i - 1] + (thr - ys[i - 1]) * (xs[i] - xs[i - 1]) / max(ys[i] - ys[i - 1], 1e-12))
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    root = drive_root() / "audit_cache"
    n, rng = 512, np.random.default_rng(2025)
    R = ap.cached(root, "ref_s1_n512", None)
    A = ap.cond_data("real_vs_real", n, root)
    perR = em.score_clips(R["video"], R["omega"], R["tau"])
    perA = em.score_clips(A["video"], A["omega"], A["tau"])
    floor_draw = {"r_E": perA["r_energy"], "r_E^win": perA["r_energy_win_max"], "r_K": perA["kin_rad_max"]}
    facs = sorted({round(1 + s * p / 100, 4) for p in PCTS for s in (-1, 1)} | set(REQUIRED))
    rows = {}
    for f in facs:
        d = ap.cached(root, f"det_g{f}_s{ap.SEED_TEST}_n{n}", lambda f=f: ap.make(ap.SEED_TEST, n, g=f))
        with ap.world(f, 1.0):
            sim = pv.simulate(n, seed=ap.SEED_TEST)
        assert np.allclose(d["omega"], sim["omega"])
        per = em.score_clips(d["video"], d["omega"], d["tau"])
        scale = pv._scale(sim["theta"], sim["omega"])
        pg, pw = predicted("gravity", f, sim["theta"], sim["omega"], sim["tau"])
        P = {"r_E": pg / scale, "r_E^win": pw.max(1) / scale, "r_K": np.zeros(n)}
        r = {"factor": f, "dg_abs": float((f - 1) * pv.G), "measured": {}, "predicted": {}}
        for lab, k in KEYS.items():
            au = em.auroc(perR[k], per[k])
            bs = [em.auroc(perR[k][rng.integers(0, n, n)], per[k][rng.integers(0, n, n)]) for _ in range(200)]
            r["measured"][lab] = {"auroc": au, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "median_score": float(np.median(per[k]))}
            pa = [em.auroc(perR[k], P[lab] + floor_draw[lab][rng.integers(0, n, n)]) for _ in range(200)]
            ps = [em.auroc(perR[k], np.abs(P[lab] + rng.choice([-1.0, 1.0], n) * floor_draw[lab][rng.integers(0, n, n)])) for _ in range(200)]
            r["predicted"][lab] = {"auroc": float(np.mean(pa)), "sd": float(np.std(pa)), "auroc_random_sign": float(np.mean(ps)), "sd_random_sign": float(np.std(ps)),
                                   "median_predicted_residual": float(np.median(P[lab]))}
        rows[f] = r
        print(f, {l: (round(r["measured"][l]["auroc"], 3), round(r["predicted"][l]["auroc"], 3)) for l in KEYS}, flush=True)
    lim = {}
    for lab in ("r_E", "r_E^win"):
        for src in ("measured", "predicted", "predicted_random_sign"):
            res = {}
            for sign, name in ((1, "positive_dg"), (-1, "negative_dg")):
                fs = [f for f in facs if (f - 1) * sign > 0]
                fs.sort(key=lambda f: abs(f - 1))
                x = [abs(f - 1) * 100 for f in fs]
                y = [rows[f]["predicted" if src.startswith("predicted") else src][lab]["auroc_random_sign" if src.endswith("random_sign") else "auroc"] for f in fs]
                res[name] = limit(x, y)
            vals = [v for v in res.values() if v is not None]
            res["smallest_pct"] = min(vals) if vals else None
            res["smallest_abs_dg"] = None if not vals else min(vals) / 100 * pv.G
            lim[f"{lab}_{src}"] = res
    atomic_json(OUT / "detlimit.json", {"git": git_info(), "n": n, "grid_factors": facs, "required": REQUIRED, "rows": {str(k): v for k, v in rows.items()},
                                        "detection_limit_auroc_ge_0.9": lim, "method": __doc__})
    f3 = lambda x: "n/a" if x is None else f"{x:.3g}"
    md = ["# Gravity detection limit: measured vs predicted AUROC (T=32, n=512 per condition, clean uint8)", "",
          "Measured AUROC (95% bootstrap CI) / predicted AUROC (Proposition 1: predicted residual from true states + a floor draw resampled from real test clips): additive (as specified; optimistic, |P+e| <= P+|e|) and, in brackets, with a random sign on the floor draw. * = in the requested list. r_K has no gravity term (prediction = floor only).", "",
          "| g factor | dg (m/s²) | r_E measured | r_E predicted (add. [rand. sign]) | r_E^win measured | r_E^win predicted (add. [rand. sign]) | r_K measured | r_K predicted |", "|---|---|---|---|---|---|---|---|"]
    for f in facs:
        r = rows[f]
        c = lambda lab: (f"{r['measured'][lab]['auroc']:.3f} [{r['measured'][lab]['ci95'][0]:.3f}, {r['measured'][lab]['ci95'][1]:.3f}]", f"{r['predicted'][lab]['auroc']:.3f} [{r['predicted'][lab]['auroc_random_sign']:.3f}]")
        md.append(f"| {f}{'*' if f in REQUIRED else ''} | {r['dg_abs']:+.3f} | " + " | ".join(" | ".join(c(l)) for l in KEYS) + " |")
    md += ["", "## Detection limit: smallest |dg| with AUROC ≥ 0.9 (linear interpolation on the grid; smaller of the two signs)", "",
           "A value equal to the smallest grid point (0.05%) is censored (the limit is at or below it). Percent of g0 with the two signs in brackets.", "",
           "| check | measured: % of g0 | measured: |dg| (m/s²) | predicted (additive): % | predicted (additive): |dg| | predicted (random sign): % | predicted (random sign): |dg| |", "|---|---|---|---|---|---|---|"]
    for lab in ("r_E", "r_E^win"):
        m, p, q = lim[f"{lab}_measured"], lim[f"{lab}_predicted"], lim[f"{lab}_predicted_random_sign"]
        g = lambda x: f"{f3(x['smallest_pct'])} (+: {f3(x['positive_dg'])}, −: {f3(x['negative_dg'])})"
        md.append(f"| {lab} | {g(m)} | {f3(m['smallest_abs_dg'])} | {g(p)} | {f3(p['smallest_abs_dg'])} | {g(q)} | {f3(q['smallest_abs_dg'])} |")
    new_path(OUT / "detlimit.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cvpr27.figures import INK2, style
    style()
    col = {"r_E": "#6250d6", "r_E^win": "#1baf7a", "r_K": "#e87ba4"}
    mk = {"r_E": "o", "r_E^win": "s", "r_K": "v"}
    xs = np.array(facs) * 1.0
    xp = (xs - 1) * 100
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for lab in KEYS:
        y = [rows[f]["measured"][lab]["auroc"] for f in facs]
        lo = [rows[f]["measured"][lab]["ci95"][0] for f in facs]
        hi = [rows[f]["measured"][lab]["ci95"][1] for f in facs]
        ax.plot(xp, y, color=col[lab], marker=mk[lab], label=f"{lab} measured")
        ax.fill_between(xp, lo, hi, color=col[lab], alpha=0.12, lw=0)
        ax.plot(xp, [rows[f]["predicted"][lab]["auroc"] for f in facs], color=col[lab], ls="--", lw=1.2, label=f"{lab} predicted")
        if lab != "r_K":
            ax.plot(xp, [rows[f]["predicted"][lab]["auroc_random_sign"] for f in facs], color=col[lab], ls=":", lw=1.2, label=f"{lab} predicted (random sign)")
    ax.axhline(0.9, color=INK2, ls=":", lw=0.8)
    ax.axhline(0.5, color=INK2, ls=":", lw=0.5)
    ax.text(0, 0.905, "AUROC 0.9", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xticks([-10, -3, -1, -0.3, 0, 0.3, 1, 3, 10], ["−10", "−3", "−1", "−0.3", "0", "0.3", "1", "3", "10"])
    ax.set(xlabel="gravity error (g/g$_0$ − 1), % (symlog)", ylabel="AUROC vs real", ylim=(0.4, 1.02))
    ax.legend(fontsize=5.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    for e in ("pdf", "png"):
        fig.savefig(new_path(OUT / f"fig_detlimit.{e}"), bbox_inches="tight")
    print("wrote fig_detlimit")


if __name__ == "__main__":
    main()
