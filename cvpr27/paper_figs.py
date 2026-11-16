"""
Paper figures/tables on the audit setup (T=32, DT=0.04, same seeds and test clips as results_fast.json). CPU-light; run alongside the FVD run.

    PHYSGEN_ROOT=~/physgen_local PV_T=32 PV_DT=0.04 PV_WIN=8 python -m cvpr27.paper_figs [teaser] [noise]

teaser: teaser_fig.{pdf,png}   4 rows (real | gravity x1.25 | damping x0 | time-reversed) x 8 frames of ONE test clip (same forcing + initial state
        in every row, |theta|>1 rad), faint pivot-to-bob line, right column = that condition's POPULATION numbers read from results_fast.json.
noise : noise_table.{md,json}  Gaussian pixel noise sigma in {0,.02,.05,.1} added to the float render, re-quantised to uint8 (real test clips);
        r_E, r_E^win (= windowed worst-window residual r_energy_win_max), r_K (= kin_rad_max) median+IQR, invalid-frame fraction, and AUROC of
        real-clean (reference) vs gravity x1.05 rendered with the SAME sigma (and, as a false-alarm check, vs real with the same sigma).
Outputs in $PHYSGEN_ROOT/paper_figs, new files only.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from cvpr27 import audit_paper as ap
from cvpr27 import evalmetrics as em
from cvpr27 import physvid as pv
from cvpr27.common import atomic_json, drive_root, git_info, new_path

OUT = drive_root() / "paper_figs"


def sim(n, seed, g=1.0, c=1.0):
    with ap.world(g, c):
        return pv.simulate(n, seed=seed)


def fmt(x):
    return f"{x:.2g}" if abs(x) >= 1e-2 else f"{x:.1e}".replace("e-0", "e-").replace("e+0", "e")


def teaser(n=512):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cvpr27.figures import INK, INK2, style
    style()
    fast = json.loads((drive_root() / "audit" / "results_fast.json").read_text())
    conds = {**fast["R3"]["conditions"], **fast["R2"]["conditions"]}
    root = drive_root() / "audit_cache"
    rows = [("real", "real_vs_real"), ("gravity ×1.25", "gravity=1.25"), ("damping ×0", "damping=0.0"), ("time-reversed", "time_reversed")]
    th = {"real_vs_real": sim(n, ap.SEED_TEST)["theta"], "gravity=1.25": sim(n, ap.SEED_TEST, g=1.25)["theta"], "damping=0.0": sim(n, ap.SEED_TEST, c=0.0)["theta"]}
    th["time_reversed"] = th["real_vs_real"][:, ::-1]
    vid = {k: ap.cond_data(k, n, root)["video"] for k in th}
    # one clip: large swing in the real row, and the rows' trajectories visibly differ
    div = sum(np.abs(th["real_vs_real"] - th[k]).mean(1) for k in th if k != "real_vs_real")
    ok = np.abs(th["real_vs_real"]).max(1) > 1.0
    i = int(np.argmax(np.where(ok, div, -1)))
    fr = list(range(0, 32, 4))
    W, Hh, lab_w, fw, rh, hdr = 7.0, 2.0, 1.15, 0.46, 0.46, 0.16
    fig = plt.figure(figsize=(W, Hh))
    for r, (label, key) in enumerate(rows):
        y0 = Hh - hdr - (r + 1) * rh
        fig.text(0.02 / W * 1.0, (y0 + rh / 2) / Hh, label, fontsize=8, va="center", ha="left", color=INK)
        for j, f in enumerate(fr):
            ax = fig.add_axes([(lab_w + j * fw + 0.02) / W, (y0 + 0.02) / Hh, (fw - 0.04) / W, (rh - 0.04) / Hh])
            ax.imshow(vid[key][i, f], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            b = (pv.PIVOT[1] + pv.R_PX * np.sin(th[key][i, f]), pv.PIVOT[0] + pv.R_PX * np.cos(th[key][i, f]))
            ax.plot([pv.PIVOT[1], b[0]], [pv.PIVOT[0], b[1]], color="#ff9f43", lw=0.5, alpha=0.7)
            ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5)
            ax.axis("off")
        c = conds[key]
        t = (f"W1 {fmt(c['w1']['w1_gyro'])} · MMD² {fmt(c['mmd']['subsample_median'])}\\n"
             f"r$_E$ {fmt(c['physical']['r_energy']['median'])} · AUROC {c['auroc_vs_real']['r_energy']:.2f}").replace("\\n", "\n")
        fig.text((lab_w + 8 * fw + 0.08) / W, (y0 + rh / 2) / Hh, t, fontsize=8, va="center", ha="left", color=INK)
    for j, f in enumerate(fr):
        fig.text((lab_w + (j + 0.5) * fw) / W, (Hh - hdr / 2) / Hh, f"{f}", fontsize=8, ha="center", va="center", color=INK2)
    fig.text((lab_w + 8 * fw + 0.08) / W, (Hh - hdr / 2) / Hh, f"population audit (n={n})", fontsize=8, ha="left", va="center", color=INK2)
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(new_path(OUT / f"teaser_fig.{e}"), dpi=300)
    plt.close(fig)
    print(f"teaser: clip {i} (max|theta| {np.abs(th['real_vs_real'][i]).max():.2f} rad), wrote {OUT}/teaser_fig.*")


def noise(n=512, sigmas=(0.0, 0.02, 0.05, 0.1)):
    OUT.mkdir(parents=True, exist_ok=True)
    real, grav = sim(n, ap.SEED_TEST), sim(n, ap.SEED_TEST, g=1.05)
    clean = pv.render(real["theta"])
    rend = {"real": clean, "grav105": pv.render(grav["theta"])}
    K = {"r_E": "r_energy", "r_E^win": "r_energy_win_max", "r_K": "kin_rad_max"}
    rows, per0 = [], None
    for si, s in enumerate(sigmas):
        rng = np.random.default_rng(1000 + si)
        q = lambda x: pv.to_uint8(np.clip(x + rng.normal(0, s, x.shape), 0, 1)) if s > 0 else pv.to_uint8(x)
        vr, vg = q(rend["real"]), q(rend["grav105"])
        pr = em.score_clips(vr, real["omega"], real["tau"])
        pg = em.score_clips(vg, grav["omega"], grav["tau"])
        if per0 is None:
            per0 = pr                                                           # real-clean reference (sigma = 0)
        r = {"sigma": s, "n": n, "invalid_frame_frac": float(pr["invalid_frac_clip"].mean()), "invalid_frame_frac_grav105": float(pg["invalid_frac_clip"].mean())}
        for nm, k in K.items():
            r[nm] = {"median": float(np.median(pr[k])), "q25": float(np.percentile(pr[k], 25)), "q75": float(np.percentile(pr[k], 75))}
        r["auroc_clean_vs_grav105_same_sigma"] = {nm: em.auroc(per0[K[nm]], pg[K[nm]]) for nm in ("r_E", "r_E^win")}
        r["auroc_clean_vs_real_same_sigma"] = {nm: em.auroc(per0[K[nm]], pr[K[nm]]) for nm in ("r_E", "r_E^win")}
        r["auroc_same_sigma_real_vs_grav105"] = {nm: em.auroc(pr[K[nm]], pg[K[nm]]) for nm in ("r_E", "r_E^win")}
        rows.append(r)
        print(f"sigma {s}: r_E {r['r_E']['median']:.3g} r_E^win {r['r_E^win']['median']:.3g} r_K {r['r_K']['median']:.3g} invalid {r['invalid_frame_frac']:.3g} "
              f"AUROC grav1.05 {r['auroc_clean_vs_grav105_same_sigma']}", flush=True)
    atomic_json(OUT / "noise_table.json", {"git": git_info(), "setup": {"T": pv.T, "DT": pv.DT, "WIN": pv.WIN, "n": n, "seed": ap.SEED_TEST, "sigmas": list(sigmas)}, "rows": rows,
                                           "note": "r_E^win = r_energy_win_max, r_K = kin_rad_max. Noise added to the float [0,1] render, clipped to [0,1], re-quantised to uint8. Reference for AUROC = the same real clips at sigma 0."})
    f = lambda x: f"{x:.3g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"
    md = ["# Noise robustness of the audit (T=32, n=%d real clips; gravity ×1.05 rendered with the same σ)" % n, "",
          "σ = std of Gaussian noise added to the float [0,1] render, then clipped and re-quantised to uint8. Median [IQR]. AUROC: real-clean vs "
          "gravity ×1.05 at the same σ (detection); last two columns: real-clean vs real at the same σ (false-alarm check, 0.5 = no false alarms); last two columns: noisy real vs noisy gravity at the same σ (the like-for-like comparison, noise cancels).", "",
          "| σ | r_E | r_E^win | r_K (rad) | invalid frames | AUROC r_E (g×1.05) | AUROC r_E^win (g×1.05) | AUROC r_E (real, σ) | AUROC r_E^win (real, σ) | AUROC r_E (real σ vs g×1.05 σ) | AUROC r_E^win (real σ vs g×1.05 σ) |", "|" + "---|" * 11]
    for r in rows:
        c = lambda k: f"{f(r[k]['median'])} [{f(r[k]['q25'])}, {f(r[k]['q75'])}]"
        a, b = r["auroc_clean_vs_grav105_same_sigma"], r["auroc_clean_vs_real_same_sigma"]
        md.append(f"| {r['sigma']} | {c('r_E')} | {c('r_E^win')} | {c('r_K')} | {f(r['invalid_frame_frac'])} | {f(a['r_E'])} | {f(a['r_E^win'])} | {f(b['r_E'])} | {f(b['r_E^win'])} | {f(r['auroc_same_sigma_real_vs_grav105']['r_E'])} | {f(r['auroc_same_sigma_real_vs_grav105']['r_E^win'])} |")
    new_path(OUT / "noise_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    which = sys.argv[1:] or ["teaser", "noise"]
    if "teaser" in which:
        teaser()
    if "noise" in which:
        noise()
