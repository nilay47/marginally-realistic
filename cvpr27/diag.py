"""
nb7 backend (CPU, no training):  python -m cvpr27.diag --run joint_s0 --tag diag

Diagnoses an (unfinished) joint run from its saved samples_ckpt_<step>.npz files, for every checkpoint found:
  1. physics on ALL clips vs on VALID-ONLY clips (no invalid frame), next to the real-data floor
  2. within-checkpoint SHUFFLE baseline: generated video i paired with generated sensors j (single-cycle permutations, never
     the identity, 20 of them): matched vs shuffled medians and AUROC(matched vs shuffled, positives = shuffled). If the model
     learned ANY coupling, matched is better than shuffled. The same test on real data is the reference
  3. which energy-balance term fails: dE, work (int tau*omega), dissipation (c int omega^2), residual; gyro/torque std ratio
     generated/real; lag of theta_video vs the integrated gyro (cross-correlation of d(theta_video)/dt with the gyro)
  4. figure: 6 samples, theta_video vs theta0 + cumint(gyro)
Writes NEW files under <root>/eval_<tag>/: diag.json, diag.md, fig_diag_theta.{pdf,png}. Every number is computed here (Rule 4).
Samples are in physical units, so no normalisation stats are needed. Needs only the cached held-out set (nb0).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson

from cvpr27 import evalmetrics as em
from cvpr27 import physvid as pv
from cvpr27.common import atomic_json, drive_root, git_info, new_path
from cvpr27.data import load_split

KEYS = ["r_energy", "r_energy_win_max", "r_energy_win_med", "kin_rad_max", "kin_rad_med"]
MAX_LAG = 8


# ------------------------------------------------------------------ helpers
def load_ckpts(run: str):
    out = []
    for p in sorted(glob.glob(str(drive_root() / run / "samples_ckpt_*.npz"))):
        step = int(re.search(r"samples_ckpt_(\d+)\.npz", p).group(1))
        z = np.load(p)
        out.append({"step": step, "video": z["video"], "omega": z["omega"].astype(np.float64),
                    "tau": z["tau"].astype(np.float64), "file": Path(p).name})
    if not out:
        raise FileNotFoundError(f"no samples_ckpt_*.npz under {drive_root() / run}")
    return out


def med(x):
    return float(np.median(x))


def physics(v, om, tau):
    per = em.score_clips(v, om, tau)
    ok = per["invalid_frac_clip"] == 0
    res = {"n": int(len(om)), "n_valid": int(ok.sum()), "invalid_clip_frac": float(1 - ok.mean()),
           "invalid_frame_frac": float(per["invalid_frac_clip"].mean()), "all": em.summarise(per)}
    res["valid_only"] = em.summarise({k: x[ok] for k, x in per.items()}) if ok.sum() >= 2 else None
    return res, per


def single_cycle_perm(n, rng):
    """Random permutation j with j[i] != i for every i (one n-cycle)."""
    p = rng.permutation(n)
    j = np.empty(n, dtype=int)
    j[p] = np.roll(p, -1)
    return j


def shuffle_test(th, om, tau, n_perm, seed):
    """Matched vs shuffled (video i with sensors j) on theta already extracted from the video."""
    rng = np.random.default_rng(seed)
    matched = pv.diagnostics(th, om, tau)
    sh = {k: [] for k in KEYS}
    au = {k: [] for k in KEYS}
    for _ in range(n_perm):
        j = single_cycle_perm(len(om), rng)
        d = pv.diagnostics(th, om[j], tau[j])
        for k in KEYS:
            sh[k].append(med(d[k]))
            au[k].append(em.auroc(matched[k], d[k]))
    return {"n_perm": n_perm,
            "matched_median": {k: med(matched[k]) for k in KEYS},
            "shuffled_median_mean": {k: float(np.mean(sh[k])) for k in KEYS},
            "shuffled_median_min": {k: float(np.min(sh[k])) for k in KEYS},
            "shuffled_median_max": {k: float(np.max(sh[k])) for k in KEYS},
            "shuffled_over_matched": {k: float(np.mean(sh[k]) / max(med(matched[k]), 1e-12)) for k in KEYS},
            "auroc_matched_vs_shuffled_mean": {k: float(np.mean(au[k])) for k in KEYS},
            "auroc_matched_vs_shuffled_min": {k: float(np.min(au[k])) for k in KEYS}}


def energy_terms(th, om, tau):
    """Same definitions as pv.energy_residual over the full clip (Simpson, dx=DT)."""
    E = pv._energy(th, om)
    dE = E[:, -1] - E[:, 0]
    work = simpson(tau * om, dx=pv.DT, axis=1)
    diss = pv.C * simpson(om ** 2, dx=pv.DT, axis=1)
    resid = np.abs(dE - (work - diss))
    scale = pv._scale(th, om)
    return {"abs_dE": med(np.abs(dE)), "abs_work": med(np.abs(work)), "abs_dissipation": med(np.abs(diss)),
            "abs_residual": med(resid), "r_energy": med(resid / scale),
            "median_scale_mean_abs_E": med(scale),
            "resid_if_work_dropped": med(np.abs(dE + diss)),          # |dE - (0 - diss)|
            "resid_if_dissipation_dropped": med(np.abs(dE - work)),   # |dE - (work - 0)|
            "corr_dE_vs_work_minus_diss": float(np.corrcoef(dE, work - diss)[0, 1]),
            "median_sign_agreement_dE_vs_work_minus_diss": float(np.mean(np.sign(dE) == np.sign(work - diss)))}


def lags(th, om):
    """Lag (frames) of theta_video relative to the gyro: positive = the video lags the gyro. Pearson correlation of
    d(theta_video)/dt with the trapezoid-midpoint gyro over the overlap, best lag in [-MAX_LAG, MAX_LAG] per clip."""
    dth = np.diff(th, axis=1) / pv.DT
    om_mid = 0.5 * (om[:, :-1] + om[:, 1:])
    n, m = dth.shape
    best, corr0, bestc = np.zeros(n, int), np.zeros(n), np.full(n, -np.inf)
    for L in range(-MAX_LAG, MAX_LAG + 1):
        a, b = (dth[:, L:], om_mid[:, :m - L]) if L >= 0 else (dth[:, :m + L], om_mid[:, -L:])
        a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
        c = (a * b).sum(1) / np.maximum(np.sqrt((a * a).sum(1) * (b * b).sum(1)), 1e-12)
        if L == 0:
            corr0 = c
        upd = c > bestc
        best[upd], bestc[upd] = L, c[upd]
    return {"median_best_lag_frames": med(best), "frac_best_lag_0": float(np.mean(best == 0)),
            "lag_hist": {int(L): int((best == L).sum()) for L in range(-MAX_LAG, MAX_LAG + 1)},
            "median_corr_at_lag0": med(corr0), "median_corr_at_best_lag": med(bestc)}


def theta_gap(th, om):
    """Mean |theta_video - (theta0 + cumint gyro)| per clip (radians), theta0 = theta_video[0]."""
    ti = th[:, :1] + cumulative_trapezoid(om, dx=pv.DT, axis=1, initial=0)
    return np.abs(th - ti).mean(1)


def sensor_stats(om, tau, real_om, real_tau):
    return {"gyro_std": float(om.std()), "torque_std": float(tau.std()),
            "gyro_std_over_real": float(om.std() / real_om.std()), "torque_std_over_real": float(tau.std() / real_tau.std()),
            "gyro_mean": float(om.mean()), "torque_mean": float(tau.mean())}


# ------------------------------------------------------------------ figure
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def figure(ck, th, out_dir: Path, run: str, nplot=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "lines.linewidth": 1.2, "legend.frameon": False, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.dpi": 200})
    _, ok = pv.extract_theta(ck["video"])
    good = np.where(ok.all(1))[0]
    pick = list(good[:nplot]) + [i for i in range(len(th)) if i not in set(good[:nplot])][:max(0, nplot - len(good))]
    ti = th[:, :1] + cumulative_trapezoid(ck["omega"], dx=pv.DT, axis=1, initial=0)
    t = np.arange(pv.T) * pv.DT
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.0), sharex=True)
    for ax, i in zip(axs.ravel(), pick):
        ax.plot(t, th[i], color="#2a78d6", label="theta from video pixels")
        ax.plot(t, ti[i], color="#eb6834", ls="--", label="theta0 + cumint(gyro)")
        ax.set_title(f"sample {i}" + ("" if ok[i].all() else " (has invalid frames)"), fontsize=7.5)
    for ax in axs[1]:
        ax.set_xlabel("time (s)")
    for ax in axs[:, 0]:
        ax.set_ylabel("angle (rad)")
    axs[0, 0].legend(fontsize=6.5, loc="best")
    fig.suptitle(f"{run} step {ck['step']}: video angle vs integrated gyro (one jointly generated clip each)", fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(new_path(out_dir / f"fig_diag_theta.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ markdown
def f(x, p=3):
    return "n/a" if x is None else (f"{x:.{p}g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}")


def write_md(out_dir: Path, r: dict):
    real = r["real"]
    md = [f"# Joint-run diagnostics: {r['run']}", "",
          f"Checkpoints: {[c['step'] for c in r['checkpoints']]} (n per checkpoint: {[c['physics']['n'] for c in r['checkpoints']]}). "
          f"Real held-out: n={real['physics']['n']}. Shuffle test: {r['n_perm']} single-cycle permutations (never the identity).", "",
          "## 1. Physics, all clips vs valid-only clips (medians)", "",
          "| step | clips | n valid | invalid clips | r_energy | win_max | win_med | kin_max (rad) | kin_med (rad) |", "|---|---|---|---|---|---|---|---|---|"]
    for c in r["checkpoints"]:
        p = c["physics"]
        for lab, blk in (("all", p["all"]), ("valid-only", p["valid_only"])):
            md.append(f"| {c['step']} | {lab} | {p['n'] if lab == 'all' else p['n_valid']} | {f(p['invalid_clip_frac'])} | " +
                      " | ".join("n/a" if blk is None else f(blk[k]["median"]) for k in KEYS) + " |")
    pr = real["physics"]["all"]
    md.append(f"| real | all | {real['physics']['n']} | {f(real['physics']['invalid_clip_frac'])} | " + " | ".join(f(pr[k]["median"]) for k in KEYS) + " |")
    md += ["", "## 2. Within-checkpoint shuffle baseline: did the model learn ANY video-sensor coupling?", "",
           "Median of each diagnostic with matched pairs vs sensors of another generated clip (mean over permutations); "
           "AUROC with positives = shuffled (0.5 = shuffling is undetectable = no coupling; 1.0 = full coupling).", "",
           "| source | " + " | ".join(f"{k} matched / shuffled / AUROC" for k in ["r_energy", "r_energy_win_med", "kin_rad_med"]) + " |",
           "|---|" + "---|" * 3]
    rows = [(f"step {c['step']}", c["shuffle"]) for c in r["checkpoints"]] + [("real", real["shuffle"])]
    for lab, s in rows:
        md.append(f"| {lab} | " + " | ".join(
            f"{f(s['matched_median'][k])} / {f(s['shuffled_median_mean'][k])} / {f(s['auroc_matched_vs_shuffled_mean'][k], 3)}"
            for k in ["r_energy", "r_energy_win_med", "kin_rad_med"]) + " |")
    md += ["", "All five diagnostics, AUROC (matched vs shuffled):", "", "| source | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
    for lab, s in rows:
        md.append(f"| {lab} | " + " | ".join(f(s["auroc_matched_vs_shuffled_mean"][k], 3) for k in KEYS) + " |")
    md += ["", "## 3. Which energy-balance term fails? (medians over clips; theta from pixels, omega = gyro, tau = torque)", "",
           "| source | abs dE | abs work (int tau w) | abs dissipation (c int w^2) | abs residual | r_energy | resid if work dropped | resid if dissipation dropped | corr(dE, work-diss) | sign agreement |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, e in [(f"step {c['step']}", c["energy_terms"]) for c in r["checkpoints"]] + [("real", real["energy_terms"])]:
        md.append(f"| {lab} | " + " | ".join(f(e[k]) for k in ["abs_dE", "abs_work", "abs_dissipation", "abs_residual", "r_energy", "resid_if_work_dropped",
                                                                  "resid_if_dissipation_dropped", "corr_dE_vs_work_minus_diss",
                                                                  "median_sign_agreement_dE_vs_work_minus_diss"]) + " |")
    md += ["", "## 4. Sensor scale and video-vs-gyro timing", "",
           "| source | gyro std / real | torque std / real | median best lag (frames, + = video lags gyro) | clips with lag 0 | median corr dtheta/dt vs gyro (lag 0) | median corr (best lag) | median mean abs(theta_video - theta0 - cumint gyro) (rad) |",
           "|---|---|---|---|---|---|---|---|"]
    for lab, c in [(f"step {c['step']}", c) for c in r["checkpoints"]] + [("real", real)]:
        s, l = c["sensors"], c["lag"]
        md.append(f"| {lab} | {f(s['gyro_std_over_real'])} | {f(s['torque_std_over_real'])} | {f(l['median_best_lag_frames'])} | "
                  f"{f(l['frac_best_lag_0'])} | {f(l['median_corr_at_lag0'])} | {f(l['median_corr_at_best_lag'])} | {f(c['theta_gap_median_rad'])} |")
    md += ["", f"Figure: `fig_diag_theta.png` (6 samples from step {r['checkpoints'][-1]['step']})."]
    new_path(out_dir / "diag.md").write_text("\n".join(md) + "\n")
    return md


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="joint_s0")
    ap.add_argument("--tag", default="diag")
    ap.add_argument("--n_perm", type=int, default=20)
    a = ap.parse_args(argv)
    out_dir = drive_root() / f"eval_{a.tag}"
    if (out_dir / "diag.json").exists():
        print(f"{out_dir}/diag.json exists: use a new --tag (nothing is overwritten)."); return
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    held = load_split("heldout")
    rv, rom, rtau = held["video"], held["omega"].astype(np.float64), held["tau"].astype(np.float64)
    cks = load_ckpts(a.run)
    print(f"{a.run}: {len(cks)} checkpoints {[c['step'] for c in cks]}; real held-out n={len(rv)}")

    def analyse(v, om, tau, seed, ref=None):
        res = {}
        res["physics"], _ = physics(v, om, tau)
        th, _ = pv.extract_theta(v)
        res["shuffle"] = shuffle_test(th, om, tau, a.n_perm, seed)
        res["energy_terms"] = energy_terms(th, om, tau)
        res["sensors"] = sensor_stats(om, tau, rom, rtau)
        res["lag"] = lags(th, om)
        res["theta_gap_median_rad"] = med(theta_gap(th, om))
        return res, th

    real, _ = analyse(rv, rom, rtau, seed=1)
    print(f"real done ({time.time() - t0:.0f}s)")
    out = []
    th_last = None
    for i, c in enumerate(cks):
        res, th = analyse(c["video"], c["omega"], c["tau"], seed=100 + i)
        res.update(step=c["step"], file=c["file"])
        out.append(res)
        th_last = th
        s, p = res["shuffle"], res["physics"]
        print(f"step {c['step']:>6}: n={p['n']} invalid_clips={p['invalid_clip_frac']:.3f} "
              f"r_energy={p['all']['r_energy']['median']:.3g} | shuffle AUROC r_energy={s['auroc_matched_vs_shuffled_mean']['r_energy']:.3f} "
              f"kin_med={s['auroc_matched_vs_shuffled_mean']['kin_rad_med']:.3f} | gyro_std/real={res['sensors']['gyro_std_over_real']:.3g} "
              f"torque_std/real={res['sensors']['torque_std_over_real']:.3g} lag={res['lag']['median_best_lag_frames']:g}")
    r = {"run": a.run, "n_perm": a.n_perm, "git": git_info(), "real": real, "checkpoints": out,
         "seconds": round(time.time() - t0, 1)}
    atomic_json(out_dir / "diag.json", r)
    md = write_md(out_dir, r)
    figure(cks[-1], th_last, out_dir, a.run)
    print("\n".join(md))
    print(f"wrote {out_dir}/diag.json, diag.md, fig_diag_theta.(pdf|png)  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
