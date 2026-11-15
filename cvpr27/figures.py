"""
Paper figures from results.json + the training runs' metric files:  python -m cvpr27.figures --tag main

Design (dataviz skill): one axis per panel (no dual axes: measures on different scales get their own panel),
fixed categorical order joint=blue, independent=orange, cascaded=aqua, shuffle_oracle=yellow (validated, light
mode), thin marks, recessive grid, legend + direct labels + distinct markers (aqua/yellow are <3:1 on white, so
labels are mandatory; results.md is the table view). Text stays in ink colours, never the series colour.
Writes PDF and PNG; never overwrites.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cvpr27.common import drive_root, new_path

INK, INK2, GRID, FLOOR = "#0b0b0b", "#52514e", "#e6e5e1", "#8a8985"
C = {"joint": "#2a78d6", "independent": "#eb6834", "cascaded": "#1baf7a", "shuffle_oracle": "#eda100",
     "video_only": "#6250d6", "sensor_only": "#e87ba4"}
MK = {"joint": "o", "independent": "s", "cascaded": "^", "shuffle_oracle": "D", "video_only": "v", "sensor_only": "P"}
PHYS = [("r_energy", "global energy residual"), ("r_energy_win_max", "worst 8-frame window"),
        ("kin_rad_max", "video–gyro mismatch (rad)")]


def style():
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "lines.linewidth": 1.4, "lines.markersize": 4, "legend.frameon": False,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200})


def save(fig, d: Path, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(new_path(d / f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", d / name)


def curve(run: str):
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(drive_root() / run / "metrics_0*.json")))]


def fig_training(res, d, jr="joint_s0"):
    rows = curve(jr)
    st = [r["step"] for r in rows]
    fl = res["floors"]["summary"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    vm = [r["metrics"].get("video_metric", {}).get("value") for r in rows]
    vname = (res["video_metric"] or {}).get("name", "video metric")
    if all(v is not None for v in vm):
        ax[0].plot(st, vm, color=C["joint"], marker=MK["joint"])
    ax[0].set(title=f"{vname} (n={rows[0]['n']})", xlabel="training step")
    ax[1].plot(st, [r["metrics"]["w1_mean"] for r in rows], color=C["joint"], marker=MK["joint"])
    ax[1].set(title="sensor W1", xlabel="training step")
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    for (k, lab), c, m in zip(PHYS, cols, "os^"):
        y = [r["metrics"]["physical"][k]["median"] for r in rows]
        ax[2].plot(st, y, color=c, marker=m, label=lab)
        ax[2].axhline(fl[k]["median"], color=c, ls=(0, (1, 2)), lw=0.9)
    ax[2].set(yscale="log", title="physical validity (dotted = real-data floor)", xlabel="training step")
    ax[2].legend(fontsize=6.5, loc="upper right")
    save(fig, d, "fig_e2_training")


def fig_ratio(res, d):
    e3 = res["E3_ratio_curve"]
    st = [b["step"] for b in e3["by_step"]]
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.plot(st, [b["ratio_of_medians"] for b in e3["by_step"]], color=C["joint"], marker="o", label="joint model")
    ax.axhline(e3["real_ratio_of_medians"], color=FLOOR, ls=(0, (1, 2)))
    ax.text(st[-1], e3["real_ratio_of_medians"], "real data", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.set(yscale="log", xlabel="training step", ylabel="worst-window / global residual",
           title="Local violations vs the global balance")
    ax.legend(fontsize=7)
    save(fig, d, "fig_e3_ratio")


def fig_variants(res, d):
    V = ["joint", "independent", "cascaded", "shuffle_oracle"]
    B, R0 = res["bootstrap_ci_main_draw"], res["draws"][0]

    def sc(v, k):
        if k == "video_metric":
            st = B[v].get(k)
            return None if st is None else st["subsample_median"]        # set-level: subsample median, equal m for all
        if k in ("w1_mean", "psd_mean"):
            return R0[v].get(k)
        return R0[v]["physical"][k]["median"] if "physical" in R0[v] else None

    panels = [("w1_mean", "sensor W1 (lower = more realistic)", False),
              ("video_metric", (res["video_metric"] or {}).get("name", "video") + f" (median of m={res['args']['m_sub']} subsets)", False)] + [(k, t, True) for k, t in PHYS]
    panels = [p for p in panels if sc("joint", p[0]) is not None]
    fig, axs = plt.subplots(1, len(panels), figsize=(1.9 * len(panels), 2.7))
    for ax, (k, title, logy) in zip(np.atleast_1d(axs), panels):
        for i, v in enumerate(V):
            y = sc(v, k)
            if y is None:
                continue
            lo, hi = B[v][k]["subsample_ci95"] if k == "video_metric" else B[v][k]
            ax.bar(i, y, 0.55, color=C[v], yerr=[[max(y - lo, 0)], [max(hi - y, 0)]],
                   error_kw={"lw": 0.8, "capsize": 2, "ecolor": INK2})
        if k in res["floors"]["median_and_ci"]:
            ax.axhline(res["floors"]["median_and_ci"][k][0], color=FLOOR, ls=(0, (1, 2)))
        ax.set(title=title, xticks=range(len(V)), xticklabels=["joint", "indep.", "casc.", "shuffle\noracle"])
        ax.tick_params(axis="x", labelsize=6.5)
        if logy:
            ax.set_yscale("log")
    save(fig, d, "fig_e4_variants")


def fig_sample_eff(res, d):
    E = res["E6_sample_efficiency"]
    fig, axs = plt.subplots(1, len(E), figsize=(3.4 * len(E), 2.7), sharey=True)
    cols = {"r_energy": "#2a78d6", "r_energy_win_max": "#eb6834", "kin_rad_max": "#1baf7a", "w1_mean": "#eda100",
            "mmd": "#e87ba4", "video_metric": "#6250d6"}
    mk = {"r_energy": "o", "r_energy_win_max": "s", "kin_rad_max": "^", "w1_mean": "D", "mmd": "P", "video_metric": "v"}
    for ax, (name, e) in zip(np.atleast_1d(axs), E.items()):
        ns = sorted(int(n) for n in e["p_correct_by_n"])
        keys = list(next(iter(e["p_correct_by_n"].values())).keys())
        for k in keys:
            y = [e["p_correct_by_n"][str(n)].get(k) for n in ns]
            ax.plot(ns, y, color=cols[k], marker=mk[k], label=k)
            ax.text(ns[-1], y[-1], " " + k, fontsize=6, va="center", color=INK2)
        ax.axhline(0.5, color=FLOOR, ls=(0, (1, 2)))
        ax.axhline(0.95, color=FLOOR, ls=(0, (3, 2)), lw=0.7)
        ax.set(xscale="log", title=name.replace("_", " "), xlabel="samples n", ylim=(-0.02, 1.02))
    np.atleast_1d(axs)[0].set_ylabel("P(metric ranks the worse model worse)")
    save(fig, d, "fig_e6_sample_efficiency")


def fig_auroc(res, d):
    A = res["E7_auroc"]
    rows, cols = list(A), ["r_energy", "r_energy_win_max", "kin_rad_max", "r_energy_win_med", "kin_rad_med"]
    M = np.array([[A[r][c] for c in cols] for r in rows])
    fig, ax = plt.subplots(figsize=(4.6, 0.38 * len(rows) + 1.0))
    im = ax.imshow(M, vmin=0.5, vmax=1.0, cmap="Blues", aspect="auto")
    ax.set(xticks=range(len(cols)), xticklabels=cols, yticks=range(len(rows)), yticklabels=rows)
    ax.tick_params(axis="x", rotation=30, labelsize=6.5)
    ax.grid(False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color=INK if M[i, j] < 0.8 else "white")
    ax.set_title("Per-sample detection AUROC (real held-out = negatives)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    save(fig, d, "fig_e7_auroc")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--joint_run", default="joint_s0")
    ap.add_argument("--suffix", default=None, help="v2 -> joint run joint_v2 and (default) tag v2")
    a = ap.parse_args(argv)
    if a.suffix:
        a.joint_run = f"joint_{a.suffix}"
        if a.tag == "main":
            a.tag = a.suffix
    out = drive_root() / f"eval_{a.tag}"
    res = json.loads((out / "results.json").read_text())
    d = out / "figures"
    d.mkdir(exist_ok=True)
    style()
    fig_training(res, d, a.joint_run)
    fig_ratio(res, d)
    fig_variants(res, d)
    fig_sample_eff(res, d)
    fig_auroc(res, d)


if __name__ == "__main__":
    main()
