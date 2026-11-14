"""
nb5 backend: seed robustness of the joint model.   python -m cvpr27.seeds --tag seeds

Reads joint_s0 / joint_s1 / joint_s2 (config.json, metrics_*.json, metrics_final_ancestral.json) and writes, as NEW files
under <root>/eval_<tag>/: seeds_report.json, seeds_report.md, fig_seeds_training.{pdf,png}.
Every number comes from those files. Seeds differ only in model init / batch order / noise (the training set is
simulation seed 0 for all of them), so this measures training stochasticity, not data resampling.

Decoupling statistic (defined here, computed per seed): s_W = first checkpoint whose sensor-W1 is within 10% of the
minimum W1 on the curve; decoupling = r_energy(s_W) / r_energy(last checkpoint). Large = physics kept improving
long after the marginal metric stopped.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from cvpr27.common import atomic_json, drive_root, new_path


def load_run(name):
    d = drive_root() / name
    cfg = json.loads((d / "config.json").read_text())
    rows = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(d / "metrics_0*.json")))]
    fin = json.loads((d / "metrics_final_ancestral.json").read_text())
    return cfg, rows, fin


def summarise(name):
    cfg, rows, fin = load_run(name)
    step = [r["step"] for r in rows]
    w1 = np.array([r["metrics"]["w1_mean"] for r in rows])
    re = np.array([r["metrics"]["physical"]["r_energy"]["median"] for r in rows])
    sW = step[int(np.argmax(w1 <= 1.1 * w1.min()))]
    i = step.index(sW)
    p = lambda m, k: m["physical"][k]["median"]
    return {"run": name, "seed": cfg["args"]["seed"], "git": cfg["git"]["commit"], "n_params": cfg["n_params"],
            "curve": {"step": step, "w1_mean": w1.tolist(), "r_energy": re.tolist(),
                      "r_energy_win_max": [p(r["metrics"], "r_energy_win_max") for r in rows],
                      "kin_rad_max": [p(r["metrics"], "kin_rad_max") for r in rows],
                      "ratio_win_over_glob": [r["metrics"]["physical"]["ratio_of_medians_win_over_glob"] for r in rows],
                      "video_metric": [r["metrics"].get("video_metric", {}).get("value") for r in rows]},
            "decoupling": {"s_W": sW, "w1_min": float(w1.min()), "r_energy_at_sW": float(re[i]),
                           "r_energy_last": float(re[-1]), "ratio": float(re[i] / re[-1])},
            "final_ancestral": {"n": fin["n"], "w1_mean": fin["metrics"]["w1_mean"], "psd_mean": fin["metrics"]["psd_mean"],
                                "video_metric": fin["metrics"].get("video_metric", {}).get("value"),
                                "r_energy": p(fin["metrics"], "r_energy"), "r_energy_win_max": p(fin["metrics"], "r_energy_win_max"),
                                "kin_rad_max": p(fin["metrics"], "kin_rad_max"),
                                "invalid_frame_frac": fin["metrics"]["invalid_frame_frac"],
                                "mmd": fin["metrics"]["mmd"]["value"]},
            "floor": cfg["floor_real_heldout"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="seeds")
    ap.add_argument("--runs", nargs="+", default=["joint_s0", "joint_s1", "joint_s2"])
    a = ap.parse_args(argv)
    out = drive_root() / f"eval_{a.tag}"
    out.mkdir(exist_ok=True)
    rep = [summarise(r) for r in a.runs]
    atomic_json(out / "seeds_report.json", {"runs": rep})

    f = lambda x: "n/a" if x is None else (f"{x:.3g}" if abs(x) >= 1e-3 else f"{x:.2e}")
    keys = ["w1_mean", "video_metric", "r_energy", "r_energy_win_max", "kin_rad_max", "mmd", "invalid_frame_frac"]
    md = ["# Joint model across seeds", "", "| run | seed | " + " | ".join(keys) + " | s_W | decoupling r_energy(s_W)/r_energy(last) |",
          "|---|---|" + "---|" * (len(keys) + 2)]
    for r in rep:
        md.append(f"| {r['run']} | {r['seed']} | " + " | ".join(f(r["final_ancestral"][k]) for k in keys)
                  + f" | {r['decoupling']['s_W']} | {f(r['decoupling']['ratio'])} |")
    md += ["", "Final = 250-step ancestral, n=%d. s_W and decoupling are defined in the module docstring." % rep[0]["final_ancestral"]["n"]]
    new_path(out / "seeds_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cvpr27.figures import C, FLOOR, style
    style()
    cols, mk = ["#2a78d6", "#eb6834", "#1baf7a"], "os^"
    fig, ax = plt.subplots(1, 2, figsize=(5.8, 2.5))
    for r, c, m in zip(rep, cols, mk):
        ax[0].plot(r["curve"]["step"], r["curve"]["w1_mean"], color=c, marker=m, label=f"seed {r['seed']}")
        ax[1].plot(r["curve"]["step"], r["curve"]["r_energy"], color=c, marker=m, label=f"seed {r['seed']}")
    ax[1].axhline(rep[0]["floor"]["r_energy"]["median"], color=FLOOR, ls=(0, (1, 2)))
    ax[0].set(xlabel="training step", title="sensor W1")
    ax[1].set(xlabel="training step", yscale="log", title="global energy residual (dotted = real floor)")
    ax[0].legend(fontsize=7)
    for e in ("pdf", "png"):
        fig.savefig(new_path(out / f"fig_seeds_training.{e}"), bbox_inches="tight")


if __name__ == "__main__":
    main()
