"""Seed-robustness report for seeds 1 and 2 (seed 0 values are the paper's).

    python seeds_report.py --out /path/to/outdir   # reads <out>/seed1, <out>/seed2
"""
import argparse
import json
from pathlib import Path

import numpy as np

from metrics import evaluate

S0 = {
    "step": list(range(2500, 30001, 2500)),
    "w1_mean": [0.3519, 0.0518, 0.0635, 0.0768, 0.0712, 0.0725, 0.0765, 0.0680, 0.0882, 0.0850, 0.0846, 0.0833],
    "r_energy": [0.7180, 0.3025, 0.1826, 0.1209, 0.0855, 0.0554, 0.0455, 0.0356, 0.0229, 0.0172, 0.0129, 0.0098],
}
S0_FLOOR = {"floor_r_energy": 1.22e-5, "floor_r_dyn": 7.24e-8, "floor_r_kin": 9.17e-6}


def chmean(m, p):
    return float(np.mean([m[f"{p}_{c}"] for c in "xvaF"]))


def boot(real, fake, key, B=200):
    rng = np.random.default_rng(0)
    n = len(fake)
    vals = [evaluate(real, fake[rng.integers(0, n, n)])[key] for _ in range(B)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def seed_report(d):
    log = json.load(open(d / "log.json"))
    real = np.load(d / "real.npy")
    sam = {k: np.load(d / f"samples_{k}.npy") for k in ("independent", "joint", "joint_proj")}

    jl = log["joint"]
    m512 = evaluate(real[:512], sam["joint"][:512])
    ck = jl[:-1]  # intermediate checkpoints, n=512; jl[-1] is 30k at n=1024
    curve = {"step": [e["epoch"] for e in ck] + [jl[-1]["epoch"]],
             "w1_mean": [e["w1_mean"] for e in ck] + [m512["w1_mean"]],
             "r_energy": [e["r_energy"] for e in ck] + [m512["r_energy"]],
             "r_dyn": [e["r_dyn"] for e in ck] + [m512["r_dyn"]]}

    i5 = curve["step"].index(5000)
    w = np.array(curve["w1_mean"][i5:])
    h = len(w) // 2
    improving = bool(w[h:].mean() < 0.9 * w[:h].mean() and w[-1] < w[0])
    dec = {"r_energy_5k": curve["r_energy"][i5], "r_energy_30k_n512": curve["r_energy"][-1],
           "ratio": curve["r_energy"][i5] / curve["r_energy"][-1],
           "w1_min_5k_30k": float(w.min()), "w1_max_5k_30k": float(w.max()),
           "w1_sustained_improvement": improving,
           "w1_reason": f"w1 {w[0]:.4f} at 5k vs {w[-1]:.4f} at 30k; mean of first half {w[:h].mean():.4f} "
                        f"vs second half {w[h:].mean():.4f} (improvement = second half >10% lower and end below start)"}

    final = {}
    for k in sam:
        m = log[k][-1]
        final[k] = {"w1_mean": m["w1_mean"], "psd_mean": chmean(m, "psd"), "acf_mean": chmean(m, "acf"),
                    "r_energy": m["r_energy"], "r_dyn": m["r_dyn"], "r_kin": m["r_kin"]}

    ci = {k: {key: boot(real, sam[k], key) for key in ("w1_mean", "r_energy")}
          for k in ("independent", "joint")}
    ri, rj = final["independent"]["r_energy"], final["joint"]["r_energy"]
    checks = {"w1_indep_hi_reaches_joint_lo": ci["independent"]["w1_mean"][1] >= ci["joint"]["w1_mean"][0],
              "indep_lo_over_joint_hi_r_energy": ci["independent"]["r_energy"][0] / ci["joint"]["r_energy"][1]}
    fl = jl[-1]
    return {"curve": curve, "decoupling": dec, "final_n1024": final, "bootstrap_ci": ci, "ci_checks": checks,
            "ratio_indep_joint_r_energy": ri / rj,
            "partial_projection": {"joint_proj_r_energy": final["joint_proj"]["r_energy"], "joint_r_energy": rj,
                                   "proj_higher_than_joint": final["joint_proj"]["r_energy"] > rj},
            "floors": {k: fl[k] for k in ("floor_r_energy", "floor_r_dyn", "floor_r_kin")}}


def table(rep):
    f = lambda v: v if isinstance(v, str) else f"{v:.4g}"
    ci = lambda r, k, key: "[%.3g, %.3g]" % tuple(r["bootstrap_ci"][k][key])
    P = lambda r, k, key: r["final_n1024"][k][key]
    rows = [
        ("joint W1, 30k (n=512)", S0["w1_mean"][-1], lambda r: r["curve"]["w1_mean"][-1]),
        ("joint r_energy, 5k (n=512)", S0["r_energy"][1], lambda r: r["decoupling"]["r_energy_5k"]),
        ("joint r_energy, 30k (n=512)", S0["r_energy"][-1], lambda r: r["decoupling"]["r_energy_30k_n512"]),
        ("decoupling r_energy(5k)/(30k)", S0["r_energy"][1] / S0["r_energy"][-1], lambda r: r["decoupling"]["ratio"]),
        ("W1 min, 5k-30k", min(S0["w1_mean"][1:]), lambda r: r["decoupling"]["w1_min_5k_30k"]),
        ("W1 max, 5k-30k", max(S0["w1_mean"][1:]), lambda r: r["decoupling"]["w1_max_5k_30k"]),
        ("W1 sustained improvement", "no", lambda r: "yes" if r["decoupling"]["w1_sustained_improvement"] else "no"),
        ("independent W1 (n=1024)", 0.0449, lambda r: P(r, "independent", "w1_mean")),
        ("independent W1 CI", "[0.036, 0.054]", lambda r: ci(r, "independent", "w1_mean")),
        ("joint W1 (n=1024)", 0.0662, lambda r: P(r, "joint", "w1_mean")),
        ("joint W1 CI", "[0.053, 0.083]", lambda r: ci(r, "joint", "w1_mean")),
        ("W1 CIs overlap (indep hi >= joint lo)", "yes", lambda r: "yes" if r["ci_checks"]["w1_indep_hi_reaches_joint_lo"] else "no"),
        ("independent r_energy", 2.95, lambda r: P(r, "independent", "r_energy")),
        ("independent r_energy CI", "[2.62, 3.28]", lambda r: ci(r, "independent", "r_energy")),
        ("joint r_energy", 0.0101, lambda r: P(r, "joint", "r_energy")),
        ("joint r_energy CI", "[0.0091, 0.0111]", lambda r: ci(r, "joint", "r_energy")),
        ("indep r_energy / joint r_energy", 292.0, lambda r: r["ratio_indep_joint_r_energy"]),
        ("indep_lo / joint_hi (r_energy)", 2.62 / 0.0111, lambda r: r["ci_checks"]["indep_lo_over_joint_hi_r_energy"]),
        ("joint_proj r_energy", 0.0111, lambda r: P(r, "joint_proj", "r_energy")),
        ("joint_proj r_energy > joint", "yes", lambda r: "yes" if r["partial_projection"]["proj_higher_than_joint"] else "no"),
        ("independent psd / acf (mean)", "n/a", lambda r: "%.3g / %.3g" % (P(r, "independent", "psd_mean"), P(r, "independent", "acf_mean"))),
        ("joint psd / acf (mean)", "n/a", lambda r: "%.3g / %.3g" % (P(r, "joint", "psd_mean"), P(r, "joint", "acf_mean"))),
        ("joint_proj psd / acf (mean)", "n/a", lambda r: "%.3g / %.3g" % (P(r, "joint_proj", "psd_mean"), P(r, "joint_proj", "acf_mean"))),
        ("independent r_dyn / r_kin", "n/a", lambda r: "%.3g / %.3g" % (P(r, "independent", "r_dyn"), P(r, "independent", "r_kin"))),
        ("joint r_dyn / r_kin", "n/a", lambda r: "%.3g / %.3g" % (P(r, "joint", "r_dyn"), P(r, "joint", "r_kin"))),
        ("joint_proj r_dyn / r_kin", "n/a", lambda r: "%.3g / %.3g" % (P(r, "joint_proj", "r_dyn"), P(r, "joint_proj", "r_kin"))),
        ("floor r_energy", S0_FLOOR["floor_r_energy"], lambda r: r["floors"]["floor_r_energy"]),
        ("floor r_dyn", S0_FLOOR["floor_r_dyn"], lambda r: r["floors"]["floor_r_dyn"]),
        ("floor r_kin", S0_FLOOR["floor_r_kin"], lambda r: r["floors"]["floor_r_kin"]),
    ]
    out = ["| metric | seed 0 (paper) | seed 1 | seed 2 |", "|---|---|---|---|"]
    for name, ref, fn in rows:
        out.append(f"| {name} | {f(ref)} | {f(fn(rep['seed1']))} | {f(fn(rep['seed2']))} |")
    for s in ("seed1", "seed2"):
        out.append(f"\n{s}: {rep[s]['decoupling']['w1_reason']}")
    return "\n".join(out)


def figure(rep, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = [("seed 0 (paper)", S0, "o", "-", "#08519c", "#a50f15"),
         ("seed 1", rep["seed1"]["curve"], "s", "--", "#3182bd", "#de2d26"),
         ("seed 2", rep["seed2"]["curve"], "^", ":", "#6baed6", "#fb6a4a")]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for name, c, mk, ls, cb, cr in S:
        ax[0].plot(c["step"], c["w1_mean"], color=cb, marker=mk, ls=ls, ms=3, lw=1, label=name)
        ax[1].plot(c["step"], c["r_energy"], color=cr, marker=mk, ls=ls, ms=3, lw=1, label=name)
    ax[1].axhline(rep["seed1"]["floors"]["floor_r_energy"], color="grey", ls=(0, (1, 2)), lw=1, label="real-data floor")
    ax[1].set_yscale("log")
    ax[0].set_ylabel("mean W1 (n=512)")
    ax[1].set_ylabel(r"$r_{\mathrm{energy}}$ (n=512)")
    for a in ax:
        a.set_xlabel("training step")
        a.legend(fontsize=6, frameon=False)
        a.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_training_3seeds.pdf")
    fig.savefig(out / "fig_training_3seeds.png", dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    out = Path(ap.parse_args().out)
    rep = {"seed1": seed_report(out / "seed1"), "seed2": seed_report(out / "seed2"),
           "seed0_paper": {"curve": S0, "floors": S0_FLOOR}}
    (out / "seeds_report.json").write_text(json.dumps(rep, indent=2, default=float))
    figure(rep, out)
    print(table(rep))


if __name__ == "__main__":
    main()
