"""python figures.py -> figs/fig1_selection.pdf, fig2_training.pdf, table1.tex"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT, FIG = Path("results"), Path("figs")
FIG.mkdir(exist_ok=True)
log = json.load(open(OUT / "log.json"))

names = ["oracle_shuffled", "independent", "joint"]
labels = ["shuffled\noracle", "independent", "joint"]
floor = log["joint"][-1]["floor_r_energy"]  # from full log, before filtering
w1 = [log[n][-1]["w1_mean"] for n in names]
en = [log[n][-1]["r_energy"] for n in names]
dy = [log[n][-1]["r_dyn"] for n in names]

# --- Figure 1: the headline. Best marginal score, worst physics. ------------
fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.4, 2.7))
xs = np.arange(3)

axl.bar(xs, w1, 0.55, color=["#2ca02c", "#2ca02c", "#1f77b4"])
axl.set_xticks(xs); axl.set_xticklabels(labels, fontsize=8)
axl.set_ylabel("mean $W_1$ (normalised)", fontsize=9)
axl.set_title("Marginal realism\n(lower is better)", fontsize=9)
axl.annotate("exactly 0", (0, w1[0]), textcoords="offset points", xytext=(0, 3),
             ha="center", fontsize=8, color="#2ca02c", weight="bold")

# exact zeros cannot be drawn on a log axis; clamp to the floor and mark them
def clamp(vals):
    return [max(v, floor) for v in vals]

axr.bar(xs - 0.19, clamp(en), 0.36, label=r"$|r_{\rm energy}|$", color="#d62728")
axr.bar(xs + 0.19, clamp(dy), 0.36, label=r"$|r_{\rm dyn}|$", color="#ff9896")
axr.axhline(floor, ls=":", color="grey", lw=1)
axr.text(2.45, floor * 1.5, "data floor", color="grey", fontsize=7, ha="right")
for i, v in enumerate(dy):
    if v < floor:
        axr.annotate("0", (i + 0.19, floor), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=7, color="grey")
axr.set_yscale("log")
axr.set_xticks(xs); axr.set_xticklabels(labels, fontsize=8)
axr.set_ylabel("median relative residual", fontsize=9)
axr.set_title("Physical validity\n(lower is better)", fontsize=9)
axr.legend(fontsize=7)
axr.annotate("worst", (0 - 0.19, en[0]), textcoords="offset points",
             xytext=(0, 3), ha="center", fontsize=8, color="#d62728",
             weight="bold")
fig.tight_layout()
fig.savefig(FIG / "fig1_selection.pdf")
print("wrote", FIG / "fig1_selection.pdf")

# --- Figure 2: training dynamics -------------------------------------------
# drop the final checkpoint: it was evaluated at n=1024 while every earlier
# point used n=512, and empirical W1 is biased upward at smaller n, so the
# two are not comparable. r_energy is unaffected but we keep the axes aligned.
j = [r for r in log["joint"] if r["epoch"] < 30000]
# append the 30k checkpoint re-evaluated at n=512 so the whole curve is at
# one sample size (empirical W1 is biased upward at smaller n)
if "joint_n512" in log:
    j = j + log["joint_n512"]
ep = [r["epoch"] for r in j]
fig, ax = plt.subplots(figsize=(5.0, 3.0))
ax.plot(ep, [r["w1_mean"] for r in j], "o-", color="#1f77b4",
        label="marginal realism (mean $W_1$)")
ax.set_xlabel("training step", fontsize=9)
ax.set_ylabel("mean $W_1$ (normalised)", color="#1f77b4", fontsize=9)
ax.set_ylim(0, 0.40)
ax.tick_params(axis="y", labelcolor="#1f77b4")

ax2 = ax.twinx()
ax2.plot(ep, [r["r_energy"] for r in j], "s-", color="#d62728",
         label="physical validity (energy budget)")
ax2.axhline(floor, ls=":", color="grey", lw=1)
ax2.text(27000, floor * 1.6, "data floor", color="grey", fontsize=7, ha="right")
ax2.set_ylabel(r"median $|r_{\rm energy}|$", color="#d62728", fontsize=9)
ax2.set_yscale("log")
ax2.tick_params(axis="y", labelcolor="#d62728")

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
ax.set_title("Marginal realism saturates by 5k steps;\n"
             "the energy budget keeps improving to 30k", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig2_training.pdf")
print("wrote", FIG / "fig2_training.pdf")

# --- Table 1 ----------------------------------------------------------------
# Floors MUST come from the n=1024 entry, not from j[-1]: j has the n=512
# endpoint appended for the curve, and its floors are computed on 512 real
# samples, which gave three different floor values across the paper.
F1024 = log["joint"][-1]
bs = log.get("bootstrap", {})


def ci(name, key, fallback):
    if name in bs and key in bs[name]:
        m, lo, hi = bs[name][key]
        return f"{m:.4f} [{lo:.3f}, {hi:.3f}]" if key == "w1" \
            else f"{m:.2e} [{lo:.2e}, {hi:.2e}]"
    return f"{fallback:.4f}" if key == "w1" else f"{fallback:.2e}"


rows = []
for n, lab in zip(names, labels):
    m = log[n][-1]
    lab = lab.replace("\\n", " ")
    rows.append(f"{lab} & {ci(n,'w1',m['w1_mean'])} & "
                f"{ci(n,'r_energy',m['r_energy'])} & {m['r_dyn']:.2e} \\\\")
tex = ("\\begin{tabular}{lccc}\n\\toprule\nVariant & mean $W_1$ & "
       "$|r_{\\rm energy}|$ & $|r_{\\rm dyn}|$ \\\\\n"
       "\\midrule\n" + "\n".join(rows) +
       f"\n\\midrule\nreal data (floor) & --- & {F1024['floor_r_energy']:.2e} & "
       f"{F1024['floor_r_dyn']:.2e} \\\\\n"
       "\\bottomrule\n\\end{tabular}\n")
(FIG / "table1.tex").write_text(tex)
print("wrote", FIG / "table1.tex")

print(f"\n  floor r_energy {F1024['floor_r_energy']:.3e}  "
      f"r_dyn {F1024['floor_r_dyn']:.3e}  r_kin {F1024['floor_r_kin']:.3e}")
print(f"  independent {en[1]/en[2]:.0f}x worse on energy than joint")
