"""
Two soundness fixes, both on already-saved samples. No GPU, no retraining.

    python stats.py

1. BOOTSTRAP ERROR BARS ON W1. The headline marginal gap (independent 0.044 vs
   joint 0.066) is narrower than the spread we observed between two independent
   1024-sample draws from the same joint model (0.066-0.085). Without an
   uncertainty estimate the gap is indefensible. Model weights were not saved,
   so we cannot redraw; instead we bootstrap the saved sample sets, which
   approximates the same sampling distribution.

2. SAMPLE-SIZE-MATCHED ENDPOINT. Intermediate checkpoints were evaluated at
   n=512 and the final one at n=1024. Empirical W1 is biased upward at smaller
   n, so the training curve and its endpoint were not comparable. Re-evaluating
   the saved 30k samples at n=512 gives a clean endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from metrics import evaluate

OUT = Path("results")
B = 200
RNG = np.random.default_rng(0)


def boot(real, fake, key, b=B):
    """Percentile bootstrap over the generated sample set."""
    n = len(fake)
    vals = []
    for _ in range(b):
        idx = RNG.integers(0, n, n)
        vals.append(evaluate(real, fake[idx])[key])
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    real = np.load(OUT / "real.npy")
    log = json.load(open(OUT / "log.json"))

    print(f"=== bootstrap 95% CI, n=1024, B={B} ===")
    print(f"{'variant':16s} {'mean W1':>18s} {'mean r_energy':>22s}")
    out = {}
    for name in ("independent", "joint"):
        p = OUT / f"samples_{name}.npy"
        if not p.exists():
            continue
        fake = np.load(p)
        w, wlo, whi = boot(real, fake, "w1_mean")
        e, elo, ehi = boot(real, fake, "r_energy")
        out[name] = {"w1": [w, wlo, whi], "r_energy": [e, elo, ehi]}
        print(f"{name:16s} {w:.4f} [{wlo:.4f},{whi:.4f}]   "
              f"{e:.4f} [{elo:.4f},{ehi:.4f}]")

    if len(out) == 2:
        i, j = out["independent"]["w1"], out["joint"]["w1"]
        overlap = i[2] >= j[1]
        print(f"\n  W1 intervals {'OVERLAP' if overlap else 'are DISJOINT'}"
              f"  -> {'soften the marginal gap claim' if overlap else 'gap is defensible'}")
        ie, je = out["independent"]["r_energy"], out["joint"]["r_energy"]
        print(f"  r_energy intervals "
              f"{'OVERLAP' if ie[2] >= je[1] else 'are DISJOINT'}")

    print("\n=== 30k joint re-evaluated at n=512 (matches the curve) ===")
    p = OUT / "samples_joint.npy"
    if p.exists():
        fake = np.load(p)
        sub = evaluate(real[:512], fake[:512])
        print(f"  n=512:  w1 {sub['w1_mean']:.4f}   r_energy {sub['r_energy']:.4f}")
        print(f"  n=1024: w1 {log['joint'][-1]['w1_mean']:.4f}   "
              f"r_energy {log['joint'][-1]['r_energy']:.4f}")
        log["joint_n512"] = [{**sub, "epoch": 30000}]

    log["bootstrap"] = out
    with open(OUT / "log.json", "w") as f:
        json.dump(log, f, indent=2, default=float)
    print(f"\nupdated {OUT/'log.json'}")


if __name__ == "__main__":
    main()
