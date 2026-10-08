"""
Two additions that need no retraining. Run inside the pendulum directory.

    python extra.py

(1) OTHER MARGINAL METRICS. metrics.py already computes per-channel
    log-power-spectrum and autocorrelation errors and stores them in log.json;
    they were simply never reported. If those also fail to prefer the joint
    model, the claim strengthens from "W1 is insufficient" to "several standard
    marginal metrics are insufficient".

(2) JOINT MMD BASELINE. The Discussion argues a generic joint two-sample test
    would also flag the shuffled oracle, but we never ran one. This computes an
    unbiased RBF-MMD^2 on the flattened 4x128 trajectories with the median
    heuristic bandwidth, against a real-vs-heldout-real floor -- the same
    discipline applied to every other metric in the paper.

    Only the final 30k samples were saved, so this cannot test whether MMD
    tracks the late-training energy improvement. It tests the architecture
    comparison only. Report that limitation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim import CH

OUT = Path("results")
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------- MMD ----

def _sqdist(A, B):
    return (A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T


def mmd2(X, Y, gamma=None):
    """Unbiased RBF MMD^2. X, Y are (n, d)."""
    if gamma is None:
        Z = np.vstack([X, Y])
        d = _sqdist(Z, Z)
        med = np.median(d[np.triu_indices_from(d, 1)])
        gamma = 1.0 / (med + 1e-12)
    Kxx, Kyy, Kxy = (np.exp(-gamma * _sqdist(a, b))
                     for a, b in ((X, X), (Y, Y), (X, Y)))
    n, m = len(X), len(Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return (Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
            - 2 * Kxy.mean()), gamma


def flat(a, n):
    return a[:n].reshape(n, -1).astype(np.float64)


def main():
    log = json.load(open(OUT / "log.json"))
    real = np.load(OUT / "real.npy")

    # ---------------- (1) spectral / autocorrelation marginals -------------
    print("=== per-channel marginal metrics (already in log.json) ===")
    print(f"{'metric':10s} " + " ".join(f"{c:>9s}" for c in CH) + f" {'mean':>9s}")
    for key, lab in (("w1", "W1"), ("psd", "logPSD"), ("acf", "ACF")):
        for name in ("independent", "joint"):
            if name not in log:
                continue
            m = log[name][-1]
            vals = [m.get(f"{key}_{c}") for c in CH]
            if any(v is None for v in vals):
                continue
            print(f"{lab:6s}{name[:4]:>4s} " +
                  " ".join(f"{v:9.4f}" for v in vals) +
                  f" {np.mean(vals):9.4f}")
    print("\n  If independent wins or ties on all three, the claim generalises")
    print("  beyond W1 to standard spectral and temporal marginal metrics.")

    # ---------------- (2) joint MMD ---------------------------------------
    print("\n=== joint RBF-MMD^2 (median heuristic, n=512) ===")
    n = 512
    ref_a, ref_b = flat(real, n), flat(real[n:], min(n, len(real) - n))
    floor, gamma = mmd2(ref_a, ref_b)          # real vs held-out real
    print(f"  {'real vs held-out real (floor)':34s} {floor: .3e}")

    orc = np.empty_like(real)
    for c in range(real.shape[1]):
        orc[:, c] = real[RNG.permutation(len(real)), c]
    rows = [("shuffled oracle", orc)]
    for name in ("independent", "joint"):
        p = OUT / f"samples_{name}.npy"
        if p.exists():
            rows.append((name, np.load(p)))

    res = {}
    for lab, arr in rows:
        v, _ = mmd2(ref_a, flat(arr, n), gamma)
        res[lab] = v
        print(f"  {lab:34s} {v: .3e}   ({v/abs(floor):,.0f}x floor)")

    if "independent" in res and "joint" in res:
        ok = res["independent"] > res["joint"]
        print(f"\n  MMD {'DOES' if ok else 'does NOT'} rank joint above independent.")
        print("  Either way, report it: if it ranks correctly the recommendation")
        print("  becomes 'use a joint criterion, a physics residual being the")
        print("  cheapest interpretable one'; if not, marginal-style blindness")
        print("  extends to generic joint discrepancies too.")

    log["mmd"] = {"floor": float(floor),
                  **{k: float(v) for k, v in res.items()}}
    with open(OUT / "log.json", "w") as f:
        json.dump(log, f, indent=2, default=float)
    print(f"\nupdated {OUT/'log.json'}")


if __name__ == "__main__":
    main()
