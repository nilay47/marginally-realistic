"""
Two cheap additions, both run on already-saved samples. No retraining.

    python oracle.py

1. SHUFFLED ORACLE. Take real trajectories and permute each channel
   independently across the dataset. Every per-channel metric -- value
   distribution, power spectrum, autocorrelation -- is preserved EXACTLY,
   because the multiset of channel realisations is unchanged. Only the
   pairing between channels is destroyed. This is a construction proof that
   no collection of marginal metrics can certify joint physical validity.

2. FULL-CONSTRAINT PROJECTION. The projector used in the first draft enforced
   a = dv/dt and ma+cv+kx = F but omitted v = dx/dt. That omission matters:
   with all three linear relations imposed, multiplying the ODE by v and
   integrating yields the energy balance, so the earlier claim that a linear
   projector cannot reach a quadratic identity was not established. This
   recomputes the projection with all three constraints so the claim can be
   checked rather than asserted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from metrics import evaluate
from sim import C, DT, K, M, T_LEN, diff_matrix

OUT = Path("results")
_D = diff_matrix()


def shuffled_oracle(real: np.ndarray, seed: int = 0) -> np.ndarray:
    """Permute each channel independently. Marginals exactly preserved."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(real)
    for c in range(real.shape[1]):
        out[:, c] = real[rng.permutation(real.shape[0]), c]
    return out


def full_projection_matrix() -> np.ndarray:
    """Projector onto {v = Dx, a = Dv, ma + cv + kx = F}."""
    T = T_LEN
    I, Z = np.eye(T), np.zeros((T, T))
    A = np.block([
        [-_D,      I,       Z,       Z],   # v - D x   = 0
        [Z,       -_D,      I,       Z],   # a - D v   = 0
        [K * I,    C * I,   M * I,  -I],   # ma+cv+kx-F = 0
    ])
    return np.eye(4 * T) - A.T @ np.linalg.pinv(A @ A.T) @ A


def main():
    real = np.load(OUT / "real.npy")
    log = json.load(open(OUT / "log.json"))

    print("=== shuffled oracle ===")
    orc = shuffled_oracle(real)
    m = evaluate(real, orc)
    log["oracle_shuffled"] = [{**m, "epoch": 0}]
    print(f"  w1 {m['w1_mean']:.4f} (should be ~0)   "
          f"r_energy {m['r_energy']:.4f}   r_dyn {m['r_dyn']:.4f}")

    print("\n=== full-constraint projection of joint samples ===")
    p = OUT / "samples_joint.npy"
    if p.exists():
        z = np.load(p)
        P = full_projection_matrix()
        zp = (z.reshape(len(z), -1) @ P.T).reshape(-1, 4, T_LEN).astype(np.float32)
        np.save(OUT / "samples_joint_fullproj.npy", zp)
        mf = evaluate(real, zp)
        log["joint_fullproj"] = [{**mf, "epoch": log["joint"][-1]["epoch"]}]
        jm = log["joint"][-1]
        print(f"  joint          w1 {jm['w1_mean']:.4f}  "
              f"r_energy {jm['r_energy']:.5f}")
        print(f"  + partial proj w1 {log['joint_proj'][-1]['w1_mean']:.4f}  "
              f"r_energy {log['joint_proj'][-1]['r_energy']:.5f}")
        print(f"  + FULL proj    w1 {mf['w1_mean']:.4f}  "
              f"r_energy {mf['r_energy']:.5f}")
        print("\n  If FULL proj drops r_energy sharply, the earlier 'quadratic")
        print("  constraints are invisible to linear projection' claim is WRONG")
        print("  and the projection result must be reframed or dropped.")

    with open(OUT / "log.json", "w") as f:
        json.dump(log, f, indent=2, default=float)
    print(f"\nupdated {OUT/'log.json'}")


if __name__ == "__main__":
    main()
