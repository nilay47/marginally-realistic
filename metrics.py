"""
Two families of metric, deliberately kept apart. The paper's claim is that
they DECOUPLE: family A converges while family B plateaus.

A. MARGINAL REALISM  -- each channel judged on its own
     W1 on the value distribution, PSD error, autocorrelation error
B. PHYSICAL VALIDITY -- the joint realisation judged against the governing laws
     median |r_kin| (parameter-free), median |r_dyn|

Residuals are reported RELATIVE to the natural scale of the channel they
live in, so they are dimensionless and comparable across variants.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from sim import CH, energy_residual, residuals


def _psd(traj, ch):
    return np.abs(np.fft.rfft(traj[:, ch], axis=-1)) ** 2


def _acf(traj, ch, lags=24):
    s = traj[:, ch]
    s = s - s.mean(-1, keepdims=True)
    var = (s ** 2).mean(-1) + 1e-12
    return np.stack([(s[:, :s.shape[1] - l] * s[:, l:]).mean(-1) / var
                     for l in range(lags)], 1)


def marginal(real: np.ndarray, fake: np.ndarray) -> dict:
    out = {}
    w1s = []
    for c in range(real.shape[1]):
        scale = real[:, c].std() + 1e-12
        w1 = wasserstein_distance(real[:, c].ravel(), fake[:, c].ravel()) / scale
        pr, pf = _psd(real, c).mean(0), _psd(fake, c).mean(0)
        psd = np.abs(np.log(pf + 1e-12) - np.log(pr + 1e-12)).mean()
        acf = np.abs(_acf(fake, c).mean(0) - _acf(real, c).mean(0)).mean()
        out[f"w1_{CH[c]}"] = w1
        out[f"psd_{CH[c]}"] = psd
        out[f"acf_{CH[c]}"] = acf
        w1s.append(w1)
    out["w1_mean"] = float(np.mean(w1s))
    return out


def physical(traj: np.ndarray, real: np.ndarray) -> dict:
    rk, rd = residuals(traj)
    a_scale = np.median(np.abs(real[:, 2])) + 1e-12
    F_scale = np.median(np.abs(real[:, 3])) + 1e-12
    return {
        "r_energy": float(np.median(np.abs(energy_residual(traj)))),
        "r_dyn": float(np.median(np.abs(rd)) / F_scale),
        "r_kin": float(np.median(np.abs(rk)) / a_scale),
    }


def evaluate(real: np.ndarray, fake: np.ndarray) -> dict:
    return {**marginal(real, fake), **physical(fake, real), 
            **{f"floor_{k}": v for k, v in physical(real, real).items()}}
