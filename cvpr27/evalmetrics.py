"""
Metrics (spec section 3). Everything operates on PHYSICAL units: video uint8, gyro/torque in
rad/s and N m. Callers de-normalise first (data.Norm).

Per-sample physical scores come from physvid.diagnostics on theta extracted from pixels. Clips with
invalid frames are KEPT in the scores (their theta is garbage and shows up as a large residual) and
are counted separately via invalid_frame_frac / invalid_clip_frac.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, wasserstein_distance

from cvpr27 import physvid as pv

DIAG_KEYS = ["r_energy", "r_energy_win_max", "r_energy_win_med", "kin_rad_max", "kin_rad_med", "r_energy_E0"]


def check_finite(name: str, x) -> None:
    """Rule 5: stop and report instead of silently carrying NaN/inf."""
    x = np.asarray(x)
    if not np.isfinite(x).all():
        raise FloatingPointError(f"non-finite values in {name}: {np.count_nonzero(~np.isfinite(x))}/{x.size}")


# ------------------------------------------------------------------ physical
def score_clips(video_u8, omega, tau) -> dict:
    """Per-sample arrays: the six diagnostics, plus invalid-frame fraction per clip."""
    th, ok = pv.extract_theta(video_u8)
    d = pv.diagnostics(th, omega, tau)
    for k, v in d.items():
        check_finite(k, v)
    d["invalid_frac_clip"] = (~ok).mean(1)
    # ratio used for E3 (windowed vs global): per-sample, guarded against 0/0
    d["win_over_glob"] = d["r_energy_win_max"] / np.maximum(d["r_energy"], 1e-12)
    return d


def summarise(per: dict) -> dict:
    out = {}
    for k in DIAG_KEYS + ["win_over_glob"]:
        out[k] = {"median": float(np.median(per[k])), "p95": float(np.percentile(per[k], 95))}
    out["invalid_frame_frac"] = float(per["invalid_frac_clip"].mean())
    out["invalid_clip_frac"] = float((per["invalid_frac_clip"] > 0).mean())
    out["ratio_of_medians_win_over_glob"] = float(np.median(per["r_energy_win_max"]) /
                                                  max(np.median(per["r_energy"]), 1e-12))
    return out


def boot_median(x, B=200, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    v = np.array([np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(B)])
    return [float(np.median(x)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ------------------------------------------------------------------ sensor marginals
def sensor_w1(real_om, real_tau, om, tau) -> dict:
    out = {}
    for name, r, g in (("gyro", real_om, om), ("torque", real_tau, tau)):
        out[f"w1_{name}"] = float(wasserstein_distance(r.ravel(), g.ravel()) / (r.std() + 1e-12))
    out["w1_mean"] = float(np.mean([out["w1_gyro"], out["w1_torque"]]))
    return out


def _psd(x):
    return np.abs(np.fft.rfft(x, axis=-1)) ** 2


def sensor_psd(real_om, real_tau, om, tau) -> dict:
    """Mean |log PSD_gen - log PSD_real| over frequency bins (workshop definition), per channel."""
    out = {}
    for name, r, g in (("gyro", real_om, om), ("torque", real_tau, tau)):
        pr, pg = _psd(r).mean(0), _psd(g).mean(0)
        out[f"psd_{name}"] = float(np.abs(np.log(pg + 1e-12) - np.log(pr + 1e-12)).mean())
    out["psd_mean"] = float(np.mean([out["psd_gyro"], out["psd_torque"]]))
    return out


def sensor_marginals(real_om, real_tau, om, tau) -> dict:
    return {**sensor_w1(real_om, real_tau, om, tau), **sensor_psd(real_om, real_tau, om, tau)}


# ------------------------------------------------------------------ joint MMD
def mmd_features(video_u8, omega, tau, norm) -> np.ndarray:
    """[flattened 16x16 video (in [-1,1]), gyro, torque (standardised)] per clip. float32."""
    v = torch.from_numpy(video_u8).float() / 127.5 - 1.0                # (n,T,32,32)
    v = F.avg_pool2d(v, 2).flatten(1).numpy()                            # (n, T*256)
    s = norm.sens_to_model(omega, tau).reshape(len(omega), -1)           # (n, 2T)
    return np.concatenate([v, s], 1).astype(np.float32)


def _sqd(a, b):
    return (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T


class MMDPool:
    """RBF MMD^2 (unbiased) with ONE bandwidth for all comparisons: median heuristic on the real pool.

    Holds the full kernel blocks so subsets / bootstrap resamples are index gathers. Pairs of
    identical original indices (from resampling with replacement) are excluded from within-set sums.
    """

    def __init__(self, real_feats, gen_feats, device="cpu", real_cache=None):
        """real_cache = (gamma, Krr) from a previous pool over the same real features (saves recompute)."""
        dev = torch.device(device)
        R = torch.from_numpy(real_feats).to(dev).double()
        G = torch.from_numpy(gen_feats).to(dev).double()
        if real_cache is None:
            drr = _sqd(R, R)
            iu = torch.triu_indices(len(R), len(R), 1, device=dev)
            self.gamma = float(1.0 / drr[iu[0], iu[1]].median().clamp_min(1e-12))
            self.Krr = torch.exp(-self.gamma * drr.clamp_min(0)).cpu().numpy()
        else:
            self.gamma, self.Krr = real_cache
        k = lambda d: torch.exp(-self.gamma * d.clamp_min(0)).cpu().numpy()
        self.Kgg, self.Kgr = k(_sqd(G, G)), k(_sqd(G, R))
        self.nr, self.ng = len(R), len(G)

    def real_cache(self):
        return (self.gamma, self.Krr)

    @staticmethod
    def _within(K, idx):
        sub = K[np.ix_(idx, idx)]
        m = idx[:, None] != idx[None, :]
        return sub[m].sum() / m.sum()

    def mmd2(self, gi, ri):
        """gi indexes the generated set, ri the real pool."""
        gi, ri = np.asarray(gi), np.asarray(ri)
        return float(self._within(self.Kgg, gi) + self._within(self.Krr, ri) - 2 * self.Kgr[np.ix_(gi, ri)].mean())

    def floor(self, a_idx, b_idx):
        """real-vs-heldout-real estimator value (should be ~0; noise level of the estimator)."""
        a, b = np.asarray(a_idx), np.asarray(b_idx)
        return float(self._within(self.Krr, a) + self._within(self.Krr, b) - 2 * self.Krr[np.ix_(a, b)].mean())


# ------------------------------------------------------------------ detection
def auroc(neg, pos) -> float:
    """P(score_pos > score_neg), ties 0.5. Larger score = 'more impossible'."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    r = rankdata(np.concatenate([neg, pos]))
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))
