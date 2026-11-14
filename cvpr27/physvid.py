"""
physvid.py -- rendered pendulum video + gyro + torque benchmark (CVPR 27 MVP).

Modalities per sample (T frames at dt):
    video  : (T, H, W) float in [0, 1], a Gaussian bob on black, fixed pivot
    gyro   : (T,) angular rate omega [rad/s]
    torque : (T,) external torque tau [N m]

Physics: m l^2 th'' + c th' + m g l sin th = tau(t), m = l = 1, g = 9.81, c = 0.4,
band-limited random torque (16 Fourier modes, 0.5-8 rad/s), RK4 at dt/10,
same protocol as the workshop pendulum (Appendix E).

Diagnostics (all derivative-free on generated signals):
  1. global energy balance, theta from PIXELS, omega from GYRO, tau from TORQUE
       E = 0.5 w^2 + g (1 - cos th);  E(T) - E(0) = int tau w dt - c int w^2 dt
  2. multi-window energy balance (same identity on sliding windows)
  3. integrated visual-gyro consistency
       th_video(t_b) - th_video(t_a) = int_{t_a}^{t_b} w_gyro dt

Every diagnostic is reported next to its floor on real data:
  - numerical floor   : true simulator theta (no rendering)
  - visual floor      : theta extracted from rendered frames (float)
  - visual floor uint8: theta extracted from 8-bit frames (what you train on)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.integrate import simpson

# ---------------------------------------------------------------- constants
# Clip length / frame step / window length can be overridden by env vars BEFORE import:
#   PV_T=64 PV_DT=0.04 PV_WIN=8  (training default: 64 frames over 2.56 s)
M, L, G, C = 1.0, 1.0, 9.81, 0.4
DT = float(os.environ.get("PV_DT", 0.02))
T = int(os.environ.get("PV_T", 128))
SUB = 10
N_MODES, F_LO, F_HI = 16, 0.5, 8.0
TAU_AMP = 8.0                       # = workshop pendulum SIGMA; gives matching theta stats at T=64, DT=0.04
                                    # (retuned from 3.0, which was far less nonlinear; see cvpr27/CVPR27_SPEC.md 1.1)

H = W = 32
PIVOT = (15.5, 15.5)                # (row, col) in pixel coordinates, image centre
R_PX = 12.0                         # bob radius on screen; full circle fits in 32x32
SIGMA = 1.5                         # Gaussian bob width in pixels

WIN = int(os.environ.get("PV_WIN", 16))   # window length in frames, Simpson on WIN+1 points
STRIDE = max(1, WIN // 8)


# ---------------------------------------------------------------- simulator
def _forcing_params(rng, n):
    w = rng.uniform(F_LO, F_HI, size=(n, N_MODES))
    a = rng.normal(size=(n, N_MODES)) / np.sqrt(N_MODES)
    p = rng.uniform(0, 2 * np.pi, size=(n, N_MODES))
    return w, a, p


def _tau(t, w, a, p):
    # t scalar or (n,), returns (n,)
    t = np.asarray(t)[..., None]
    return TAU_AMP * np.sum(a * np.sin(w * t + p), axis=-1)


def simulate(n, seed=0):
    """Returns dict with theta, omega, tau of shape (n, T)."""
    rng = np.random.default_rng(seed)
    w, a, p = _forcing_params(rng, n)
    th = rng.normal(0, 0.3, size=n)
    om = rng.normal(0, 1.0, size=n)
    h = DT / SUB

    def f(t, th, om):
        acc = (_tau(t, w, a, p) - C * om - M * G * L * np.sin(th)) / (M * L * L)
        return om, acc

    TH, OM = np.empty((n, T)), np.empty((n, T))
    t = 0.0
    for k in range(T):
        TH[:, k], OM[:, k] = th, om
        for _ in range(SUB):
            k1 = f(t, th, om)
            k2 = f(t + h / 2, th + h / 2 * k1[0], om + h / 2 * k1[1])
            k3 = f(t + h / 2, th + h / 2 * k2[0], om + h / 2 * k2[1])
            k4 = f(t + h, th + h * k3[0], om + h * k3[1])
            th = th + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            om = om + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            t += h
    tgrid = np.arange(T) * DT
    TAU = np.stack([_tau(tt, w, a, p) for tt in tgrid], axis=1)
    return {"theta": TH, "omega": OM, "tau": TAU}


# ---------------------------------------------------------------- renderer
_rr, _cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")


def render(theta, chunk=256):
    """theta (n, T) -> video (n, T, H, W) float32 in [0, 1]. Pendulum hangs downward."""
    n = theta.shape[0]
    out = np.empty((n, T, H, W), dtype=np.float32)
    for s in range(0, n, chunk):
        th = theta[s:s + chunk]
        r0 = PIVOT[0] + R_PX * np.cos(th)          # row grows downward
        c0 = PIVOT[1] + R_PX * np.sin(th)
        d2 = (_rr - r0[..., None, None]) ** 2 + (_cc - c0[..., None, None]) ** 2
        out[s:s + chunk] = np.exp(-d2 / (2 * SIGMA ** 2))
    return out


def to_uint8(video):
    return np.clip(np.round(video * 255), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- extractor
def extract_theta(video, half=3, min_peak=0.3, bg=0.05):
    """
    Analytic pixel -> angle extractor (no learning).
    Per frame: locate the brightest pixel, take an intensity-weighted centroid in a
    (2*half+1)^2 window around it after subtracting a background level, then
    theta = atan2(col - pivot_col, row - pivot_row). Angles are unwrapped over time.

    Returns theta (n, T) and a validity mask (n, T): False where the peak is too
    dim (no bob) or the window is empty. Generated samples with many invalid frames
    should be counted, not silently dropped.
    """
    v = video.astype(np.float32)
    if video.dtype == np.uint8:
        v = v / 255.0
    n = v.shape[0]
    flat = v.reshape(n, T, -1)
    idx = flat.argmax(-1)
    peak = np.take_along_axis(flat, idx[..., None], -1)[..., 0]
    pr, pc = np.divmod(idx, W)

    offs = np.arange(-half, half + 1)
    rr = np.clip(pr[..., None, None] + offs[:, None], 0, H - 1)      # (n,T,k,1)
    cc = np.clip(pc[..., None, None] + offs[None, :], 0, W - 1)      # (n,T,1,k)
    rr_b = np.broadcast_to(rr, rr.shape[:-1] + (len(offs),))
    cc_b = np.broadcast_to(cc, cc.shape[:-2] + (len(offs), len(offs)))
    ni = np.arange(n)[:, None, None, None]
    ti = np.arange(T)[None, :, None, None]
    patch = v[ni, ti, rr_b, cc_b]
    wts = np.clip(patch - bg, 0, None)
    s = wts.sum((-1, -2))
    valid = (peak >= min_peak) & (s > 1e-8)
    s = np.where(s > 1e-8, s, 1.0)
    r = (wts * rr_b).sum((-1, -2)) / s
    c = (wts * cc_b).sum((-1, -2)) / s
    th = np.arctan2(c - PIVOT[1], r - PIVOT[0])
    th = np.unwrap(th, axis=1)
    return th, valid


# ---------------------------------------------------------------- diagnostics
def _energy(th, om):
    return 0.5 * M * L * L * om ** 2 + M * G * L * (1 - np.cos(th))


def _scale(th, om):
    """Per-sample energy scale: mean |E| over the trajectory (robust to E(0) ~ 0)."""
    return np.maximum(np.abs(_energy(th, om)).mean(1), 1e-8)


def energy_residual(th, om, tau, a=0, b=T):
    """|E(b)-E(a) - (int tau w - c int w^2)| over frames [a, b), per sample, unnormalised."""
    seg = slice(a, b)
    E = _energy(th[:, seg], om[:, seg])
    work = simpson(tau[:, seg] * om[:, seg], dx=DT, axis=1)
    diss = C * simpson(om[:, seg] ** 2, dx=DT, axis=1)
    return np.abs(E[:, -1] - E[:, 0] - (work - diss))


def windows():
    return [(s, s + WIN + 1) for s in range(0, T - WIN, STRIDE)]


def diagnostics(th, om, tau):
    """Per-sample scores. All dimensionless except kin_rad (radians)."""
    scale = _scale(th, om)
    E0 = np.maximum(np.abs(_energy(th[:, :1], om[:, :1]))[:, 0], 1e-8)
    glob = energy_residual(th, om, tau)
    win_e = np.stack([energy_residual(th, om, tau, a, b) for a, b in windows()], 1)
    win_k = np.stack([np.abs(th[:, b - 1] - th[:, a] -
                             simpson(om[:, a:b], dx=DT, axis=1)) for a, b in windows()], 1)
    return {
        "r_energy_E0": glob / E0,                 # workshop convention (|E(0)|)
        "r_energy": glob / scale,                 # global, mean-|E| normalised
        "r_energy_win_max": win_e.max(1) / scale, # worst window
        "r_energy_win_med": np.median(win_e, 1) / scale,
        "kin_rad_max": win_k.max(1),              # worst-window pixel-vs-gyro mismatch, radians
        "kin_rad_med": np.median(win_k, 1),
    }


def summarise(d):
    return {k: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))}
            for k, v in d.items()}


# ---------------------------------------------------------------- corruptions (unit tests only)
def orthogonal_torque_corruption(om, tau, rng, rel=0.3):
    """
    tau' = tau + d, with int d * w dt = 0 exactly under Simpson weights, so the GLOBAL
    energy residual is unchanged while the torque is wrong everywhere. Constructed to
    defeat the global check; a unit test, never a headline result.
    """
    n = tau.shape[0]
    t = np.arange(T) * DT
    wf, af, pf = _forcing_params(rng, n)
    s = np.sum(af[:, None, :] * np.sin(wf[:, None, :] * t[None, :, None] + pf[:, None, :]), -1)
    num = simpson(s * om, dx=DT, axis=1)
    den = simpson(om * om, dx=DT, axis=1)
    d = s - (num / den)[:, None] * om
    d *= rel * tau.std(1, keepdims=True) / d.std(1, keepdims=True)
    return tau + d


# ---------------------------------------------------------------- self-test
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="physvid_selftest.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed + 1)

    t0 = time.time()
    d = simulate(args.n, seed=args.seed)
    th, om, tau = d["theta"], d["omega"], d["tau"]
    stats = {
        "median_abs_theta": float(np.median(np.abs(th))),
        "frac_samples_exceed_1rad": float((np.abs(th).max(1) > 1).mean()),
        "frac_over_the_top": float((np.abs(th).max(1) > np.pi).mean()),
        "max_rad_per_frame": float(np.abs(np.diff(th, axis=1)).max()),
    }
    vid = render(th)
    vid8 = to_uint8(vid)
    th_f, ok_f = extract_theta(vid)
    th_8, ok_8 = extract_theta(vid8)

    res = {"n": args.n, "seed": args.seed, "theta_stats": stats,
           "extractor": {"invalid_frac_float": float(1 - ok_f.mean()),
                         "invalid_frac_uint8": float(1 - ok_8.mean()),
                         "theta_err_rad_median_uint8": float(np.median(np.abs(th_8 - th))),
                         "theta_err_rad_p99_uint8": float(np.percentile(np.abs(th_8 - th), 99))}}

    res["floor_numerical"] = summarise(diagnostics(th, om, tau))
    res["floor_visual_float"] = summarise(diagnostics(th_f, om, tau))
    res["floor_visual_uint8"] = summarise(diagnostics(th_8, om, tau))

    # corruption 1: video and sensors mispaired (shuffle oracle in video form)
    perm = rng.permutation(args.n)
    res["corrupt_mispaired"] = summarise(diagnostics(th_8[perm], om, tau))
    # corruption 2: video lags sensors by 3 frames (60 ms)
    th_lag = np.concatenate([np.repeat(th_8[:, :1], 3, 1), th_8[:, :-3]], 1)
    res["corrupt_lag3"] = summarise(diagnostics(th_lag, om, tau))
    # corruption 3: torque wrong locally, globally balanced
    tau_c = orthogonal_torque_corruption(om, tau, rng)
    res["corrupt_orth_torque"] = summarise(diagnostics(th_8, om, tau_c))
    # robustness: pixel noise on uint8 video
    for s in (0.02, 0.05, 0.1):
        noisy = to_uint8(np.clip(vid + rng.normal(0, s, vid.shape).astype(np.float32), 0, 1))
        thn, okn = extract_theta(noisy)
        r = summarise(diagnostics(thn, om, tau))
        r["invalid_frac"] = float(1 - okn.mean())
        res[f"robust_pixel_noise_{s}"] = r
    res["seconds"] = round(time.time() - t0, 1)

    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)

    # compact console table
    keys = ["r_energy", "r_energy_win_max", "kin_rad_max"]
    print(json.dumps(res["theta_stats"]), json.dumps(res["extractor"]), sep="\n")
    print(f"{'condition':28s}" + "".join(f"{k:>20s}" for k in keys))
    for cond in ["floor_numerical", "floor_visual_float", "floor_visual_uint8",
                 "corrupt_mispaired", "corrupt_lag3", "corrupt_orth_torque",
                 "robust_pixel_noise_0.02", "robust_pixel_noise_0.05", "robust_pixel_noise_0.1"]:
        print(f"{cond:28s}" + "".join(f"{res[cond][k]['median']:20.3e}" for k in keys))
    print(f"done in {res['seconds']}s -> {args.out}")


if __name__ == "__main__":
    main()
