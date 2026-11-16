"""
JOB 2: a second system. Driven damped oscillator  m x'' + c x' + k x = F(t)  (workshop sim.py constants M,C,K = 1.0, 0.4, 10.0; band-limited forcing: random
cosine series, N_MODES=16, frequencies U(0.5, 8), amplitude N(0, SIGMA/sqrt(16)), SIGMA=3). T=32, DT=0.04 (same as the pendulum audit), RK4 with 10 substeps.
Render: 32x32 Gaussian blob (sigma 1.5 px) moving horizontally on row 15.5; column = 15.5 + s*x with s fixed so that the 99th percentile of |x| (train sim,
seed 0, n=2000) spans 70% of the half... i.e. +-0.35*32 px (motion fills ~70% of the width). Sensors: velocity v ('gyro' slot in the shared code) and force F ('torque' slot).
Extractor: brightest pixel + background-subtracted intensity centroid (as physvid.extract_theta), x_hat = (col_centroid - 15.5)/s.
Checks (nominal m,c,k audit):  r_E   global energy balance |E(T)-E(0) - (int F v - c int v^2)| / mean|E|, E = 0.5 m v^2 + 0.5 k x_hat^2
                               r_E^win  max over 9-point windows (WIN=8, stride 1) of the same, / mean|E|
                               kin     max over windows of |x_hat(b)-x_hat(a) - int v dt|  (video-velocity consistency, in x units)
Reference R: 512 clips (seed 1); test clips A and every condition: 512 clips (seed 2; same forcing and initial state, only the named change differs).
Conditions: stiffness x{.9,.95,1.05,1.1,1.25}, damping x{0,2,4}, force-sensor gain x{1.1,1.25}, lag {1,4}, time_reversed, mispaired, cancelling_force
(F' = F + d, d built so int d v dt = 0 under Simpson: the global energy balance is unchanged). Same protocol as the pendulum audit (ap.eval_cond):
W1(v), W1(F), MMD^2 (m=128 subsets), medians+IQR of the 3 checks, AUROC vs R. FVD not computed.   python -m cvpr27.osc2
Outputs (new files only): ~/physgen_local/paper_figs/oscillator_results.{md,json}; sims cached under audit_cache/osc/.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.integrate import simpson

from cvpr27 import audit_paper as ap
from cvpr27 import evalmetrics as em
from cvpr27 import physvid as pv
from cvpr27.common import atomic_json, drive_root, fmt_s, git_info, new_path
from cvpr27.data import Norm

M, C, K = 1.0, 0.4, 10.0
SIGMA, N_MODES, W_MAX = 3.0, 16, 8.0
H = W = 32
PIX_SIGMA, ROW, COL0 = 1.5, 15.5, 15.5
SUB = 10
OUT = drive_root() / "paper_figs"
T, DT = pv.T, pv.DT
_rr, _cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")


def forcing_params(rng, n):
    return rng.uniform(0.5, W_MAX, (n, N_MODES)), rng.normal(0, SIGMA / np.sqrt(N_MODES), (n, N_MODES)), rng.uniform(0, 2 * np.pi, (n, N_MODES))


def force_at(p, t):
    w, a, ph = p
    t = np.asarray(t)
    if t.ndim == 0:
        return (a * np.cos(w * t + ph)).sum(1)
    return np.einsum("nm,nmt->nt", a, np.cos(w[:, :, None] * t[None, None, :] + ph[:, :, None]))


def simulate(n, seed, kf=1.0, cf=1.0):
    """x, v, F at the recorded times (n, T). True dynamics use k*kf, c*cf; the forcing/initial state depend only on (n, seed)."""
    rng = np.random.default_rng(seed)
    p = forcing_params(rng, n)
    x, v = rng.normal(0, 0.3, n), rng.normal(0, 1.0, n)
    k, c, h = K * kf, C * cf, DT / SUB
    f = lambda t, x_, v_: (v_, (force_at(p, t) - c * v_ - k * x_) / M)
    X, V = np.empty((n, T)), np.empty((n, T))
    t = 0.0
    for i in range(T):
        X[:, i], V[:, i] = x, v
        for _ in range(SUB):
            k1 = f(t, x, v); k2 = f(t + h / 2, x + h / 2 * k1[0], v + h / 2 * k1[1]); k3 = f(t + h / 2, x + h / 2 * k2[0], v + h / 2 * k2[1]); k4 = f(t + h, x + h * k3[0], v + h * k3[1])
            x = x + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); v = v + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]); t += h
    return {"x": X, "v": V, "F": force_at(p, np.arange(T) * DT), "p": p}


_S = None


def scale():
    """pixels per unit x: fixed from the train sim so that the 99th percentile of |x| reaches 0.35*32 px (motion fills ~70% of the width)."""
    global _S
    if _S is None:
        tr = simulate(2000, 0)
        _S = 0.35 * W / float(np.percentile(np.abs(tr["x"]), 99))
    return _S


def render(x):
    col = COL0 + scale() * x
    d2 = (_rr - ROW) ** 2 + (_cc - col[..., None, None]) ** 2
    return np.exp(-d2 / (2 * PIX_SIGMA ** 2)).astype(np.float32)


def to_u8(v):
    return np.clip(np.round(v * 255), 0, 255).astype(np.uint8)


def extract_x(video, half=3, min_peak=0.3, bg=0.05):
    v = video.astype(np.float32) / 255.0
    n = v.shape[0]
    flat = v.reshape(n, T, -1)
    idx = flat.argmax(-1)
    peak = np.take_along_axis(flat, idx[..., None], -1)[..., 0]
    pr, pc = np.divmod(idx, W)
    offs = np.arange(-half, half + 1)
    rr = np.clip(pr[..., None, None] + offs[:, None], 0, H - 1)
    cc = np.clip(pc[..., None, None] + offs[None, :], 0, W - 1)
    rr_b = np.broadcast_to(rr, rr.shape[:-1] + (len(offs),))
    cc_b = np.broadcast_to(cc, cc.shape[:-2] + (len(offs), len(offs)))
    patch = v[np.arange(n)[:, None, None, None], np.arange(T)[None, :, None, None], rr_b, cc_b]
    wts = np.clip(patch - bg, 0, None)
    s = wts.sum((-1, -2))
    valid = (peak >= min_peak) & (s > 1e-8)
    s = np.where(s > 1e-8, s, 1.0)
    c = (wts * cc_b).sum((-1, -2)) / s
    return (c - COL0) / scale(), valid


def score_osc(video, v, F):
    """Per-clip scores (same keys as the pendulum audit so ap.eval_cond can be reused): r_energy, r_energy_win_max, kin_rad_max (= x-unit mismatch)."""
    xh, ok = extract_x(video)
    E = 0.5 * M * v ** 2 + 0.5 * K * xh ** 2
    scl = np.maximum(np.abs(E).mean(1), 1e-8)
    res = lambda a, b: np.abs(E[:, b - 1] - E[:, a] - (simpson(F[:, a:b] * v[:, a:b], dx=DT, axis=1) - C * simpson(v[:, a:b] ** 2, dx=DT, axis=1)))
    glob = res(0, T)
    ws = pv.windows()
    win_e = np.stack([res(a, b) for a, b in ws], 1)
    win_k = np.stack([np.abs(xh[:, b - 1] - xh[:, a] - simpson(v[:, a:b], dx=DT, axis=1)) for a, b in ws], 1)
    d = {"r_energy": glob / scl, "r_energy_win_max": win_e.max(1) / scl, "kin_rad_max": win_k.max(1), "invalid_frac_clip": (~ok).mean(1)}
    for k_, a_ in d.items():
        em.check_finite(k_, a_)
    return d


def clip(sim, gain=1.0):
    return {"video": to_u8(render(sim["x"])), "omega": sim["v"].astype(np.float64), "tau": (gain * sim["F"]).astype(np.float64)}


def cancelling_force(v, F, rng, rel=0.3):
    p = forcing_params(rng, len(F))
    s = force_at(p, np.arange(T) * DT)
    num, den = simpson(s * v, dx=DT, axis=1), simpson(v * v, dx=DT, axis=1)
    d = s - (num / den)[:, None] * v
    d *= rel * F.std(1, keepdims=True) / d.std(1, keepdims=True)
    return F + d


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--n", type=int, default=512)
    ap_.add_argument("--m_sub", type=int, default=128)
    ap_.add_argument("--B", type=int, default=200)
    ap_.add_argument("--smoke", action="store_true")
    a = ap_.parse_args()
    if a.smoke:
        a.n, a.m_sub, a.B = 16, 4, 6
    a.with_fvd, a.n_fvd = False, 0
    assert pv.T == 32 and abs(pv.DT - 0.04) < 1e-12 and pv.WIN == 8
    t0 = time.time()
    em.score_clips = score_osc                                    # eval_cond (audit_paper) scores clips through em.score_clips
    ap.em.score_clips = score_osc
    n, S = a.n, scale()
    OUT.mkdir(parents=True, exist_ok=True)
    root = drive_root() / "audit_cache" / ("osc_smoke" if a.smoke else "osc")
    root.mkdir(parents=True, exist_ok=True)

    def cached(name, fn):
        p = root / f"{name}_n{n}.npz"
        if p.exists():
            z = np.load(p); return {k: z[k] for k in ("video", "omega", "tau")}
        d = fn(); np.savez(p, **d); return d
    # --- reference (floors) + norm stats
    R = cached("ref_s1", lambda: clip(simulate(n, 1)))
    tr = simulate(2000, 0)
    norm = Norm({"gyro_mean": float(tr["v"].mean()), "gyro_std": float(tr["v"].std()), "torque_mean": float(tr["F"].mean()), "torque_std": float(tr["F"].std())})
    per_R = score_osc(R["video"], R["omega"], R["tau"])
    floor = em.summarise({**per_R, **{k: per_R["r_energy"] for k in ("r_energy_E0", "r_energy_win_med", "kin_rad_med")}}) if False else None
    fl = {"r_energy": float(np.median(per_R["r_energy"])), "r_energy_win_max": float(np.median(per_R["r_energy_win_max"])), "kin_max": float(np.median(per_R["kin_rad_max"])),
          "invalid_frame_frac": float(per_R["invalid_frac_clip"].mean())}
    print(f"scale {S:.2f} px/unit x | real floors (512 clips): {fl}", flush=True)
    if fl["r_energy"] > 1e-2 or fl["invalid_frame_frac"] > 0.01:
        raise RuntimeError(f"STOP (rule): real floor check failed: {fl}")
    ref = {"om": R["omega"], "tau": R["tau"], "norm": norm, "per": per_R, "metric": None,
           "mmd_feats": em.mmd_features(R["video"], R["omega"], R["tau"], norm)}
    # --- conditions
    base = simulate(n, 2)
    conds = {"real_vs_real": lambda: clip(base)}
    for v_ in (0.9, 0.95, 1.05, 1.1, 1.25):
        conds[f"stiffness={v_}"] = lambda v_=v_: clip(simulate(n, 2, kf=v_))
    for v_ in (0.0, 2.0, 4.0):
        conds[f"damping={v_}"] = lambda v_=v_: clip(simulate(n, 2, cf=v_))
    for v_ in (1.1, 1.25):
        conds[f"force_gain={v_}"] = lambda v_=v_: clip(base, gain=v_)
    for k_ in (1, 4):
        conds[f"lag={k_}"] = lambda k_=k_: {**clip(base), "video": ap.lag_video(clip(base)["video"], k_)}
    conds["time_reversed"] = lambda: (lambda d: {"video": np.ascontiguousarray(d["video"][:, ::-1]), "omega": -d["omega"][:, ::-1].copy(), "tau": d["tau"][:, ::-1].copy()})(clip(base))
    conds["mispaired"] = lambda: (lambda d, p: {**d, "video": d["video"][p]})(clip(base), np.where(np.random.default_rng(77).permutation(n) == np.arange(n),
                                                                                                 (np.random.default_rng(77).permutation(n) + 1) % n, np.random.default_rng(77).permutation(n)))
    conds["cancelling_force"] = lambda: (lambda d: {**d, "tau": cancelling_force(d["omega"], d["tau"], np.random.default_rng(78))})(clip(base))
    rng = np.random.default_rng(31)
    res = {"git": git_info(), "system": {"m": M, "c": C, "k": K, "T": T, "DT": DT, "px_per_unit_x": S, "sigma_forcing": SIGMA, "n": n, "seeds": {"ref": 1, "test": 2, "train_for_norm_and_scale": 0}},
           "ci_methods": {"per_sample": f"percentile bootstrap B={a.B}", "set_level": em.SUBSAMPLE_METHOD, "m": a.m_sub}, "real_floor_reference": fl, "conditions": {}}
    for i, (name, fn) in enumerate(conds.items()):
        d = cached(name.replace("=", ""), fn)
        c = ap.eval_cond(name, d, ref, a, rng)
        res["conditions"][name] = c
        print(f"[{i+1}/{len(conds)}] {name}: W1v {c['w1']['w1_gyro']:.3g} W1F {c['w1']['w1_torque']:.3g} MMD {c['mmd']['subsample_median']:.3g} r_E {c['physical']['r_energy']['median']:.3g} "
              f"win {c['physical']['r_energy_win_max']['median']:.3g} kin {c['physical']['kin_rad_max']['median']:.3g} AUROC {[round(c['auroc_vs_real'][k], 3) for k in ap.KEYS]} ({fmt_s(time.time()-t0)})", flush=True)
    res["seconds"] = time.time() - t0
    tag = "oscillator_smoke" if a.smoke else "oscillator_results"
    atomic_json(OUT / f"{tag}.json", res)
    f = ap._f
    s = lambda d: f"{f(d['subsample_median'])} [{f(d['subsample_ci95'][0])}, {f(d['subsample_ci95'][1])}]"
    md = ["# Second system: driven damped oscillator (m=1, c=0.4, k=10), T=32, DT=0.04, n=512 per condition vs a 512-clip real reference", "",
          f"Blob moves horizontally, {S:.1f} px per unit x (p99|x| fills 70% of the width). Sensors: velocity v, force F. r_E = global energy balance (E = ½mv² + ½kx̂²), r_E^win = worst 9-point window, kin = worst-window |x̂(b)−x̂(a)−∫v| (x units), all nominal m, c, k. "
          f"Real floors (reference clips): r_E {f(fl['r_energy'])}, win_max {f(fl['r_energy_win_max'])}, kin {f(fl['kin_max'])}, invalid frames {f(fl['invalid_frame_frac'])}. FVD not computed. MMD² cell: median [2.5-97.5] over {a.B} subsets of m={a.m_sub}; "
          f"W1 and physics: point [95% bootstrap CI, B={a.B}], physics also IQR. AUROC: real reference = negatives.", "",
          "| condition | W1 v | W1 F | MMD² | r_E med [IQR] | r_E^win med | kin_max med [IQR] | invalid frames | AUROC r_E | AUROC r_E^win | AUROC kin |", "|" + "---|" * 11]
    for name, c in res["conditions"].items():
        p, w = c["physical"], c["w1"]
        md.append(f"| {name} | {f(w['w1_gyro'])} [{f(w['w1_gyro_ci95'][0])}, {f(w['w1_gyro_ci95'][1])}] | {f(w['w1_torque'])} [{f(w['w1_torque_ci95'][0])}, {f(w['w1_torque_ci95'][1])}] | {s(c['mmd'])} | "
                  f"{f(p['r_energy']['median'])} [{f(p['r_energy']['q25'])}, {f(p['r_energy']['q75'])}] | {f(p['r_energy_win_max']['median'])} | {f(p['kin_rad_max']['median'])} [{f(p['kin_rad_max']['q25'])}, {f(p['kin_rad_max']['q75'])}] | "
                  f"{f(c['invalid_frame_frac'])} | " + " | ".join(f(c["auroc_vs_real"][k]) for k in ap.KEYS) + " |")
    new_path(OUT / f"{tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
