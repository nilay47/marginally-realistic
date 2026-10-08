"""
Damped driven PENDULUM. Drop-in replacement for sim.py: same public API, so
ddpm.py, metrics.py, train.py, stats.py and figures.py all run unchanged.

    m l^2 th'' + c th' + m g l sin(th) = tau(t)

Channels (same order as the oscillator version):
    0: th     angle              (proprioception)
    1: om     angular velocity   (proprioception)
    2: al     angular accel      (IMU)
    3: tau    applied torque     (torque sensor)

The point of this system is that the restoring term is sin(th), not th, so the
dynamics is genuinely nonlinear. The energy balance identity keeps exactly the
same shape:

    E(T) - E(0) = int tau*om dt  -  c int om^2 dt,
    E = 0.5 m l^2 om^2 + m g l (1 - cos th)

so r_energy is still derivative-free and still floors out at quadrature error.

NOTE on projection: the dynamic constraint contains sin(th) and is therefore
NOT an affine subspace, so no exact linear projector exists for it. Here
projection_matrix() enforces the two KINEMATIC relations only
(om = d th/dt, al = d om/dt). Say this explicitly in the paper: on a nonlinear
system, post-hoc linear repair cannot even in principle reach the dynamics.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

# --- system constants -------------------------------------------------------
M, L, G, C = 1.0, 1.0, 9.81, 0.4      # natural freq sqrt(g/l) = 3.13 rad/s
K = M * G * L                          # small-angle stiffness, for reference
SIGMA = 8.0                            # torque amplitude: large enough that
                                       # sin(th) departs clearly from th
N_MODES, W_MAX = 16, 8.0
T_LEN = 128
DT = 0.02
SUBSAMPLE = 10
EDGE = 2
CH = ["th", "om", "al", "tau"]


def _forcing(n_traj: int, t: np.ndarray, rng) -> np.ndarray:
    """Band-limited random torque; see sim.py docstring for why not OU."""
    w = rng.uniform(0.5, W_MAX, (n_traj, N_MODES))
    amp = rng.normal(0.0, SIGMA / np.sqrt(N_MODES), (n_traj, N_MODES))
    ph = rng.uniform(0, 2 * np.pi, (n_traj, N_MODES))
    return np.einsum("nm,nmt->nt",
                     amp, np.cos(w[:, :, None] * t[None, None, :]
                                 + ph[:, :, None]))


def simulate(n_traj: int = 20000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt_f = DT / SUBSAMPLE
    n_fine = T_LEN * SUBSAMPLE
    t_fine = np.arange(n_fine) * dt_f
    TAU = _forcing(n_traj, t_fine, rng)

    th = rng.normal(0.0, 0.3, n_traj)
    om = rng.normal(0.0, 1.0, n_traj)
    out = np.zeros((n_traj, 4, T_LEN))

    def acc(th_, om_, tau_):
        return (tau_ - C * om_ - M * G * L * np.sin(th_)) / (M * L * L)

    rec = 0
    for i in range(n_fine):
        if i % SUBSAMPLE == 0:
            out[:, 0, rec] = th
            out[:, 1, rec] = om
            out[:, 2, rec] = acc(th, om, TAU[:, i])
            out[:, 3, rec] = TAU[:, i]
            rec += 1
            if rec == T_LEN:
                break
        T0 = TAU[:, i]
        T1 = TAU[:, min(i + 1, n_fine - 1)]
        Th = 0.5 * (T0 + T1)
        k1t, k1o = om, acc(th, om, T0)
        k2t, k2o = om + .5 * dt_f * k1o, acc(th + .5 * dt_f * k1t, om + .5 * dt_f * k1o, Th)
        k3t, k3o = om + .5 * dt_f * k2o, acc(th + .5 * dt_f * k2t, om + .5 * dt_f * k2o, Th)
        k4t, k4o = om + dt_f * k3o, acc(th + dt_f * k3t, om + dt_f * k3o, T1)
        th = th + dt_f / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
        om = om + dt_f / 6 * (k1o + 2 * k2o + 2 * k3o + k4o)
    return out.astype(np.float32)


# ------------------------------------------------------------------ residuals

def diff_matrix(T: int = T_LEN, dt: float = DT) -> np.ndarray:
    D = np.zeros((T, T))
    for t in range(2, T - 2):
        D[t, t - 2], D[t, t - 1] = 1.0 / (12 * dt), -8.0 / (12 * dt)
        D[t, t + 1], D[t, t + 2] = 8.0 / (12 * dt), -1.0 / (12 * dt)
    for t in (1, T - 2):
        D[t, t - 1], D[t, t + 1] = -0.5 / dt, 0.5 / dt
    D[0, 0], D[0, 1] = -1.0 / dt, 1.0 / dt
    D[-1, -2], D[-1, -1] = -1.0 / dt, 1.0 / dt
    return D


_D = diff_matrix()


def energy_residual(traj: np.ndarray) -> np.ndarray:
    """Global energy budget. Nonlinear potential, still derivative-free."""
    th, om, tau = traj[:, 0], traj[:, 1], traj[:, 3]
    E = 0.5 * M * L * L * om ** 2 + M * G * L * (1.0 - np.cos(th))
    work = simpson(tau * om, dx=DT, axis=1)
    diss = C * simpson(om ** 2, dx=DT, axis=1)
    return (E[:, -1] - E[:, 0] - work + diss) / (np.abs(E[:, 0]) + 1e-9)


def residuals(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    th, om, al, tau = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]
    sl = slice(EDGE, -EDGE)
    r_kin = (al - om @ _D.T)[:, sl]
    r_dyn = (M * L * L * al + C * om + M * G * L * np.sin(th) - tau)[:, sl]
    return r_kin, r_dyn


def projection_matrix() -> np.ndarray:
    """
    Kinematic constraints ONLY: om = D th and al = D om. The dynamic relation
    contains sin(th) and does not define an affine subspace, so it admits no
    exact linear projector. That is itself worth reporting.
    """
    T = T_LEN
    I, Z = np.eye(T), np.zeros((T, T))
    A = np.block([
        [-_D,  I,   Z,  Z],
        [Z,   -_D,  I,  Z],
    ])
    return np.eye(4 * T) - A.T @ np.linalg.solve(A @ A.T, A)


def project(traj: np.ndarray, P: np.ndarray | None = None) -> np.ndarray:
    P = projection_matrix() if P is None else P
    n = traj.shape[0]
    return (traj.reshape(n, -1) @ P.T).reshape(n, 4, T_LEN).astype(np.float32)


if __name__ == "__main__":
    d = simulate(2000, seed=0)
    th = d[:, 0]
    rk, rd = residuals(d)
    print("PENDULUM  shape", d.shape)
    for i, c in enumerate(CH):
        print(f"  {c}: mean {d[:, i].mean():+.3f}  std {d[:, i].std():.3f}")

    # Is it actually nonlinear? If sin(th) ~ th everywhere this is just the
    # oscillator again and the appendix claim would be empty.
    lin_err = np.abs(np.sin(th) - th) / (np.abs(np.sin(th)) + 1e-9)
    print(f"\nNONLINEARITY CHECK")
    print(f"  median |th|            = {np.median(np.abs(th)):.3f} rad")
    print(f"  90th pct |th|          = {np.percentile(np.abs(th), 90):.3f} rad")
    print(f"  frac |th| > 1.0 rad    = {(np.abs(th) > 1.0).mean():.3f}")
    print(f"  median |sin th - th|/|sin th| = {np.median(lin_err):.4f}"
          f"   <-- want > 0.02, ideally > 0.05")
    print(f"  frac |th| > pi (rotations) = {(np.abs(th) > np.pi).mean():.4f}"
          f"   <-- want small; if > 0.05 lower SIGMA")

    print(f"\nREAL-DATA FLOOR")
    print(f"  median |r_energy| = {np.median(np.abs(energy_residual(d))):.4e}"
          f"   <-- HEADLINE. Want < 1e-3.")
    print(f"  median |r_dyn|    = {np.median(np.abs(rd)):.4e}")
    print(f"  median |r_kin|    = {np.median(np.abs(rk)):.4e}")
