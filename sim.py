"""
Stochastic mass-spring-damper -> 4-channel multimodal sensor trajectories.

    m x'' + c x' + k x = F(t),     F ~ Ornstein-Uhlenbeck

Channels (this order everywhere in the codebase):
    0: x  displacement   (proprioception)
    1: v  velocity       (proprioception)
    2: a  acceleration   (IMU)
    3: F  applied force  (force sensor)

Two exact relations hold in the data, and they are what we test generated
samples against:

    KINEMATIC   r_kin = a - dv/dt              (parameter-free)
    DYNAMIC     r_dyn = m*a + c*v + k*x - F    (needs m, c, k)

The kinematic one is the headline because verifying it requires no knowledge
of the system at all -- any sample violating it is impossible on its face.

Integration is done with RK4 at a fine dt then subsampled, so the recorded
signals satisfy both relations to well under discretisation error. The
residual you measure on REAL data is the floor; report it in the paper.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

# --- system constants (fixed for v1: keeps r_dyn exactly computable) ---
M, C, K = 1.0, 0.4, 10.0     # mass, damping, stiffness
SIGMA = 3.0                  # forcing amplitude
N_MODES, W_MAX = 16, 8.0     # band-limited forcing; w_n = sqrt(k/m) = 3.16 rad/s
T_LEN = 128                  # recorded timesteps per trajectory
DT = 0.02                    # recorded sampling interval
SUBSAMPLE = 10               # integrator steps per recorded step
CH = ["x", "v", "a", "F"]


def _forcing(n_traj: int, t: np.ndarray, rng) -> np.ndarray:
    """
    Band-limited random forcing: a random Fourier series capped at W_MAX.

    Replaces the Ornstein-Uhlenbeck process used earlier. OU sample paths are
    rough, and that roughness -- not truncation order -- set the error floor on
    every quadrature and difference we compute. Band-limiting makes F analytic,
    so the energy-budget floor becomes genuine quadrature error.
    W_MAX = 8 sits well above the natural frequency (3.16) so the system is
    still properly excited, and well below Nyquist (157 at dt = 0.02).
    """
    w = rng.uniform(0.5, W_MAX, (n_traj, N_MODES))
    amp = rng.normal(0.0, SIGMA / np.sqrt(N_MODES), (n_traj, N_MODES))
    ph = rng.uniform(0, 2 * np.pi, (n_traj, N_MODES))
    return np.einsum("nm,nmt->nt",
                     amp, np.cos(w[:, :, None] * t[None, None, :]
                                 + ph[:, :, None]))


def simulate(n_traj: int = 20000, seed: int = 0) -> np.ndarray:
    """Returns float32 array (n_traj, 4, T_LEN) in physical units."""
    rng = np.random.default_rng(seed)
    dt_fine = DT / SUBSAMPLE
    n_fine = T_LEN * SUBSAMPLE

    t_fine = np.arange(n_fine) * dt_fine
    F_fine = _forcing(n_traj, t_fine, rng)

    x = rng.normal(0.0, 0.3, n_traj)
    v = rng.normal(0.0, 1.0, n_traj)

    out = np.zeros((n_traj, 4, T_LEN), dtype=np.float64)

    def accel(x_, v_, F_):
        return (F_ - C * v_ - K * x_) / M

    rec = 0
    for i in range(n_fine):
        if i % SUBSAMPLE == 0:
            out[:, 0, rec] = x
            out[:, 1, rec] = v
            out[:, 2, rec] = accel(x, v, F_fine[:, i])
            out[:, 3, rec] = F_fine[:, i]
            rec += 1
            if rec == T_LEN:
                break

        # RK4 on [x, v] holding F piecewise-linear across the fine step
        F0 = F_fine[:, i]
        F1 = F_fine[:, min(i + 1, n_fine - 1)]
        Fh = 0.5 * (F0 + F1)

        k1x, k1v = v, accel(x, v, F0)
        k2x, k2v = v + 0.5 * dt_fine * k1v, accel(x + 0.5 * dt_fine * k1x,
                                                  v + 0.5 * dt_fine * k1v, Fh)
        k3x, k3v = v + 0.5 * dt_fine * k2v, accel(x + 0.5 * dt_fine * k2x,
                                                  v + 0.5 * dt_fine * k2v, Fh)
        k4x, k4v = v + dt_fine * k3v, accel(x + dt_fine * k3x,
                                            v + dt_fine * k3v, F1)

        x = x + dt_fine / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + dt_fine / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)

    return out.astype(np.float32)


# ------------------------------------------------------------------ residuals

EDGE = 2  # timesteps trimmed at each end before scoring (stencil width)


def diff_matrix(T: int = T_LEN, dt: float = DT) -> np.ndarray:
    """
    4th-order central differences in the interior, 2nd-order one step in,
    one-sided at the very ends. The 4th-order stencil matters: with 2nd-order
    the finite-difference error alone puts the r_kin floor at ~6% of signal
    scale, which is large enough to mask the effect being measured.
    """
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
    """
    THE HEADLINE METRIC. Global energy budget for the damped oscillator:

        E(T) - E(0) = integral F*v dt  -  c * integral v^2 dt,
        E = 0.5*m*v^2 + 0.5*k*x^2

    Uses only x, v, F. No differentiation anywhere, so the floor is trapezoid
    integration error rather than finite-difference error -- typically several
    orders of magnitude smaller. This is why it replaced r_kin as the primary
    metric: r_kin's floor sat at ~5% of signal scale, large enough that a
    slightly over-smoothed generative sample could score BELOW real data.

    It is also a better scientific test. Energy conservation is a GLOBAL
    constraint over the whole trajectory: a sample can look locally plausible
    at every timestep and still get the total energy budget badly wrong.

    Returns (N,), normalised by |E(0)|.
    """
    x, v, F = traj[:, 0], traj[:, 1], traj[:, 3]
    E = 0.5 * M * v ** 2 + 0.5 * K * x ** 2
    work = simpson(F * v, dx=DT, axis=1)
    diss = C * simpson(v ** 2, dx=DT, axis=1)
    return (E[:, -1] - E[:, 0] - work + diss) / (np.abs(E[:, 0]) + 1e-9)


def residuals(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    traj (N, 4, T) physical units -> (r_kin, r_dyn), each (N, T - 2*EDGE).

    The first and last EDGE timesteps are dropped: the stencil degrades there
    and those columns would otherwise dominate the median. Scoring on an
    interior window is standard and should be stated in the paper.
    """
    x, v, a, F = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]
    sl = slice(EDGE, -EDGE)
    r_kin = (a - v @ _D.T)[:, sl]
    r_dyn = (M * a + C * v + K * x - F)[:, sl]
    return r_kin, r_dyn


def projection_matrix() -> np.ndarray:
    """
    Orthogonal projector onto {z : both residuals vanish}, for z = [x;v;a;F]
    flattened to 4T. This is an exact linear projection, not a heuristic --
    it is the strongest possible post-hoc physics repair, which is what makes
    it a fair upper baseline.
    """
    T = T_LEN
    I, Z = np.eye(T), np.zeros((T, T))
    A = np.block([
        [Z,        -_D,      I,       Z],   # a - D v = 0
        [-K * I,   -C * I,  -M * I,   I],   # F - (m a + c v + k x) = 0
    ])
    return np.eye(4 * T) - A.T @ np.linalg.solve(A @ A.T, A)


def project(traj: np.ndarray, P: np.ndarray | None = None) -> np.ndarray:
    P = projection_matrix() if P is None else P
    n = traj.shape[0]
    return (traj.reshape(n, -1) @ P.T).reshape(n, 4, T_LEN).astype(np.float32)


if __name__ == "__main__":
    d = simulate(2000, seed=0)
    rk, rd = residuals(d)
    print("data shape", d.shape)
    for i, c in enumerate(CH):
        print(f"  {c}: mean {d[:, i].mean():+.3f}  std {d[:, i].std():.3f}")
    print(f"\nREAL-DATA FLOOR (report these in the paper):")
    print(f"  median |r_energy| = {np.median(np.abs(energy_residual(d))):.4e}"
          f"   <-- HEADLINE. Want < 1e-3.")
    print(f"  median |r_dyn|    = {np.median(np.abs(rd)):.4e}"
          f"   (scale |F| = {np.median(np.abs(d[:, 3])):.3f})")
    print(f"  median |r_kin|    = {np.median(np.abs(rk)):.4e}"
          f"   (scale |a| = {np.median(np.abs(d[:, 2])):.3f})"
          f"  <-- secondary; high floor is expected")
