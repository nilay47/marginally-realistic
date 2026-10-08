"""1D DDPM: sinusoidal-time U-Net, cosine schedule, ancestral sampler, EMA."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------- network

class TimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(),
                                 nn.Linear(dim * 4, dim * 4))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=t.device) / (half - 1))
        ang = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([ang.sin(), ang.cos()], dim=-1))


class Block(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(min(8, cout), cout)
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.temb = nn.Linear(tdim, cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = F.silu(self.n1(self.c1(x)))
        h = h + self.temb(t)[:, :, None]
        h = F.silu(self.n2(self.c2(h)))
        return h + self.skip(x)


class UNet1D(nn.Module):
    """Length 128 -> 64 -> 32 -> 64 -> 128."""

    def __init__(self, channels: int = 4, base: int = 64):
        super().__init__()
        td = base * 4
        self.temb = TimeEmb(base)
        c1, c2, c3 = base, base * 2, base * 2

        self.inp = nn.Conv1d(channels, c1, 3, padding=1)
        self.d1 = Block(c1, c1, td)
        self.ds1 = nn.Conv1d(c1, c1, 4, stride=2, padding=1)
        self.d2 = Block(c1, c2, td)
        self.ds2 = nn.Conv1d(c2, c2, 4, stride=2, padding=1)
        self.mid = Block(c2, c3, td)
        self.us2 = nn.ConvTranspose1d(c3, c2, 4, stride=2, padding=1)
        self.u2 = Block(c2 + c2, c2, td)
        self.us1 = nn.ConvTranspose1d(c2, c1, 4, stride=2, padding=1)
        self.u1 = Block(c1 + c1, c1, td)
        self.out = nn.Conv1d(c1, channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        e = self.temb(t)
        h0 = self.inp(x)
        h1 = self.d1(h0, e)
        h2 = self.d2(self.ds1(h1), e)
        m = self.mid(self.ds2(h2), e)
        u2 = self.u2(torch.cat([self.us2(m), h2], 1), e)
        u1 = self.u1(torch.cat([self.us1(u2), h1], 1), e)
        return self.out(u1)


class EMA:
    """Exponential moving average of weights. Sampling from the raw weights
    gives markedly worse samples; this is standard practice for diffusion."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.state_dict().values(),
                        model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            else:
                s.copy_(p)


# ----------------------------------------------------------------- diffusion

class Diffusion:
    def __init__(self, steps: int = 250, device: str = "cpu"):
        self.T = steps
        s = 0.008
        t = torch.linspace(0, steps, steps + 1, device=device) / steps
        ac = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ac = ac / ac[0]
        betas = (1 - ac[1:] / ac[:-1]).clamp(1e-6, 0.999)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.ac = torch.cumprod(self.alphas, 0)
        self.sqrt_ac = self.ac.sqrt()
        self.sqrt_1mac = (1 - self.ac).sqrt()

    def loss(self, model, x0):
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.sqrt_ac[t][:, None, None] * x0 + \
             self.sqrt_1mac[t][:, None, None] * noise
        return F.mse_loss(model(xt, t), noise)

    @torch.no_grad()
    def sample(self, model, n, channels, length, device, batch=512):
        outs = []
        for start in range(0, n, batch):
            b = min(batch, n - start)
            x = torch.randn(b, channels, length, device=device)
            for i in reversed(range(self.T)):
                t = torch.full((b,), i, device=device, dtype=torch.long)
                eps = model(x, t)
                # predict x0, clamp, then form the posterior mean. Clamping is
                # what keeps the trajectory from drifting off-manifold.
                x0 = (x - self.sqrt_1mac[i] * eps) / self.sqrt_ac[i]
                x0 = x0.clamp(-5, 5)
                if i > 0:
                    ac_prev = self.ac[i - 1]
                    mean = (ac_prev.sqrt() * self.betas[i] / (1 - self.ac[i]) * x0
                            + self.alphas[i].sqrt() * (1 - ac_prev)
                            / (1 - self.ac[i]) * x)
                    var = self.betas[i] * (1 - ac_prev) / (1 - self.ac[i])
                    x = mean + var.sqrt() * torch.randn_like(x)
                else:
                    x = x0
            outs.append(x)
        return torch.cat(outs, 0)


# ---------------------------------------------------------- go/no-go gate

def swiss_roll_gate(device: str = "cpu", steps: int = 4000):
    """
    Trains a tiny DDPM on a 2D swiss roll and measures how close its samples
    land to the real manifold, via nearest-neighbour distance to the training
    set, normalised by the real set's own NN distance.

        ratio ~ 1-3   samples sit on the manifold        -> PASS
        ratio >> 3    samples are off-manifold noise     -> schedule/sampler bug

    (An earlier version correlated radius against arctan2 angle. That was wrong:
    unwrapping requires an ordered sequence and these samples are unordered, so
    it could never pass. Fixed.)
    """
    import numpy as np

    th = np.random.rand(4096) * 3 * np.pi + 1.5 * np.pi
    pts = (np.stack([th * np.cos(th), th * np.sin(th)], 1) / 10.0).astype(np.float32)
    x0 = torch.tensor(pts, device=device)[:, :, None].repeat(1, 1, 16)

    model = UNet1D(channels=2, base=32).to(device)
    ema = EMA(model, 0.995)
    diff = Diffusion(250, device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for i in range(steps):
        idx = torch.randint(0, x0.shape[0], (256,), device=device)
        loss = diff.loss(model, x0[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        ema.update(model)
        if i % 1000 == 0:
            print(f"  gate step {i}  loss {loss.item():.4f}")

    gen = diff.sample(ema.shadow, 512, 2, 16, device).mean(-1).cpu().numpy()

    def nn_dist(a, b):
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        return np.median(d.min(1))

    ref = nn_dist(pts[:512], pts[512:1536])
    got = nn_dist(gen, pts[512:1536])   # same candidate set as ref, else biased
    ratio = got / (ref + 1e-12)
    ok = ratio < 3.0
    print(f"\n  GATE: NN-distance ratio = {ratio:.2f} "
          f"({'PASS' if ok else 'FAIL -- STOP, do not train'})")
    return ok


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    swiss_roll_gate(dev)
