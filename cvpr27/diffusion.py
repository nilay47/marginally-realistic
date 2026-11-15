"""
Cosine-schedule eps-prediction diffusion over (video, sensors) pairs (spec 2.4).

  loss      : MSE(video eps) + sens_w * MSE(sensor eps, readout = spatial mean)  (sens_w=1: equal weight; v2 runs use 10)
  sample    : mode="ancestral" (250 steps, final samples) or "ddim" (50 steps, checkpoint evals)
  cond_video: replacement sampling for `cascaded` -- at every reverse step the video channel is
              overwritten with a q-sample of the GIVEN video at the current noise level, and the
              returned video is the given video exactly; only the sensors are generated.
x0 is clamped each step: video to [-1,1], sensors to +-5 (standardised units, as in the workshop).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from cvpr27.models import Variant


class Diffusion:
    def __init__(self, steps: int = 250, device="cpu"):
        self.T = steps
        s = 0.008
        t = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
        ac = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ac = ac / ac[0]
        betas = (1 - ac[1:] / ac[:-1]).clamp(1e-6, 0.999)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, 0)
        f = lambda x: x.float().to(device)
        self.betas, self.alphas, self.ac = f(betas), f(alphas), f(acp)
        self.sqrt_ac, self.sqrt_1mac = self.ac.sqrt(), (1 - self.ac).sqrt()
        self.ac_prev = torch.cat([torch.ones(1, device=device), self.ac[:-1]])

    @staticmethod
    def _bc(a, t, x):
        return a[t].view(-1, *([1] * (x.dim() - 1)))

    def q_sample(self, x0, t, noise):
        return self._bc(self.sqrt_ac, t, x0) * x0 + self._bc(self.sqrt_1mac, t, x0) * noise

    # ------------------------------------------------------------------ training
    def loss(self, variant: Variant, model, v, s, sens_w: float = 1.0):
        b = (v if v is not None else s).shape[0]
        dev = (v if v is not None else s).device
        t = torch.randint(0, self.T, (b,), device=dev)
        ev = torch.randn_like(v) if variant.has_video else None
        es = torch.randn_like(s) if variant.has_sens else None
        xv = self.q_sample(v, t, ev) if variant.has_video else None
        xs = self.q_sample(s, t, es) if variant.has_sens else None
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = model(variant.pack(xv, xs), t)
        ev_h, es_h = variant.unpack(out.float())
        parts, tot = {}, 0.0
        if variant.has_video:
            parts["loss_video"] = F.mse_loss(ev_h, ev); tot = tot + parts["loss_video"]
        if variant.has_sens:
            parts["loss_sens"] = F.mse_loss(es_h, es); tot = tot + sens_w * parts["loss_sens"]
        return tot, {k: float(x.detach()) for k, x in parts.items()}

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(self, variant: Variant, model, n: int, device, mode: str = "ancestral",
               ddim_steps: int = 50, seed: int = 0, batch: int = 64, cond_video=None,
               clip_v: float = 1.0, clip_s: float = 5.0):
        """Returns (video (n,1,T,H,W) | None, sens (n,2,T) | None) as float32 CPU tensors."""
        assert mode in ("ancestral", "ddim")
        if cond_video is not None:
            assert variant.name == "joint" and cond_video.shape[0] >= n
        gen = torch.Generator(device=device).manual_seed(seed)
        rn = lambda *shape: torch.randn(*shape, device=device, generator=gen)
        T_, H, W = variant.T, variant.H, variant.W
        if mode == "ancestral":
            ts = list(range(self.T - 1, -1, -1))
        else:
            ts = sorted({int(round(x)) for x in torch.linspace(self.T - 1, 0, ddim_steps).tolist()},
                        reverse=True)
        outs_v, outs_s = [], []
        model.eval()
        for start in range(0, n, batch):
            b = min(batch, n - start)
            xv = rn(b, 1, T_, H, W) if variant.has_video else None
            xs = rn(b, 2, T_) if variant.has_sens else None
            cv = cond_video[start:start + b].to(device) if cond_video is not None else None
            for k, i in enumerate(ts):
                tt = torch.full((b,), i, device=device, dtype=torch.long)
                if cv is not None:
                    xv = self.q_sample(cv, tt, rn(*cv.shape))
                with torch.autocast(device.type if hasattr(device, "type") else str(device),
                                    dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                    out = model(variant.pack(xv, xs), tt)
                ev, es = variant.unpack(out.float())
                sa, s1 = self.sqrt_ac[i], self.sqrt_1mac[i]
                last = k == len(ts) - 1
                new = []
                for x, e, clip in ((xv, ev, clip_v), (xs, es, clip_s)):
                    if x is None:
                        new.append(None); continue
                    x0 = ((x - s1 * e) / sa).clamp(-clip, clip)
                    if last:
                        new.append(x0); continue
                    if mode == "ancestral":
                        acp, a_i, b_i, ac_i = self.ac_prev[i], self.alphas[i], self.betas[i], self.ac[i]
                        mean = (acp.sqrt() * b_i / (1 - ac_i)) * x0 + (a_i.sqrt() * (1 - acp) / (1 - ac_i)) * x
                        var = b_i * (1 - acp) / (1 - ac_i)
                        new.append(mean + var.sqrt() * rn(*x.shape))
                    else:  # DDIM, eta = 0
                        ip = ts[k + 1]
                        e2 = (x - sa * x0) / s1
                        new.append(self.sqrt_ac[ip] * x0 + self.sqrt_1mac[ip] * e2)
                xv, xs = new
            if cv is not None:
                xv = cv                          # replacement conditioning returns the given video
            if xv is not None:
                outs_v.append(xv.float().cpu())
            if xs is not None:
                outs_s.append(xs.float().cpu())
        v = torch.cat(outs_v) if outs_v else None
        s = torch.cat(outs_s) if outs_s else None
        return v, s
