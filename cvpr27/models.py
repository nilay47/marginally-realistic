"""
(2+1)D U-Net (spec 2.3), one architecture for every variant.

Tensor layout (B, C, T, H, W).
  joint        C_in = 3  (video, gyro plane, torque plane)   planes 32x32
  video_only   C_in = 1                                       planes 32x32
  sensor_only  C_in = 2  (gyro, torque)                       planes 4x4, time-only downsampling

Sensors enter as constant planes. Sensor readout = spatial mean of the sensor output channels.

ResBlock: [GN -> SiLU -> conv(1,3,3) -> conv(3,1,1)] twice, timestep embedding added after the
first pair. Encoder: 2 blocks/level; decoder: 3 blocks/level (2 + 1, each consuming a skip), the
standard improved-DDPM layout. Mults (1,2,2,4): 3 down-samples, stride (2,2,2) (or (2,1,1) for
sensor_only) so T 64->32->16->8 and, for video, space 32->16->8->4. Bottleneck = ResBlock,
full self-attention over the 8x4x4 tokens (learned positional embedding added), ResBlock.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

GROUPS = 8


class TimeEmb(nn.Module):
    def __init__(self, base: int, tdim: int):
        super().__init__()
        self.base = base
        self.mlp = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))

    def forward(self, t):
        half = self.base // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        a = t[:, None].float() * f[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(GROUPS, cin), nn.GroupNorm(GROUPS, cout)
        self.s1 = nn.Conv3d(cin, cout, (1, 3, 3), padding=(0, 1, 1))
        self.t1 = nn.Conv3d(cout, cout, (3, 1, 1), padding=(1, 0, 0))
        self.s2 = nn.Conv3d(cout, cout, (1, 3, 3), padding=(0, 1, 1))
        self.t2 = nn.Conv3d(cout, cout, (3, 1, 1), padding=(1, 0, 0))
        self.temb = nn.Linear(tdim, cout)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.t1(self.s1(F.silu(self.n1(x))))
        h = h + self.temb(F.silu(e))[:, :, None, None, None]
        h = self.t2(self.s2(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attn3D(nn.Module):
    """Full self-attention over all T*H*W tokens of the (low-res) bottleneck."""

    def __init__(self, c: int, tokens: tuple[int, int, int], heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(GROUPS, c)
        self.qkv = nn.Linear(c, 3 * c)
        self.proj = nn.Linear(c, c)
        n = tokens[0] * tokens[1] * tokens[2]
        self.pos = nn.Parameter(torch.zeros(1, n, c))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        b, c, t, h, w = x.shape
        z = self.norm(x).flatten(2).transpose(1, 2) + self.pos          # (B, N, C)
        q, k, v = self.qkv(z).chunk(3, -1)
        sp = lambda a: a.view(b, -1, self.heads, c // self.heads).transpose(1, 2)
        o = F.scaled_dot_product_attention(sp(q), sp(k), sp(v))
        o = self.proj(o.transpose(1, 2).reshape(b, -1, c))
        return x + o.transpose(1, 2).reshape(b, c, t, h, w)


class Down(nn.Module):
    def __init__(self, c, stride):
        super().__init__()
        self.conv = nn.Conv3d(c, c, 3, stride=stride, padding=1)

    def forward(self, x, e=None):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, c, scale):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv3d(c, c, 3, padding=1)

    def forward(self, x, e=None):
        return self.conv(F.interpolate(x, scale_factor=self.scale, mode="nearest"))


class UNet3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, T: int, H: int, W: int, base: int = 64,
                 mults=(1, 2, 2, 4), nblocks: int = 2, space_down: bool = True, heads: int = 4):
        super().__init__()
        tdim = base * 4
        self.temb = TimeEmb(base, tdim)
        self.stem = nn.Conv3d(in_ch, base, 3, padding=1)
        stride = (2, 2, 2) if space_down else (2, 1, 1)
        scale = stride
        nd = len(mults) - 1
        ft, fh, fw = 2 ** nd, (2 ** nd if space_down else 1), (2 ** nd if space_down else 1)
        assert T % ft == 0 and H % fh == 0 and W % fw == 0
        tokens = (T // ft, H // fh, W // fw)

        self.down = nn.ModuleList()
        chs, cur = [base], base
        for i, m in enumerate(mults):
            for _ in range(nblocks):
                self.down.append(ResBlock(cur, base * m, tdim))
                cur = base * m
                chs.append(cur)
            if i < nd:
                self.down.append(Down(cur, stride))
                chs.append(cur)
        self.mid1, self.attn, self.mid2 = (ResBlock(cur, cur, tdim), Attn3D(cur, tokens, heads),
                                           ResBlock(cur, cur, tdim))
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for j in range(nblocks + 1):
                self.up.append(ResBlock(cur + chs.pop(), base * m, tdim))
                cur = base * m
                if i > 0 and j == nblocks:
                    self.up.append(Up(cur, scale))
        self.out_norm = nn.GroupNorm(GROUPS, cur)
        self.out = nn.Conv3d(cur, out_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.tokens = tokens

    def forward(self, x, t):
        e = self.temb(t)
        h = self.stem(x)
        skips = [h]
        for m in self.down:
            h = m(h, e)
            skips.append(h)
        h = self.mid2(self.attn(self.mid1(h, e)), e)
        for m in self.up:
            if isinstance(m, ResBlock):
                h = m(torch.cat([h, skips.pop()], 1), e)
            else:
                h = m(h)
        return self.out(F.silu(self.out_norm(h)))


# ------------------------------------------------------------------ variants
class Variant:
    """Packs (video, sensors) <-> network tensors for one model variant."""

    def __init__(self, name: str, T: int, H: int, W: int):
        assert name in ("joint", "video_only", "sensor_only"), name
        self.name, self.T, self.H, self.W = name, T, H, W
        self.has_video = name in ("joint", "video_only")
        self.has_sens = name in ("joint", "sensor_only")
        self.in_ch = {"joint": 3, "video_only": 1, "sensor_only": 2}[name]
        self.plane = (H, W) if self.has_video else (4, 4)

    def build(self, base: int = 64) -> UNet3D:
        return UNet3D(self.in_ch, self.in_ch, self.T, self.plane[0], self.plane[1], base=base,
                      space_down=self.has_video)

    def pack(self, v, s):
        """v (B,1,T,H,W) or None; s (B,2,T) or None -> (B, C, T, h, w)."""
        parts = []
        if self.has_video:
            parts.append(v)
        if self.has_sens:
            parts.append(s[..., None, None].expand(-1, -1, -1, *self.plane))
        return torch.cat(parts, 1)

    def unpack(self, out):
        """-> (eps_video or None, eps_sens or None). Sensor readout = spatial mean of its channels."""
        ev = out[:, :1] if self.has_video else None
        es = out[:, 1:] if self.name == "joint" else out
        es = es.mean((-1, -2)) if self.has_sens else None
        return ev, es


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
