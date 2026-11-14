"""
Video-quality metric: FVD (I3D) if a checkpoint can be loaded AND passes a behavioural self-check,
otherwise per-frame KID with torchvision Inception-v3 (spec section 3 fallback).

The decision is made ONCE by nb0 (`decide_metric`) and written to <root>/cache/video_metric.json
(kind, source, sha256 of the checkpoint, self-check numbers, fallback reason). Every other notebook
calls `load_metric` so training-time and final numbers always use the same metric.

FVD   : I3D features (frames upsampled to 224, repeated to 3 channels, all T frames, range [-1,1]);
        Frechet distance on per-video features.  The I3D file is the StyleGAN-V torchscript export;
        we pin it by recording its sha256 in video_metric.json (computed here, never hardcoded).
KID   : Inception-v3 pool features of 8 evenly spaced frames per clip; polynomial kernel
        (x.y/d + 1)^3; clip-level U-statistic (within-clip pairs excluded), so bootstrapping over
        generated clips is exact index gathering.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cvpr27 import physvid as pv
from cvpr27.common import atomic_json, drive_root, sha256_file

I3D_URLS = ["https://www.dropbox.com/s/ge9e5ujwgv7wxs0/i3d_torchscript.pt?dl=1"]
KID_FRAMES = 8


def _cache() -> Path:
    d = drive_root() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ extractors
def _i3d_path() -> Path | None:
    """Look for a manually placed file first, then try to download."""
    p = Path(os.environ.get("PV_I3D_PATH", _cache() / "i3d_torchscript.pt"))
    if p.exists() and p.stat().st_size > 10_000_000:
        return p
    for url in I3D_URLS:
        try:
            tmp = p.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size > 10_000_000:      # an HTML interstitial is ~100 KB
                tmp.rename(p)
                return p
            tmp.unlink()
        except Exception:
            pass
    return None


class I3D:
    def __init__(self, path, device):
        self.m = torch.jit.load(str(path), map_location=device).eval()
        self.device = device

    @torch.no_grad()
    def features(self, v_u8, batch=16):
        out = []
        for s in range(0, len(v_u8), batch):
            x = torch.from_numpy(v_u8[s:s + batch]).to(self.device).float() / 127.5 - 1.0   # (b,T,H,W)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = x[:, None].expand(-1, 3, -1, -1, -1)                                          # (b,3,T,224,224)
            out.append(self.m(x.contiguous(), rescale=False, resize=False, return_features=True).float().cpu())
        return torch.cat(out).numpy().astype(np.float64)


class PVI3D:
    """Second I3D source: pytorchvideo's I3D-R50 (Kinetics-400, official weights from dl.fbaipublicfiles.com).
    Features = global mean over (T,H,W) of the last conv stage (2048-d). NOT the Inception-I3D used by the
    published FVD protocol, so values are not comparable to published FVD numbers; flagged in video_metric.json."""

    def __init__(self, device):
        from pytorchvideo.models.hub import i3d_r50
        self.m, self.device = i3d_r50(pretrained=True).eval().to(device), device
        self.blocks = self.m.blocks[:-1]

    @torch.no_grad()
    def features(self, v_u8, batch=8):
        out = []
        for s in range(0, len(v_u8), batch):
            x = torch.from_numpy(v_u8[s:s + batch]).to(self.device).float() / 255.0
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = (x[:, None].expand(-1, 3, -1, -1, -1) - 0.45) / 0.225          # Kinetics normalisation
            for b in self.blocks:
                x = b(x)
            out.append(x.mean((2, 3, 4)).float().cpu())
        return torch.cat(out).numpy().astype(np.float64)


class Inception:
    def __init__(self, device):
        import torchvision
        m = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1,
                                            aux_logits=True, transform_input=False)
        m.fc = torch.nn.Identity()
        self.m, self.device = m.to(device).eval(), device
        self.mu = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.sd = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def features(self, v_u8, batch=None):
        batch = batch or (256 if str(self.device).startswith("cuda") else 16)   # CPU: small batch, 299x299 activations are large
        idx = np.linspace(0, pv.T - 1, KID_FRAMES).round().astype(int)
        fr = v_u8[:, idx]                                    # (n, 8, H, W)
        n = fr.shape[0]
        flat = fr.reshape(n * KID_FRAMES, pv.H, pv.W)
        out = []
        for s in range(0, len(flat), batch):
            x = torch.from_numpy(flat[s:s + batch]).to(self.device).float() / 255.0
            x = F.interpolate(x[:, None], size=(299, 299), mode="bilinear", align_corners=False)
            x = (x.expand(-1, 3, -1, -1) - self.mu) / self.sd
            out.append(self.m(x).float().cpu())
        return torch.cat(out).numpy().reshape(n, KID_FRAMES, -1)


# ------------------------------------------------------------------ metric
def _fd(mu1, s1, mu2, s2):
    """Frechet distance; tr sqrt(S1 S2) via eigenvalues of S1 S2 (real, >= 0 up to noise)."""
    ev = np.linalg.eigvals(s1 @ s2)
    tr = np.sqrt(np.abs(ev.real)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr)


def _fd_sym(mu_r, cov_r, sqrt_r, mu_g, cov_g, device):
    """Same Frechet distance via the symmetric form tr sqrt(R^1/2 G R^1/2); eigvalsh (GPU if available) is much
    faster than eigvals of the non-symmetric product, which matters for bootstrap/E6 loops on 2048-d features."""
    M = sqrt_r @ cov_g @ sqrt_r
    M = 0.5 * (M + M.T)
    if str(device).startswith("cuda"):
        ev = torch.linalg.eigvalsh(torch.from_numpy(M).to(device)).cpu().numpy()
    else:
        ev = np.linalg.eigvalsh(M)
    return float(((mu_r - mu_g) ** 2).sum() + np.trace(cov_r) + np.trace(cov_g) - 2 * np.sqrt(np.clip(ev, 0, None)).sum())


class VideoMetric:
    def __init__(self, kind: str, device, i3d_path=None, flavor: str | None = None):
        self.kind, self.device, self.flavor = kind, device, flavor
        self.name = "FVD" if kind == "fvd" else "KID"
        if kind == "fvd":
            self.ext = I3D(i3d_path, device) if flavor == "torchscript" else PVI3D(device)
        else:
            self.ext = Inception(device)
        self.ref = None

    # ---- reference
    def set_reference(self, real_u8):
        key = _cache() / f"vidfeat_{self.kind}{self.flavor or ''}_heldout{len(real_u8)}.npy"
        if key.exists():
            f = np.load(key)
        else:
            f = self.ext.features(real_u8)
            tmp = key.with_name(key.stem + ".tmp.npy")
            np.save(tmp, f)
            tmp.rename(key)
        self._install_ref(f)

    def set_reference_nocache(self, real_u8):
        """Reference from an arbitrary clip set (e.g. a held-out half); never touches the on-disk feature cache."""
        self._install_ref(self.ext.features(real_u8))

    def _install_ref(self, f):
        self.ref = {"feats": f}
        if self.kind == "fvd":
            self.ref["mu"], self.ref["cov"] = f.mean(0), np.cov(f, rowvar=False)
            w, V = np.linalg.eigh(self.ref["cov"])
            self.ref["sqrt_cov"] = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
        else:
            self.ref["rr"] = self._kid_rr(f)

    # ---- KID helpers (clip-level kernel block means)
    def _blocks(self, fa, fb):
        """A[i,j] = mean over frame pairs of k(fa[i,f], fb[j,g]); (na, nb) float32 on CPU."""
        dev, D = self.device, fa.shape[-1]
        B = torch.from_numpy(fb).to(dev)
        rows = []
        for s in range(0, len(fa), 32):
            a = torch.from_numpy(fa[s:s + 32]).to(dev)
            k = (torch.einsum("ifd,jgd->ijfg", a, B) / D + 1.0) ** 3
            rows.append(k.mean((-1, -2)).cpu())
        return torch.cat(rows).numpy()

    def _kid_rr(self, f):
        A = self._blocks(f, f)
        n = len(f)
        return float((A.sum() - np.trace(A)) / (n * (n - 1)))

    # ---- generated features / value
    def prep(self, gen_u8):
        f = self.ext.features(gen_u8)
        if self.kind == "fvd":
            return {"feats": f}
        return {"feats": f, "gg": self._blocks(f, f), "gr": self._blocks(f, self.ref["feats"])}

    def value(self, p, idx=None):
        n = len(p["feats"])
        idx = np.arange(n) if idx is None else np.asarray(idx)
        if self.kind == "fvd":
            f = p["feats"][idx]
            return _fd_sym(self.ref["mu"], self.ref["cov"], self.ref["sqrt_cov"], f.mean(0),
                           np.cov(f, rowvar=False), self.device)
        gg = p["gg"][np.ix_(idx, idx)]
        mask = idx[:, None] != idx[None, :]
        m = len(idx)
        return float(gg[mask].sum() / mask.sum() + self.ref["rr"] - 2 * p["gr"][idx].mean())

    def score(self, gen_u8):
        return self.value(self.prep(gen_u8))


# ------------------------------------------------------------------ decision (nb0) / loading
def _self_check(m: "VideoMetric", real_u8) -> dict:
    """Behavioural gate: a usable video metric must separate real-vs-real from real-vs-corrupted."""
    n = len(real_u8) // 2
    a, b = real_u8[:n], real_u8[n:2 * n]
    noisy = np.clip(b.astype(np.int16) + np.random.default_rng(0).normal(0, 25, b.shape), 0, 255).astype(np.uint8)
    rev = b[:, ::-1].copy()                                              # time-reversed clips
    fa, fb, fn, fr = (m.ext.features(x) for x in (a, b, noisy, rev))
    d = lambda x, y: _fd(x.mean(0), np.cov(x, rowvar=False), y.mean(0), np.cov(y, rowvar=False))
    chk = {"real_vs_real": d(fa, fb), "real_vs_noisy": d(fa, fn), "real_vs_time_reversed": d(fa, fr),
           "feature_dim": int(fa.shape[1]), "n_per_set": int(n)}
    chk["ok"] = bool(np.isfinite([chk["real_vs_real"], chk["real_vs_noisy"]]).all()
                     and chk["real_vs_noisy"] > 5 * chk["real_vs_real"])
    return chk


def decide_metric(real_u8, device, force_kind: str | None = None) -> dict:
    """FVD source order: (1) StyleGAN-V I3D torchscript (the original FVD network), (2) pytorchvideo I3D-R50,
    (3) per-frame KID. First source that loads AND passes the self-check wins. Writes video_metric.json once."""
    path = _cache() / "video_metric.json"
    if path.exists():
        return json.loads(path.read_text())
    info = {"kind": "kid", "flavor": None, "attempts": []}
    if force_kind != "kid":
        for flavor in ("torchscript", "pytorchvideo"):
            att = {"flavor": flavor}
            try:
                if flavor == "torchscript":
                    p = _i3d_path()
                    if p is None:
                        raise RuntimeError(f"checkpoint not downloadable from {I3D_URLS} and not found at "
                                           f"{_cache() / 'i3d_torchscript.pt'}")
                    m = VideoMetric("fvd", device, p, flavor)
                    att.update(path=str(p), sha256=sha256_file(p), url=I3D_URLS[0])
                else:
                    import importlib.metadata as md
                    m = VideoMetric("fvd", device, None, flavor)
                    ck = Path(torch.hub.get_dir()) / "checkpoints" / "I3D_8x8_R50.pyth"
                    att.update(pytorchvideo=md.version("pytorchvideo"), weights="I3D_8x8_R50.pyth (Kinetics-400)",
                               sha256=sha256_file(ck) if ck.exists() else None,
                               comparable_to_published_fvd=False)
                att["check"] = _self_check(m, real_u8)
                att["ok"] = att["check"]["ok"]
            except Exception as e:                                       # noqa: BLE001
                att.update(ok=False, reason=f"{type(e).__name__}: {e}")
            info["attempts"].append(att)
            if att["ok"]:
                info.update(kind="fvd", flavor=flavor, i3d_path=att.get("path"))
                break
    info["fallback_to_kid"] = info["kind"] == "kid"
    if info["kind"] == "kid":
        import torchvision
        info["kid"] = {"frames_per_clip": KID_FRAMES, "kernel": "poly3 (x.y/d+1)^3",
                       "net": "torchvision inception_v3 IMAGENET1K_V1", "torchvision": torchvision.__version__}
    atomic_json(path, info)
    return info


def load_metric(device) -> VideoMetric:
    path = _cache() / "video_metric.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run nb0 first (it decides FVD vs KID once for all runs)")
    info = json.loads(path.read_text())
    return VideoMetric(info["kind"], device, info.get("i3d_path"), info.get("flavor"))
