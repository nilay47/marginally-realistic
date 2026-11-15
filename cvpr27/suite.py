"""
Shared evaluation suite: used for training-time checkpoint evals and for nb4.
`Reference` holds the held-out real set (2,048 clips) and everything derived from it once.
All inputs to `Reference.evaluate` are PHYSICAL (video uint8, gyro/torque in physical units).
"""
from __future__ import annotations

import numpy as np

from cvpr27 import evalmetrics as em
from cvpr27 import physvid as pv


class Reference:
    def __init__(self, heldout: dict, norm, metric=None, device="cpu"):
        self.v, self.om, self.tau, self.norm, self.metric, self.device = (
            heldout["video"], heldout["omega"].astype(np.float64), heldout["tau"].astype(np.float64),
            norm, metric, device)
        self.n = len(self.v)
        self.per = em.score_clips(self.v, self.om, self.tau)        # floors, pixels -> theta, uint8
        self.floor = em.summarise(self.per)
        self.mmd_feats = em.mmd_features(self.v, self.om, self.tau, norm)
        self._mmd_cache = None
        if metric is not None:
            metric.set_reference(self.v)

    def mmd_pool(self, gen_feats):
        pool = em.MMDPool(self.mmd_feats, gen_feats, self.device, self._mmd_cache)
        self._mmd_cache = pool.real_cache()
        return pool

    def evaluate(self, v_u8=None, om=None, tau=None, mmd=True, use_metric=True) -> dict:
        """Any subset of modalities. Returns a flat JSON-able dict; every number computed here."""
        out = {}
        if om is not None:
            out.update(em.sensor_marginals(self.om, self.tau, om, tau))
            for k in ("w1_mean", "psd_mean"):
                em.check_finite(k, out[k])
        if v_u8 is not None:
            if self.metric is not None and use_metric:
                out["video_metric"] = {"name": self.metric.name, "value": float(self.metric.score(v_u8)),
                                       "n": int(len(v_u8))}
                em.check_finite("video_metric", out["video_metric"]["value"])
            _, ok = pv.extract_theta(v_u8)
            out["invalid_frame_frac"] = float((~ok).mean())
            out["invalid_clip_frac"] = float((~ok).any(1).mean())
        if v_u8 is not None and om is not None:
            per = em.score_clips(v_u8, om, tau)
            out["physical"] = em.summarise(per)
            if mmd:
                n = len(v_u8)
                m = min(n, self.n // 2)
                pool = self.mmd_pool(em.mmd_features(v_u8, om, tau, self.norm))
                out["mmd"] = {"value": pool.mmd2(np.arange(n), np.arange(m)),
                              "floor_real_vs_heldout_real": pool.floor(np.arange(m), np.arange(m, 2 * m)),
                              "n_gen": int(n), "n_real": int(m), "gamma": pool.gamma}
        return out
