"""
nb4 backend: python -m cvpr27.eval --tag main

3 draws of 1,024 samples per variant, full 250-step ancestral sampling (draw 0 = main, draws 1-2 = redraws).
Main draw (0): point estimates + 95% percentile bootstrap CIs (B=200) over generated samples.
All draws: spread (mean / std / min / max) per metric.

Variants: joint, video_only, sensor_only, independent (video_only video + sensor_only sensors),
cascaded (video_only video, sensors in-painted by the joint model via replacement sampling),
shuffle_oracle (real held-out video + sensors of a different real clip).

Experiments written to results.json: E1 (from nb0), E2/E3 (joint_s0 checkpoint curve), E4 (variants), E5 (oracle),
E6 (samples needed to separate variants), E7 (per-sample detection AUROC), plus the spec section-6 go/no-go flags
computed from the numbers. Every number is computed in this run (Rule 4). Outputs are new files only (Rule 3).
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from cvpr27 import benchmark as bm
from cvpr27 import evalmetrics as em
from cvpr27 import physvid as pv
from cvpr27.common import atomic_json, drive_root, fmt_s, git_info, new_path
from cvpr27.data import Norm, load_split
from cvpr27.suite import Reference
from cvpr27.variants import load_ema, sample_job

PHYS = ["r_energy", "r_energy_win_max", "kin_rad_max"]
VARIANTS = ["joint", "independent", "cascaded", "shuffle_oracle", "video_only", "sensor_only"]


def pct(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


class VarData:
    """Everything needed to evaluate + bootstrap one (variant, draw)."""

    def __init__(self, name, vid, om, tau):
        self.name, self.vid, self.om, self.tau = name, vid, om, tau
        self.n = len(vid if vid is not None else om)


def evaluate_one(ref: Reference, vd: VarData, prep_cache: dict, key, want_arrays: bool):
    rec, arrays = {"n": vd.n}, {}
    if vd.om is not None:
        rec.update(em.sensor_marginals(ref.om, ref.tau, vd.om, vd.tau))
    if vd.vid is not None:
        _, ok = pv.extract_theta(vd.vid)
        rec["invalid_frame_frac"] = float((~ok).mean())
        rec["invalid_clip_frac"] = float((~ok).any(1).mean())
        if ref.metric is not None:
            if key not in prep_cache:
                prep_cache[key] = ref.metric.prep(vd.vid)
            rec["video_metric"] = float(ref.metric.value(prep_cache[key]))
            em.check_finite("video_metric", rec["video_metric"])
            arrays["prep"] = prep_cache[key]
    if vd.vid is not None and vd.om is not None:
        per = em.score_clips(vd.vid, vd.om, vd.tau)
        rec["physical"] = em.summarise(per)
        pool = ref.mmd_pool(em.mmd_features(vd.vid, vd.om, vd.tau, ref.norm))
        m = min(vd.n, ref.n // 2)
        rec["mmd"] = {"value": pool.mmd2(np.arange(vd.n), np.arange(m)),
                      "floor_real_vs_heldout_real": pool.floor(np.arange(m), np.arange(m, 2 * m))}
        arrays.update(per=per, pool=pool, m=m)
    return rec, arrays


def scalars(rec) -> dict:
    """Flat dict of the headline scalars (for spread tables)."""
    out = {k: rec[k] for k in ("w1_mean", "psd_mean", "video_metric", "invalid_frame_frac") if k in rec}
    if "physical" in rec:
        for k in PHYS + ["r_energy_win_med", "kin_rad_med"]:
            out[k] = rec["physical"][k]["median"]
        out["ratio_win_over_glob"] = rec["physical"]["ratio_of_medians_win_over_glob"]
        out["mmd"] = rec["mmd"]["value"]
    return out


def bootstrap(ref, vd: VarData, arrays: dict, B: int, seed: int, m_sub: int, rec: dict) -> dict:
    """Per-sample metrics (W1, physics medians): percentile bootstrap. Set-level metrics (video metric, MMD): subsampling
    without replacement (em.subsample_stat), m_sub from the generated set and m_sub from the reference set."""
    rng = np.random.default_rng(seed)
    n = vd.n
    stats = {}
    idxs = [rng.integers(0, n, n) for _ in range(B)]

    def run(name, fn):
        v = np.array([fn(i) for i in idxs])
        stats[name] = pct(v)

    if vd.om is not None:
        run("w1_mean", lambda i: em.sensor_w1(ref.om, ref.tau, vd.om[i], vd.tau[i])["w1_mean"])
        run("psd_mean", lambda i: em.sensor_psd(ref.om, ref.tau, vd.om[i], vd.tau[i])["psd_mean"])
    if "prep" in arrays:
        n_ref = len(ref.metric.ref["feats"])
        stats["video_metric"] = em.subsample_stat(lambda gi, ri: ref.metric.value_pair(arrays["prep"], gi, ri), n, n_ref,
                                                  min(m_sub, n, n_ref), B, rng, rec["video_metric"])
    if "per" in arrays:
        per = arrays["per"]
        for k in PHYS + ["r_energy_win_med", "kin_rad_med"]:
            run(k, lambda i, k=k: np.median(per[k][i]))
        run("ratio_win_over_glob", lambda i: np.median(per["r_energy_win_max"][i]) / max(np.median(per["r_energy"][i]), 1e-12))
        stats["mmd"] = em.subsample_stat(lambda gi, ri: arrays["pool"].mmd2(gi, ri), n, ref.n, min(m_sub, n, ref.n), B, rng,
                                         rec["mmd"]["value"])
    return stats


# ------------------------------------------------------------------ E6 pooling over draws
class Pooled:
    def __init__(self, om, tau, n):
        self.om, self.tau, self.n = om, tau, n


def pool_draws(ref, norm, draws, arrs, names):
    """Pool every available draw of each variant (E6 subsamples WITHOUT replacement from the pool, so n=1024 still varies
    between subsets). Returns (data, arrays, info); uses whatever draws exist for a variant and records how many."""
    data, arr, info = {}, {}, {}
    for name in names:
        ds = [d for d in range(len(draws)) if name in draws[d] and "per" in arrs[d][name]]
        vds = [draws[d][name] for d in ds]
        om, tau = np.concatenate([v.om for v in vds]), np.concatenate([v.tau for v in vds])
        per = {k: np.concatenate([arrs[d][name]["per"][k] for d in ds]) for k in arrs[ds[0]][name]["per"]}
        feats = np.concatenate([em.mmd_features(v.vid, v.om, v.tau, norm) for v in vds])
        a = {"per": per, "pool": ref.mmd_pool(feats), "m": min(len(om), ref.n // 2)}
        if ref.metric is not None and all("prep" in arrs[d][name] for d in ds):
            f = np.concatenate([arrs[d][name]["prep"]["feats"] for d in ds])
            a["prep"] = {"feats": f} if ref.metric.kind == "fvd" else \
                {"feats": f, "gg": ref.metric._blocks(f, f), "gr": ref.metric._blocks(f, ref.metric.ref["feats"])}
        data[name], arr[name] = Pooled(om, tau, len(om)), a
        info[name] = {"draws_used": ds, "n_pooled": int(len(om))}
    return data, arr, info


# ------------------------------------------------------------------ E6: samples needed
def e6(ref, data0: dict, arr0: dict, sizes, R: int, R_slow: int, seed: int) -> dict:
    """P(metric ranks the worse variant worse) when computed on n samples, over resamples. 0.5 = chance."""
    rng = np.random.default_rng(seed)
    out = {}
    for worse, better in (("independent", "joint"), ("cascaded", "joint")):
        res = {}
        for n in sizes:
            if n > data0[better].n:
                continue
            gi = lambda: (rng.choice(data0[worse].n, n, replace=False), rng.choice(data0[better].n, n, replace=False))
            cell = {}
            for k in PHYS:
                wins = []
                for _ in range(R):
                    a, b = gi()
                    wins.append(np.median(arr0[worse]["per"][k][a]) > np.median(arr0[better]["per"][k][b]))
                cell[k] = float(np.mean(wins))
            wins = []
            for _ in range(R):
                a, b = gi()
                wins.append(em.sensor_w1(ref.om, ref.tau, data0[worse].om[a], data0[worse].tau[a])["w1_mean"] >
                            em.sensor_w1(ref.om, ref.tau, data0[better].om[b], data0[better].tau[b])["w1_mean"])
            cell["w1_mean"] = float(np.mean(wins))
            wins = []
            for _ in range(R):
                a, b = gi()
                m = arr0[worse]["m"]
                r1, r2 = rng.choice(m, n, replace=False), rng.choice(m, n, replace=False)
                wins.append(arr0[worse]["pool"].mmd2(a, r1) > arr0[better]["pool"].mmd2(b, r2))
            cell["mmd"] = float(np.mean(wins))
            if "prep" in arr0[worse] and "prep" in arr0[better]:
                wins = []
                for _ in range(R_slow):
                    a, b = gi()
                    wins.append(ref.metric.value(arr0[worse]["prep"], a) > ref.metric.value(arr0[better]["prep"], b))
                cell["video_metric"] = float(np.mean(wins))
            res[str(n)] = cell
        needed = {}
        for k in (PHYS + ["w1_mean", "mmd"] + (["video_metric"] if any("video_metric" in c for c in res.values()) else [])):
            ok = [int(n) for n, c in res.items() if k in c and c[k] >= 0.95]
            needed[k] = min(ok) if ok else None
        out[f"{worse}_vs_{better}"] = {"p_correct_by_n": res, "min_n_for_p95": needed, "R": R, "R_video_metric": R_slow}
    return out


# ------------------------------------------------------------------ E7: per-sample detection
def corrupted_reals(ref, rng):
    vid, om, tau = ref.v, ref.om, ref.tau
    perm = rng.permutation(len(vid))
    lag = np.concatenate([np.repeat(vid[:, :1], 3, 1), vid[:, :-3]], 1)
    return {"mispaired": (vid[perm], om, tau), "lag3": (lag, om, tau),
            "orth_torque": (vid, om, pv.orthogonal_torque_corruption(om, tau, rng))}


def e7(ref, data0, arr0, seed):
    rng = np.random.default_rng(seed)
    keys = PHYS + ["r_energy_win_med", "kin_rad_med"]
    neg = ref.per
    pos = {k: em.score_clips(*v) for k, v in corrupted_reals(ref, rng).items()}
    for name in ("joint", "independent", "cascaded"):
        pos[f"generated_{name}"] = arr0[name]["per"]
    return {p: {k: em.auroc(neg[k], per[k]) for k in keys} for p, per in pos.items()}


# ------------------------------------------------------------------ main
def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--joint_run", default="joint_s0")
    ap.add_argument("--video_run", default="video_only_s0")
    ap.add_argument("--sensor_run", default="sensor_only_s0")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--sample_batch", type=int, default=128)
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--R_slow", type=int, default=30)
    ap.add_argument("--m_sub", type=int, default=512, help="subset size for set-level metrics (FVD, MMD): subsampling without replacement")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.n, a.B, a.R, a.R_slow, a.sample_batch, a.sizes, a.m_sub = 16, 10, 8, 4, 16, [4, 8, 16], 8
    return a


def main(argv=None):
    a = get_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = drive_root() / f"eval_{a.tag}"
    if (out_dir / "results.json").exists():
        print(f"{out_dir}/results.json exists: use a new --tag (nothing is overwritten)."); return
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    models = {k: load_ema(r, dev) for k, r in (("joint", a.joint_run), ("video_only", a.video_run), ("sensor_only", a.sensor_run))}
    stats = {m["norm_stats"].__repr__() for m in models.values()}
    assert len(stats) == 1, "runs were trained with different normalisation stats"
    norm = Norm(models["joint"]["norm_stats"])
    held = load_split("heldout")
    metric = None
    if not a.smoke:
        from cvpr27.videofeat import load_metric
        metric = load_metric(dev)
    ref = Reference(held, norm, metric, dev)
    print(f"reference: {ref.n} held-out clips | video metric: {metric.name + '/' + str(metric.flavor) if metric else 'none (smoke)'}")

    # ---------------- sampling (resumable, never overwrites)
    draws, rng_o = [], np.random.default_rng(12345)
    for d in range(a.draws):
        print(f"draw {d}:")
        base = 100_000 + 1000 * d
        kw = dict(n=a.n, dev=dev, tag=a.tag, norm=norm, batch=a.sample_batch)
        vo = sample_job("video_only", d, models["video_only"], seed=base + 1, **kw)
        so = sample_job("sensor_only", d, models["sensor_only"], seed=base + 2, **kw)
        jt = sample_job("joint", d, models["joint"], seed=base + 3, **kw)
        cs = sample_job("cascaded", d, models["joint"], seed=base + 4, cond_u8=vo["video"], **kw)
        assert np.array_equal(cs["video"], vo["video"]), "cascaded must return the given video exactly"
        idx = rng_o.permutation(ref.n)[:a.n]
        j = (idx + rng_o.integers(1, ref.n)) % ref.n                  # a different real clip, never the same one
        draws.append({
            "joint": VarData("joint", jt["video"], jt["omega"].astype(np.float64), jt["tau"].astype(np.float64)),
            "independent": VarData("independent", vo["video"], so["omega"].astype(np.float64), so["tau"].astype(np.float64)),
            "cascaded": VarData("cascaded", cs["video"], cs["omega"].astype(np.float64), cs["tau"].astype(np.float64)),
            "shuffle_oracle": VarData("shuffle_oracle", ref.v[idx], ref.om[j], ref.tau[j]),
            "video_only": VarData("video_only", vo["video"], None, None),
            "sensor_only": VarData("sensor_only", None, so["omega"].astype(np.float64), so["tau"].astype(np.float64))})
        print(f"  elapsed {fmt_s(time.time() - t0)}")

    # ---------------- metrics per (variant, draw)
    prep_cache, recs, arr0, arrs = {}, [], {}, []
    for d, dd in enumerate(draws):
        rec_d, arrs_d = {}, {}
        for name in VARIANTS:
            vd = dd[name]
            vsrc = {"joint": ("joint", d), "independent": ("vo", d), "cascaded": ("vo", d), "video_only": ("vo", d),
                    "shuffle_oracle": ("oracle", d)}.get(name)
            rec, arrays = evaluate_one(ref, vd, prep_cache, vsrc, want_arrays=d == 0)
            rec_d[name] = rec
            arrs_d[name] = arrays
            if d == 0:
                arr0[name] = arrays
        recs.append(rec_d)
        arrs.append(arrs_d)
        print(f"draw {d} evaluated ({fmt_s(time.time() - t0)})")

    # ---------------- bootstrap (main draw) + floors
    boot = {name: bootstrap(ref, draws[0][name], arr0[name], a.B, 7 + i, a.m_sub, recs[0][name]) for i, name in enumerate(VARIANTS)}
    rngf = np.random.default_rng(99)
    fl = {k: [float(np.median(ref.per[k]))] + pct([np.median(ref.per[k][rngf.integers(0, ref.n, ref.n)]) for _ in range(a.B)])
          for k in PHYS}
    floors = {"median_and_ci": fl, "summary": ref.floor, "n": ref.n}
    print(f"bootstrap done ({fmt_s(time.time() - t0)})")

    # ---------------- spread across draws
    spread = {}
    for name in VARIANTS:
        per_draw = [scalars(r[name]) for r in recs]
        spread[name] = {k: {"values": [p[k] for p in per_draw], "mean": float(np.mean([p[k] for p in per_draw])),
                            "std": float(np.std([p[k] for p in per_draw], ddof=1)) if len(per_draw) > 1 else None,
                            "min": float(np.min([p[k] for p in per_draw])), "max": float(np.max([p[k] for p in per_draw]))}
                        for k in per_draw[0]}

    # ---------------- E4: gaps
    def ratio(num, den, k):
        return [r[num]["physical"][k]["median"] / max(r[den]["physical"][k]["median"], 1e-12) for r in recs]
    gaps = {}
    for w in ("independent", "cascaded"):
        gaps[f"{w}_over_joint"] = {k: {"ratio_by_draw": ratio(w, "joint", k),
                                       "ci_disjoint_main": boot[w][k][0] > boot["joint"][k][1],
                                       "ratio_ci_bounds_main": boot[w][k][0] / max(boot["joint"][k][1], 1e-12)} for k in PHYS}

    # ---------------- E2/E3: joint training curve (read from the training run's files)
    jr = drive_root() / a.joint_run
    curve = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(jr / "metrics_0*.json")))]
    e3 = {"real_ratio_of_medians": ref.floor["ratio_of_medians_win_over_glob"],
          "real_median_of_ratios": ref.floor["win_over_glob"]["median"],
          "by_step": [{"step": c["step"], "n": c["n"],
                       "ratio_of_medians": c["metrics"]["physical"]["ratio_of_medians_win_over_glob"],
                       "median_of_ratios": c["metrics"]["physical"]["win_over_glob"]["median"],
                       "r_energy": c["metrics"]["physical"]["r_energy"]["median"],
                       "r_energy_win_max": c["metrics"]["physical"]["r_energy_win_max"]["median"]} for c in curve if "physical" in c["metrics"]]}

    # ---------------- E6, E7
    pdata, parr, e6_pool = pool_draws(ref, norm, draws, arrs, ["joint", "independent", "cascaded"])
    e6res = e6(ref, pdata, parr, a.sizes, a.R, a.R_slow, seed=5)
    print(f"E6 done ({fmt_s(time.time() - t0)})")
    e7res = e7(ref, draws[0], arr0, seed=6)

    # ---------------- E8: consistent time reversal of REAL clips (real data only; reference of the metric is restored)
    e8res = bm.e8_time_reversal(held, norm, metric, B=a.B, device=dev, n=64 if a.smoke else 1024,
                                m_sub=a.m_sub)
    print(f"E8 done ({fmt_s(time.time() - t0)})")

    # ---------------- go / no-go flags (spec section 6), computed
    j = recs[0]["joint"]["physical"]
    near = {k: j[k]["median"] / max(ref.floor[k]["median"], 1e-12) for k in PHYS}
    gap10 = {k: bool(gaps["independent_over_joint"][k]["ci_disjoint_main"] and
                     gaps["independent_over_joint"][k]["ratio_by_draw"][0] >= 10) for k in PHYS}
    gng = {"joint_over_pixel_floor_ratio": near, "joint_within_10x_of_floor_r_energy": bool(near["r_energy"] <= 10),
           "indep_vs_joint_disjoint_and_10x_by_diagnostic": gap10,
           "criterion2_all_three": bool(all(gap10.values())),
           "joint_invalid_frame_frac": recs[0]["joint"]["invalid_frame_frac"],
           "no_go_invalid_gt_5pct": bool(recs[0]["joint"]["invalid_frame_frac"] > 0.05),
           "criterion3_evidence": {"cascaded_over_joint_ratio_main": {k: gaps["cascaded_over_joint"][k]["ratio_by_draw"][0] for k in PHYS},
                                   "e6_min_n_for_p95": {k: v["min_n_for_p95"] for k, v in e6res.items()},
                                   "note": "criterion 3 is a judgement call: compare these against the marginal metrics' CIs/E6 in results.md"}}

    res = {"tag": a.tag, "args": vars(a), "git": git_info(), "device": str(dev),
           "video_metric": None if metric is None else {"name": metric.name, "kind": metric.kind, "flavor": metric.flavor},
           "runs": {k: {"run": m["run"], "ckpt": m["ckpt"]} for k, m in models.items()},
           "floors": floors, "draws": recs, "bootstrap_ci_main_draw": boot, "spread_over_draws": spread,
           "ci_methods": {"per_sample (W1, physics medians)": "percentile bootstrap over generated samples, B=%d" % a.B,
                          "set_level (video metric, MMD)": em.SUBSAMPLE_METHOD, "m_sub": a.m_sub, "B": a.B,
                          "E6": "subsets drawn without replacement from the draws pooled per variant (see E6_pooling)"},
           "E4_gaps": gaps, "E3_ratio_curve": e3, "E6_sample_efficiency": e6res, "E6_pooling": e6_pool, "E7_auroc": e7res, "E8_time_reversal": e8res,
           "go_no_go": gng, "E2_curve_files": [str(jr.name)], "seconds": time.time() - t0}
    atomic_json(out_dir / "results.json", res)
    write_markdown(out_dir, res)
    print(f"wrote {out_dir}/results.json and results.md  ({fmt_s(time.time() - t0)})")


def fmt(x, p=3):
    return "n/a" if x is None else (f"{x:.{p}g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}")


def write_markdown(out_dir: Path, r: dict):
    B, S = r["bootstrap_ci_main_draw"], r["spread_over_draws"]
    cols = [("w1_mean", "W1"), ("psd_mean", "PSD"), ("video_metric", (r["video_metric"] or {}).get("name") or "video"), ("mmd", "MMD²"),
            ("r_energy", "r_energy"), ("r_energy_win_max", "win_max"), ("kin_rad_max", "kin_max (rad)"),
            ("ratio_win_over_glob", "win/glob"), ("invalid_frame_frac", "invalid frames")]
    md = [f"# Results ({r['tag']})", "", f"Runs: {r['runs']}. 3 draws x {r['args']['n']} samples, 250-step ancestral. "
          f"Main draw: point estimate [95% percentile-bootstrap CI, B={r['args']['B']}] for W1, PSD and the physics diagnostics (per-sample metrics). "
          f"For {(r['video_metric'] or {}).get('name') or 'the video metric'} and MMD² (set-level) the cell is the MEDIAN [2.5-97.5 percentile] over "
          f"{r['args']['B']} random subsets of m={r['args']['m_sub']} drawn WITHOUT replacement from the generated set and from the reference; "
          "same m for every row. Full-n point estimates are in their own table, and the spread over draws also uses full-n values. "
          f"Spread: mean ± std (min–max) over {r['args']['draws']} draws.", "",
          "## Main draw", "", "| variant | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for v in VARIANTS:
        row = []
        for k, _ in cols:
            st = B[v].get(k)
            if isinstance(st, dict):      # set-level metric: subsample median [range]; full-n point goes in its own table
                row.append(f"{fmt(st['subsample_median'])} [{fmt(st['subsample_ci95'][0])}, {fmt(st['subsample_ci95'][1])}]")
                continue
            val = r["draws"][0][v].get(k) if k in ("w1_mean", "psd_mean", "video_metric", "invalid_frame_frac") else \
                (scalars(r["draws"][0][v]).get(k))
            row.append("—" if val is None else fmt(val) + (f" [{fmt(st[0])}, {fmt(st[1])}]" if st else ""))
        md.append(f"| {v} | " + " | ".join(row) + " |")
    f = r["floors"]["median_and_ci"]
    md.append(f"| real data (floor) | | | | | {fmt(f['r_energy'][0])} [{fmt(f['r_energy'][1])}, {fmt(f['r_energy'][2])}] | "
              f"{fmt(f['r_energy_win_max'][0])} [{fmt(f['r_energy_win_max'][1])}, {fmt(f['r_energy_win_max'][2])}] | "
              f"{fmt(f['kin_rad_max'][0])} [{fmt(f['kin_rad_max'][1])}, {fmt(f['kin_rad_max'][2])}] | {fmt(r['floors']['summary']['ratio_of_medians_win_over_glob'])} | "
              f"{fmt(r['floors']['summary']['invalid_frame_frac'])} |")
    fvn = (r["video_metric"] or {}).get("name") or "video metric"
    md += ["", f"## Full-n point estimates of the set-level metrics (n={r['args']['n']} generated vs the held-out set; not comparable with the m={r['args']['m_sub']} cells)", "",
           f"| variant | {fvn} | MMD² |", "|---|---|---|"]
    for v in VARIANTS:
        md.append(f"| {v} | " + " | ".join(("—" if B[v].get(k) is None else fmt(B[v][k]["full_n_point"])) for k in ("video_metric", "mmd")) + " |")
    md += ["", "## Spread over draws (mean ± std, min–max)", "", "| variant | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for v in VARIANTS:
        row = []
        for k, _ in cols:
            s = S[v].get(k)
            row.append("—" if s is None else f"{fmt(s['mean'])} ± {fmt(s['std'])} ({fmt(s['min'])}–{fmt(s['max'])})")
        md.append(f"| {v} | " + " | ".join(row) + " |")
    md += ["", "## E4 gaps (worse variant / joint, median ratio per draw; CI disjointness on main draw)", ""]
    for g, d in r["E4_gaps"].items():
        for k, x in d.items():
            md.append(f"- {g} {k}: ratios by draw {[fmt(y) for y in x['ratio_by_draw']]}; CIs disjoint (main): {x['ci_disjoint_main']}; lower CI of worse / upper CI of joint: {fmt(x['ratio_ci_bounds_main'])}")
    md += ["", "## E3 windowed/global ratio (ratio of medians), joint checkpoints vs real", "",
           f"real: {fmt(r['E3_ratio_curve']['real_ratio_of_medians'])}", "", "| step | n | win/glob | r_energy | win_max |", "|---|---|---|---|---|"]
    md += [f"| {c['step']} | {c['n']} | {fmt(c['ratio_of_medians'])} | {fmt(c['r_energy'])} | {fmt(c['r_energy_win_max'])} |" for c in r["E3_ratio_curve"]["by_step"]]
    md += ["", "## E6 minimum n with P(correct ranking) >= 0.95 (None = never reached; 0.5 = chance)", "",
           "Subsets of size n are drawn WITHOUT replacement from the draws pooled per variant: " +
           "; ".join(f"{k}: {v['n_pooled']} samples from draws {v['draws_used']}" for k, v in r["E6_pooling"].items()), ""]
    for k, d in r["E6_sample_efficiency"].items():
        md.append(f"- {k}: {d['min_n_for_p95']}")
    md += ["", "## E7 AUROC (real held-out = negatives; larger score = more impossible)", "", "| positives | " + " | ".join(PHYS + ["r_energy_win_med", "kin_rad_med"]) + " |", "|---|---|---|---|---|---|"]
    md += [f"| {p} | " + " | ".join(fmt(x[k], 3) for k in PHYS + ["r_energy_win_med", "kin_rad_med"]) + " |" for p, x in r["E7_auroc"].items()]
    md += [""] + bm.e8_markdown(r["E8_time_reversal"])
    md += ["", "## Go / no-go flags (spec section 6, computed)", "", "```", json.dumps(r["go_no_go"], indent=1, default=float), "```"]
    new_path(out_dir / "results.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
