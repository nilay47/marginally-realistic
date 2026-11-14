"""Prints a run's key numbers FROM ITS FILES (never from memory):  python -m cvpr27.summary --run joint_s0"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from cvpr27.common import drive_root


def _g(m, *ks):
    for k in ks:
        if not isinstance(m, dict) or k not in m:
            return float("nan")
        m = m[k]
    return m


def show(run: str):
    d = drive_root() / run
    cfg = json.loads((d / "config.json").read_text())
    print(f"run {run}: variant={cfg['args']['variant']} steps={cfg['args']['steps']} params={cfg['n_params']/1e6:.1f}M "
          f"git={cfg['git']['commit'][:8]}{'+dirty' if cfg['git']['dirty'] else ''} batch={cfg['args']['batch']}")
    fl = cfg["floor_real_heldout"]
    print(f"floors (real, pixels->theta, uint8): r_energy {fl['r_energy']['median']:.2e}  "
          f"win_max {fl['r_energy_win_max']['median']:.2e}  kin_max {fl['kin_rad_max']['median']:.2e} rad  "
          f"win/glob ratio-of-medians {fl['ratio_of_medians_win_over_glob']:.2f}")
    rows = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(d / "metrics_0*.json")))]
    fin = d / "metrics_final_ancestral.json"
    if fin.exists():
        rows.append(json.loads(fin.read_text()))
    print(f"{'step':>7} {'sampler':>9} {'n':>5} {'W1':>8} {'PSD':>8} {'vidmetric':>12} {'r_energy':>10} {'win_max':>10} "
          f"{'kin_max':>10} {'win/glob':>9} {'MMD2':>10} {'invalid':>8}")
    for r in rows:
        m = r["metrics"]
        vm = m.get("video_metric", {})
        print(f"{r['step']:7d} {r['sampler']:>9} {r['n']:5d} {_g(m,'w1_mean'):8.4f} {_g(m,'psd_mean'):8.3f} "
              f"{(vm.get('name','')+' '+format(vm['value'],'.3g')) if vm else 'n/a':>12} "
              f"{_g(m,'physical','r_energy','median'):10.2e} {_g(m,'physical','r_energy_win_max','median'):10.2e} "
              f"{_g(m,'physical','kin_rad_max','median'):10.2e} {_g(m,'physical','ratio_of_medians_win_over_glob'):9.2f} "
              f"{_g(m,'mmd','value'):10.2e} {_g(m,'invalid_frame_frac'):8.4f}")
    print("files:", sorted(p.name for p in d.iterdir())[:6], "...", f"({len(list(d.iterdir()))} total)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    show(ap.parse_args().run)
