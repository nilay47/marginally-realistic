"""
Rebuilds results/log.json after the json.dump crash, WITHOUT retraining.

The crash happened after np.save, so results/samples_*.npy and results/real.npy
are all on disk -- final metrics for every variant can simply be recomputed.
Only the joint model's intermediate training curve was lost, and that was
printed to stdout, so it is transcribed below.

    python rescue.py
    python figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from metrics import evaluate

OUT = Path("results")

# step, loss, w1_mean, r_energy, r_dyn  -- transcribed from the run log.
# The final row (30000) is recomputed from samples_joint.npy instead, so it
# carries the full metric set including floors and r_kin.
CURVE = [
    (2500, 0.0582, 0.3519, 0.7180, 0.0686),
    (5000, 0.0241, 0.0518, 0.3025, 0.0095),
    (7500, 0.0229, 0.0635, 0.1826, 0.0060),
    (10000, 0.0220, 0.0768, 0.1209, 0.0045),
    (12500, 0.0211, 0.0712, 0.0855, 0.0033),
    (15000, 0.0206, 0.0725, 0.0554, 0.0024),
    (17500, 0.0201, 0.0765, 0.0455, 0.0018),
    (20000, 0.0197, 0.0680, 0.0356, 0.0014),
    (22500, 0.0194, 0.0882, 0.0229, 0.0012),
    (25000, 0.0191, 0.0850, 0.0172, 0.0011),
    (27500, 0.0189, 0.0846, 0.0129, 0.0011),
]

real = np.load(OUT / "real.npy")
results = {}

for name in ("joint", "joint_proj", "independent"):
    p = OUT / f"samples_{name}.npy"
    if not p.exists():
        print(f"  missing {p}, skipping")
        continue
    results[name] = [{**evaluate(real, np.load(p)), "epoch": 30000}]

if "joint" in results:
    final = results["joint"][0]
    floors = {k: v for k, v in final.items() if k.startswith("floor_")}
    hist = [{"epoch": s, "loss": l, "w1_mean": w,
             "r_energy": re, "r_dyn": rd, **floors}
            for s, l, w, re, rd in CURVE]
    results["joint"] = hist + [final]

with open(OUT / "log.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

print(f"wrote {OUT/'log.json'}\n")
print("--- FINAL ---")
f0 = results["joint"][-1]
print(f"  real-data floor  r_energy {f0['floor_r_energy']:.6f}  "
      f"r_dyn {f0['floor_r_dyn']:.8f}  r_kin {f0['floor_r_kin']:.8f}")
for k in ("independent", "joint", "joint_proj"):
    if k not in results:
        continue
    m = results[k][-1]
    print(f"  {k:12s}  w1 {m['w1_mean']:.4f}  r_energy {m['r_energy']:.5f}  "
          f"r_dyn {m['r_dyn']:.5f}  r_kin {m['r_kin']:.5f}")
