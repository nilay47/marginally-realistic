"""Generates ONLY cvpr27/notebooks/nb7_diag.ipynb (does not touch the other notebooks):  python -m cvpr27.make_nb7"""
import json

from cvpr27 import make_notebooks as mn

md, code = mn.md, mn.code


def nb7():
    return [
        md(f"{mn.badge('nb7_diag')}\n\n# nb7: joint-run diagnostics (CPU, no training)\n"
           "Reads the saved `samples_ckpt_*.npz` of `joint_s0` (the run need not be finished; it only reads, never writes there) and asks: "
           "(1) physics on all clips vs valid-only clips; (2) **within-checkpoint shuffle baseline** (generated video i with generated sensors j, 20 permutations): "
           "did the joint model learn ANY coupling?; (3) which term of the energy balance fails (dE vs work vs dissipation), gyro/torque scale vs real, "
           "video-vs-gyro lag; (4) a figure of theta_video vs theta0 + cumint(gyro). Works on a CPU runtime (Runtime -> Change runtime type -> CPU). "
           "Writes new files `eval_diag/diag.json`, `diag.md`, `fig_diag_theta.{pdf,png}` (use a new `--tag` to rerun)."),
        md("## 1. Drive"), mn.DRIVE, md("## 2. Code"), mn.CLONE, md("## 3. Install"),
        code("!pip install -q scipy matplotlib"),
        md("## 4. Check inputs"),
        code('''
import glob, os
assert os.path.exists(f"{ROOT}/cache/pendulum_T64_dt0.04_heldout.npz"), "missing held-out cache: run nb0_benchmark first"
ck = sorted(glob.glob(f"{ROOT}/joint_s0/samples_ckpt_*.npz"))
print(len(ck), "checkpoint sample files:", [os.path.basename(p) for p in ck])
assert ck, "no joint_s0 samples_ckpt_*.npz found"
'''),
        md("## 5. Run diagnostics"),
        code("!cd /content/repo && PYTHONUNBUFFERED=1 python -u -m cvpr27.diag --run joint_s0 --tag diag"),
        md("## 6. Report and figure"),
        code('''
from IPython.display import Markdown, Image, display
display(Markdown(open(f"{ROOT}/eval_diag/diag.md").read()))
display(Image(f"{ROOT}/eval_diag/fig_diag_theta.png"))
'''),
    ]


if __name__ == "__main__":
    nb = {"cells": nb7(), "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}}}
    p = mn.OUT / "nb7_diag.ipynb"
    p.write_text(json.dumps(nb, indent=1))
    print("wrote", p)
