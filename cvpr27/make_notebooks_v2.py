"""Generates ONLY the v2 notebooks (nb1v2_joint, nb2v2_video_only, nb3v2_sensor_only, nb4v2_eval). Does not touch the others:
    python -m cvpr27.make_notebooks_v2
v2 = fast config: T=32 frames (PV_T=32, DT 0.04, WIN 8), base 32, batch 64, 10k steps max, sensor loss weight 10, 128-sample DDIM-50
checkpoint evals without FVD, FVD only on the final samples. Run names <variant>_v2; nothing from the T=64 runs is touched."""
import json

from cvpr27 import make_notebooks as mn

md, code, badge = mn.md, mn.code, mn.badge
DRIVE_ROOT = "/content/drive/MyDrive/physgen_cvpr27"
ENV = code('''
# v2 runs use T=32 frames. These must be set BEFORE cvpr27 is imported (physvid reads them at import).
%env PV_T=32
%env PV_DT=0.04
%env PV_WIN=8
''')
CACHE = code('''
%%bash
# T=32 cache (separate file names from the T=64 cache). Retries cover the case where another notebook is building it right now.
export PV_T=32 PV_DT=0.04 PV_WIN=8
cd /content/repo
for i in 1 2 3 4 5 6; do python -m cvpr27.build_cache && break; echo "cache busy, retry $i"; sleep 20; done
ls -la /content/drive/MyDrive/physgen_cvpr27/cache/ | grep T32
test -f /content/drive/MyDrive/physgen_cvpr27/cache/video_metric.json || (echo "MISSING video_metric.json: run nb0_benchmark first" && exit 1)
''')


def train_v2(name, title, variant, extra=""):
    run = f"{variant}_v2"
    d = f"{DRIVE_ROOT}/{run}"
    return [
        md(f"{badge(name)}\n\n# {title}\nFast config **v2** for `{run}`: T=32 frames, 32x32, base 32 channels, batch 64, lr 2e-4, **10k steps max**, "
           "sensor loss weight 10 (recorded in `config.json`), checkpoint every 2k steps with 128 DDIM-50 samples (physics + W1 + MMD, **no FVD**); "
           "FVD only on the final samples. Streams its log and appends to `progress.log` on Drive. Safe to rerun: a finished run (`log.json`) is skipped, "
           f"an interrupted run resumes, nothing on Drive is overwritten (new run name `{run}`). **To stop early:** interrupt the training cell, then run "
           f"the *finalize* cell (newest EMA checkpoint -> final samples + `log.json`). {extra}\n\n"
           "Needs `nb0_benchmark` finished (it decides FVD vs KID once). **If you start several v2 notebooks at once, start the joint one first** (it builds the T=32 cache; the others wait/retry)."),
        md("## 1. GPU"), mn.GPU, md("## 2. Drive"), mn.DRIVE, md("## 3. Code"), mn.CLONE, md("## 4. Install"), mn.INSTALL,
        md("## 5. T=32 environment + cache"), ENV, CACHE,
        md("## 6. TIMING (50 training steps + one sampling batch). Read the projection before launching."),
        code(f'''
%%bash
export PV_T=32 PV_DT=0.04 PV_WIN=8
cd /content/repo
PYTHONUNBUFFERED=1 python -u -m cvpr27.train --variant {variant} --run_name {run} --preset v2 --time_only
'''),
        md(f"## 7. Train `{run}` (streams; also appended to `{run}/progress.log` on Drive)"),
        code(f'''
%%bash
set -o pipefail
export PV_T=32 PV_DT=0.04 PV_WIN=8
mkdir -p "{d}"
cd /content/repo
PYTHONUNBUFFERED=1 python -u -m cvpr27.train --variant {variant} --run_name {run} --preset v2 --seed 0 2>&1 | tee -a "{d}/console_$(date +%s).log"
'''),
        md("## 8. Finalize (only if you stopped early, or the run did not reach `log.json`)\nNewest `ema_*.pt` -> final 1,024 ancestral samples + metrics (with the video metric) + `log.json`. "
           "Skipped automatically if the run already finished."),
        code(f'''
%%bash
set -o pipefail
export PV_T=32 PV_DT=0.04 PV_WIN=8
cd /content/repo
test -f "{d}/log.json" && echo "already finished (log.json exists)" || PYTHONUNBUFFERED=1 python -u -m cvpr27.train --variant {variant} --run_name {run} --preset v2 --finalize 2>&1 | tee -a "{d}/console_finalize_$(date +%s).log"
'''),
        md("## 9. Summary (read from the run's files)"),
        code(f'''
%%bash
export PV_T=32 PV_DT=0.04 PV_WIN=8
cd /content/repo && python -m cvpr27.summary --run {run}
'''),
    ]


def nb4v2():
    return [
        md(f"{badge('nb4v2_eval')}\n\n# nb4v2: evaluation of the v2 runs\nSame as nb4 (E2-E8, 3 draws x 1,024, 250-step ancestral, subsampling CIs for FVD/MMD) but for "
           "`joint_v2`, `video_only_v2`, `sensor_only_v2` at T=32 (`--suffix v2`; outputs in `eval_v2/`, samples in `eval_samples/v2/`). "
           "Needs the three v2 runs finished (`log.json`): run each notebook's finalize cell if you stopped early."),
        md("## 1. GPU"), mn.GPU, md("## 2. Drive"), mn.DRIVE, md("## 3. Code"), mn.CLONE, md("## 4. Install"), mn.INSTALL,
        md("## 5. T=32 environment"), ENV,
        md("## 6. Check inputs"),
        code('''
import os
for r in ["joint_v2", "video_only_v2", "sensor_only_v2"]:
    assert os.path.exists(f"{ROOT}/{r}/log.json"), f"{r} not finished (no log.json): finish or finalize it first"
for f in ["cache/pendulum_T32_dt0.04_heldout.npz", "cache/pendulum_T32_dt0.04_train.npz", "cache/video_metric.json"]:
    assert os.path.exists(f"{ROOT}/{f}"), f"missing {f}"
print("inputs present")
'''),
        md("## 7. Sample + evaluate"),
        code(f'''
%%bash
set -o pipefail
export PV_T=32 PV_DT=0.04 PV_WIN=8
mkdir -p {DRIVE_ROOT}/eval_v2_logs
cd /content/repo
PYTHONUNBUFFERED=1 python -u -m cvpr27.eval --suffix v2 2>&1 | tee -a "{DRIVE_ROOT}/eval_v2_logs/eval_$(date +%s).log"
'''),
        md("## 8. Figures (PDF + PNG)"),
        code('''
%%bash
export PV_T=32 PV_DT=0.04 PV_WIN=8
cd /content/repo && python -u -m cvpr27.figures --suffix v2
'''),
        md("## 9. Summary"),
        code('''
from IPython.display import Markdown, Image, display
import glob
display(Markdown(open(f"{ROOT}/eval_v2/results.md").read()))
for p in sorted(glob.glob(f"{ROOT}/eval_v2/figures/*.png")):
    print(p); display(Image(p))
'''),
    ]


def main():
    nbs = {
        "nb1v2_joint": train_v2("nb1v2_joint", "nb1v2: train the joint model (fast v2)", "joint", "**Start this one first.**"),
        "nb2v2_video_only": train_v2("nb2v2_video_only", "nb2v2: train the video-only model (fast v2)", "video_only"),
        "nb3v2_sensor_only": train_v2("nb3v2_sensor_only", "nb3v2: train the sensor-only model (fast v2)", "sensor_only"),
        "nb4v2_eval": nb4v2(),
    }
    for name, cells in nbs.items():
        nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
              "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                           "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}}
        (mn.OUT / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
        print("wrote", mn.OUT / f"{name}.ipynb")


if __name__ == "__main__":
    main()
