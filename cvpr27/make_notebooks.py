"""Generates cvpr27/notebooks/*.ipynb. Run from the repo root:  python cvpr27/make_notebooks.py"""
import json
from pathlib import Path

REPO = "nilay47/marginally-realistic"
OUT = Path(__file__).resolve().parent / "notebooks"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip().splitlines(True)}


def badge(name):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/cvpr27/notebooks/{name}.ipynb)")


GPU = code('''
import subprocess, torch
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout or "nvidia-smi: no output")
if not torch.cuda.is_available():
    raise SystemExit("No GPU: Runtime -> Change runtime type -> A100 GPU, then rerun.")
print("cuda:", torch.cuda.get_device_name(0), "| mem GB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
''')
DRIVE = code('''
from google.colab import drive
drive.mount("/content/drive")
import os
ROOT = "/content/drive/MyDrive/physgen_cvpr27"
os.makedirs(ROOT, exist_ok=True)
''')
CLONE = code(f'''
import os
if os.path.exists("/content/repo"):
    !git -C /content/repo pull
else:
    !git clone https://github.com/{REPO}.git /content/repo
%cd /content/repo
!git log -1 --format="commit %H%n%s"
''')
INSTALL = code('''
# torch / torchvision come with Colab. pytorchvideo is the second I3D source for FVD (see videofeat.py).
!pip install -q scipy matplotlib pytorchvideo
''')


def train_nb(name, title, variant, run, extra_md=""):
    cells = [
        md(f"{badge(name)}\n\n# {title}\nTrains `{run}` (spec section 2). Safe to rerun: a finished run (`log.json` present) is skipped, "
           f"an interrupted run resumes from its newest `resume_*.pt`, and nothing on Drive is ever overwritten. {extra_md}\n\n"
           "**Run `nb0_benchmark` first**: it builds the data cache and decides FVD vs KID once for every run."),
        md("## 1. GPU"), GPU, md("## 2. Drive"), DRIVE, md("## 3. Code"), CLONE, md("## 4. Install"), INSTALL,
        md("## 5. Check nb0 outputs exist"),
        code('''
import os
for f in ["cache/pendulum_T64_dt0.04_train.npz", "cache/pendulum_T64_dt0.04_heldout.npz", "cache/video_metric.json"]:
    assert os.path.exists(f"{ROOT}/{f}"), f"missing {f}: run nb0_benchmark first"
print("nb0 outputs present")
'''),
        md(f"## 6. Train `{run}`\nPrints the 200-step ETA, then one line per checkpoint (every 2k steps). Console output is also saved to a "
           "timestamped log on Drive."),
        code(f'''
%%bash
set -o pipefail
mkdir -p "/content/drive/MyDrive/physgen_cvpr27/{run}"
cd /content/repo
PYTHONUNBUFFERED=1 python -u -m cvpr27.train --variant {variant} --run_name {run} --seed 0 2>&1 | tee -a "/content/drive/MyDrive/physgen_cvpr27/{run}/console_$(date +%s).log"
'''),
        md("## 7. Summary (read from the run's files)"),
        code(f'!cd /content/repo && python -m cvpr27.summary --run {run}'),
    ]
    return cells


def nb0():
    return [
        md(f"{badge('nb0_benchmark')}\n\n# nb0: benchmark, floors, self-test, video-metric decision\n"
           "**Use a GPU runtime** (the I3D self-check and the real-reference features are computed here and cached for every other notebook). "
           "Builds the data cache (10,000 train + 2,048 held-out clips), decides FVD vs KID **once**, then computes E1: the floors, the corruption "
           "self-test and the pixel-noise sweep on the held-out real clips. Every number is computed in this run."),
        md("## 1. Runtime"),
        code('''
import subprocess, torch
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout or "no GPU (CPU run: the video-metric check will be very slow)")
print("cuda available:", torch.cuda.is_available())
'''),
        md("## 2. Drive"), DRIVE, md("## 3. Code"), CLONE, md("## 4. Install"), INSTALL,
        md("## 5. Run\nOrder: cache -> video-metric decision (tries the StyleGAN-V I3D torchscript, then pytorchvideo I3D-R50, then KID; each must pass a "
           "behavioural self-check) -> floors / corruptions / noise sweep. Stops (Rule 5) if the extractor's invalid-frame fraction on real data "
           "exceeds 1% or a clean floor exceeds 1e-2."),
        code('!cd /content/repo && python -u -m cvpr27.benchmark'),
        md("## 6. Summary"),
        code('''
import json
b = json.load(open(f"{ROOT}/benchmark/floors.json"))
vm = b["video_metric"]
print("VIDEO METRIC:", vm["kind"].upper(), "| flavor:", vm["flavor"], "| fallback_to_kid:", vm["fallback_to_kid"])
for a in vm["attempts"]:
    print("  attempt", a["flavor"], "->", "OK" if a.get("ok") else "FAILED", a.get("reason") or {k: v for k, v in a["check"].items() if k != "ok"})
print("\\ngit:", b["git"]["commit"][:8], "| rule-5 gate failures:", b["rule5_gate_failures"])
print(open(f"{ROOT}/benchmark/selftest_table.md").read())
'''),
    ]


def nb4():
    return [
        md(f"{badge('nb4_eval')}\n\n# nb4: evaluation (E1-E7)\nLoads the **final EMA weights** of `joint_s0`, `video_only_s0`, `sensor_only_s0` and samples every variant "
           "(`joint`, `video_only`, `sensor_only`, `independent`, `cascaded`, `shuffle_oracle`): **3 draws x 1,024 samples, full 250-step ancestral**. "
           "Reports the main draw (draw 0) with 95% bootstrap CIs (B=200) and the spread (mean / std / min / max) across the 3 draws; then E3, E6 "
           "(sample efficiency), E7 (per-sample AUROC), the go/no-go flags (computed), and the figures. Sampling is resumable: each (variant, draw) "
           ".npz is written once and reused on rerun. Needs nb0-nb3 finished."),
        md("## 1. GPU"), GPU, md("## 2. Drive"), DRIVE, md("## 3. Code"), CLONE, md("## 4. Install"), INSTALL,
        md("## 5. Check inputs"),
        code('''
import os
for r in ["joint_s0", "video_only_s0", "sensor_only_s0"]:
    assert os.path.exists(f"{ROOT}/{r}/log.json"), f"{r} not finished (no log.json): run its training notebook first"
assert os.path.exists(f"{ROOT}/cache/video_metric.json"), "run nb0 first"
print("inputs present")
'''),
        md("## 6. Sample + evaluate\nPrints each sampling job's time. `--tag main` writes to `eval_main/` and refuses to overwrite: use a new `--tag` to rerun "
           "the metrics (cached samples are reused)."),
        code('''
%%bash
set -o pipefail
mkdir -p /content/drive/MyDrive/physgen_cvpr27/eval_main_logs
cd /content/repo
PYTHONUNBUFFERED=1 python -u -m cvpr27.eval --tag main 2>&1 | tee -a "/content/drive/MyDrive/physgen_cvpr27/eval_main_logs/eval_$(date +%s).log"
'''),
        md("## 7. Figures (PDF + PNG)"),
        code('!cd /content/repo && python -u -m cvpr27.figures --tag main'),
        md("## 8. Summary (read from results.json / results.md)"),
        code('''
from IPython.display import Markdown, Image, display
import glob, json
display(Markdown(open(f"{ROOT}/eval_main/results.md").read()))
for p in sorted(glob.glob(f"{ROOT}/eval_main/figures/*.png")):
    print(p); display(Image(p))
'''),
    ]


def nb5():
    cells = [
        md(f"{badge('nb5_train_joint_seeds')}\n\n# nb5: joint model, seeds 1 and 2 (optional)\nTrains `joint_s1` and `joint_s2` exactly as "
           "`joint_s0` (only `--seed` changes: init, batch order and noise; the training set is simulation seed 0 for all). Each run skips if finished and "
           "resumes if interrupted. Then writes the seed report and figure (new files only). **Run only if GPU time is available.**"),
        md("## 1. GPU"), GPU, md("## 2. Drive"), DRIVE, md("## 3. Code"), CLONE, md("## 4. Install"), INSTALL,
        md("## 5. Check nb0 outputs"),
        code('''
import os
for f in ["cache/pendulum_T64_dt0.04_train.npz", "cache/pendulum_T64_dt0.04_heldout.npz", "cache/video_metric.json"]:
    assert os.path.exists(f"{ROOT}/{f}"), f"missing {f}: run nb0_benchmark first"
print("nb0 outputs present")
'''),
    ]
    for s in (1, 2):
        cells += [md(f"## {5 + s}. Train `joint_s{s}`"),
                  code(f'''
%%bash
set -o pipefail
mkdir -p "/content/drive/MyDrive/physgen_cvpr27/joint_s{s}"
cd /content/repo
PYTHONUNBUFFERED=1 python -u -m cvpr27.train --variant joint --run_name joint_s{s} --seed {s} 2>&1 | tee -a "/content/drive/MyDrive/physgen_cvpr27/joint_s{s}/console_$(date +%s).log"
''')]
    cells += [md("## 8. Seed report (joint_s0 / s1 / s2, read from their files)"),
              code('!cd /content/repo && python -u -m cvpr27.seeds --tag seeds'),
              code('''
from IPython.display import Image, display
display(Image(f"{ROOT}/eval_seeds/fig_seeds_training.png"))
''')]
    return cells


def main():
    OUT.mkdir(exist_ok=True)
    nbs = {
        "nb0_benchmark": nb0(),
        "nb1_train_joint": train_nb("nb1_train_joint", "nb1: train the joint model", "joint", "joint_s0",
                                    "**Launch this first**; it is the longest run and its checkpoints drive E2/E3."),
        "nb2_train_video_only": train_nb("nb2_train_video_only", "nb2: train the video-only model", "video_only", "video_only_s0"),
        "nb4_eval": nb4(),
        "nb5_train_joint_seeds": nb5(),
        "nb3_train_sensor_only": train_nb("nb3_train_sensor_only", "nb3: train the sensor-only model", "sensor_only", "sensor_only_s0",
                                          "Cheap (4x4 planes)."),
    }
    for name, cells in nbs.items():
        nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
              "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                           "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}}
        (OUT / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
        print("wrote", OUT / f"{name}.ipynb")


if __name__ == "__main__":
    main()
