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


def main():
    OUT.mkdir(exist_ok=True)
    nbs = {
        "nb0_benchmark": nb0(),
        "nb1_train_joint": train_nb("nb1_train_joint", "nb1: train the joint model", "joint", "joint_s0",
                                    "**Launch this first**; it is the longest run and its checkpoints drive E2/E3."),
        "nb2_train_video_only": train_nb("nb2_train_video_only", "nb2: train the video-only model", "video_only", "video_only_s0"),
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
