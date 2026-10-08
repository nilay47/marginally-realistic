# Marginally Realistic, Jointly Impossible

Code for the workshop paper. `train.py` trains the joint and independent diffusion models on the damped-oscillator data in `sim.py`; `metrics.py` holds the marginal and physical metrics.

## Extra seeds (1 and 2) on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nilay47/marginally-realistic/blob/main/notebooks/seeds_colab.ipynb)

The notebook runs `train.py --seed 1|2 --out <Drive folder>` (30k steps each), then `seeds_report.py` for the table, bootstrap CIs and `fig_training_3seeds`.

## Pendulum (Appendix E)

`sim_pendulum.py` is a drop-in replacement for `sim.py` (same API). To run the pendulum experiment, copy it over `sim.py` in a separate directory, not in this checkout.

`sim.py` in the repo root is the damped oscillator, and it is what the Colab notebook uses.
