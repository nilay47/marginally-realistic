# CVPR 2027 extension — experiment spec for Claude Code

Paper: "Beyond Visual Realism: Auditing Joint Physical Consistency in Video-Sensor Generation" (registered on OpenReview).
Deadline: **Mon Nov 16 AoE = Tue Nov 17, 6:59 AM ET**. Go/no-go: **Sun Nov 15, 8 PM ET**.
Workflow: Claude Code writes the code and notebooks and pushes to GitHub. Each notebook clones the repo, mounts Drive, runs on a Colab A100 and writes everything under `/content/drive/MyDrive/physgen_cvpr27/<run_name>/`.

> **Revision (Nov 14):** `TAU_AMP` retuned 3.0 -> 8.0 (the workshop pendulum's own torque amplitude). At 3.0 the benchmark was far less nonlinear than the workshop pendulum (median |theta| 0.25 vs 0.49 rad, 1.4% of samples beyond 1 rad vs 23%). Sections 1.1 and 1.4 below are updated; the 1.4 table is from the re-run self-test.

## Rules (read first)
1. **Leave the workshop code and results alone** (`sim.py`, `train.py`, `metrics.py`, `seeds_report.py`, etc.). Put the new work in a new folder, `cvpr27/`, with its own files.
2. **Save model weights** (EMA) at every checkpoint, plus the generated samples as `.npz`. The workshop seed-0 artifacts were lost; that must not happen again.
3. **Nothing is ever overwritten on Drive.** If `log.json` exists, skip the run (resume logic). Every run writes `config.json` with all hyperparameters and the git commit hash.
4. **Never print or hardcode a number that wasn't computed by the code in that run.** Placeholder seed-0 numbers are not allowed in any report or figure.
5. **If something looks wrong** (NaNs, a floor above 1e-2, an extractor invalid-frame fraction above 1% on real data), stop and report. Don't silently fix it.
6. **Time the first 200 training steps** and print the ETA for the full run.

## 1. Benchmark: rendered pendulum with video + gyro + torque
The reference implementation is `physvid.py`. It is tested: it runs in about 2 s on CPU and gives the numbers in section 1.4. Adopted as `cvpr27/physvid.py`.

### 1.1 Physics
- Equation: m l² θ'' + c θ' + m g l sin θ = τ(t), with m = l = 1, g = 9.81, c = 0.4.
- τ is a band-limited random Fourier series: 16 modes, frequencies 0.5–8 rad/s, **amplitude 8.0**.
- Initial conditions: θ0 ~ N(0, 0.3), ω0 ~ N(0, 1).
- Integration: RK4 with 10 substeps per frame.
- **Training config:** T = 64 frames, DT = 0.04 s (2.56 s clips), energy windows of 8 frames (env vars `PV_T=64 PV_DT=0.04 PV_WIN=8`).
- **Resulting theta statistics (n=2000, per element; workshop pendulum in brackets):** median |θ| 0.496 [0.489]; 90th pct 1.61 [1.61]; fraction |θ|>1 rad 0.234 [0.232]; fraction |θ|>π 0.024 [0.027]; median |sin θ − θ|/|sin θ| 0.042 [0.041]; max |Δθ| per frame 0.50 rad (unwrap is safe).

### 1.2 Rendering
- 32×32 grayscale frames with a Gaussian bob (σ = 1.5 px) at radius 12 px from a fixed pivot at the image centre.
- Stored as uint8.
- Sensors per frame: gyro = ω and torque = τ.

### 1.3 Pixel → θ extractor (analytic, no learning)
1. Find the brightest pixel in the frame.
2. Take the intensity-weighted centroid in a 7×7 window after subtracting a 0.05 background.
3. Compute θ = atan2(Δcol, Δrow) relative to the pivot, then unwrap over time.
4. Mark a frame invalid if its peak is below 0.3. **Always report the invalid-frame fraction for generated samples.**

### 1.4 Diagnostics (none of them differentiates generated signals)
All three use θ from pixels, ω from the gyro and τ from the torque:
- **Global energy balance.** E = ½ω² + g(1 − cos θ). The residual is |E(T) − E(0) − (∫τω dt − c∫ω² dt)|, using Simpson integration, normalised by the per-sample mean |E|.
- **Multi-window energy balance.** The same identity on windows of 8 frames with stride 1. Report the worst window and the median window.
- **Integrated video–gyro consistency.** On each window, |θ_video(t_b) − θ_video(t_a) − ∫ω_gyro dt|, in radians. Report the worst window and the median window.

Self-test results at T=64, TAU_AMP=8.0 (n=512, uint8 frames) — medians:

| condition | global | worst window | video–gyro (rad) |
|---|---|---|---|
| floor, true θ | 1.1e-4 | 7.7e-5 | 8.6e-6 |
| floor, pixels → θ | 2.5e-4 | 1.4e-3 | 9.1e-4 |
| mispaired video/sensors | 0.87 | 1.96 | 1.59 |
| video lags sensors by 3 frames | 0.22 | 0.94 | 0.45 |
| torque corrupted, globally balanced | 2.5e-4 | 0.35 | 9.1e-4 |
| pixel noise σ = 0.02 | 1.4e-3 | 9.5e-3 | 6.2e-3 |
| pixel noise σ = 0.05 | 3.6e-3 | 2.3e-2 | 1.5e-2 |
| pixel noise σ = 0.1 | 7.4e-3 | 4.7e-2 | 3.3e-2 |

(Extractor invalid-frame fraction on clean uint8 frames: 0. nb0 recomputes this whole table on the 2,048 held-out clips; those numbers, not these, go in the paper.)

## 2. Generative models

### 2.1 Data
- **Train set:** 10,000 clips, simulation seed 0.
- **Held-out real set:** 2,048 clips, seed 1. Use it for all metrics and floors.
- **Cache** both to Drive as uint8 video plus float32 sensors, and reuse the cache across notebooks.

### 2.2 Normalisation
- Video is mapped to [−1, 1].
- Gyro and torque are standardised with train-set mean and std, which are saved in `config.json`.
- **Always de-normalise before computing diagnostics.**

### 2.3 Architecture: one (2+1)D U-Net for all variants
- **Input:** video plus sensors broadcast as constant planes, giving (B, C, T, H, W). C = 3 for joint, 1 for video-only and 2 for sensor-only.
- **Sensor readout:** the spatial mean of the sensor output channels. The sensor loss is MSE(spatial_mean(ε̂_s), ε_s).
- **Residual blocks:**
  - GroupNorm → SiLU → spatial conv (1,3,3) → temporal conv (3,1,1), applied twice;
  - timestep embedding added after the first conv.
- **Base width and levels:**
  - 64 channels, multipliers (1, 2, 2, 4), two blocks per level.
  - Each downsample is stride (2, 2, 2): T 64→32→16→8 and space 32→16→8→4.
  - For sensor-only, use 4×4 planes and downsample time only.
- **Bottleneck:** full self-attention over the 8×4×4 tokens. This gives the network a global temporal receptive field, which the energy balance needs.

### 2.4 Diffusion
- Cosine schedule, 250 steps, ε-prediction.
- AdamW, lr 2e-4, warmup 500 steps, grad clip 1.0, EMA 0.999.
- bf16 autocast, batch 32, 20k steps.
- **Sampling:** DDIM with 50 steps for checkpoint evaluation, full 250-step ancestral sampling for the final samples.

### 2.5 Variants

| run | what | training |
|---|---|---|
| `joint_s0` | C=3, one model | yes |
| `video_only_s0` | C=1 | yes |
| `sensor_only_s0` | C=2, 4×4 planes | yes (cheap) |
| `independent` | video from `video_only`, sensors from `sensor_only`, sampled separately and paired | no |
| `cascaded` | video from `video_only`, then sensors in-painted with `joint` by replacement sampling (at each step, overwrite the video channel with a noised copy of the given video) | no |
| `shuffle_oracle` | real held-out video paired with sensors from a different real clip | no |
| `joint_s1`, `joint_s2` | only if time allows | yes |

## 3. Metrics
**Run every metric on 1,024 generated samples against the held-out real set.** Report 95% CIs by bootstrap over generated samples (B = 200). Because the weights are saved, also check the main numbers on 3 independent redraws of 1,024 samples.

**Marginal metrics (what the field reports):**
- **FVD:** I3D features, with frames upsampled to 224 and repeated to 3 channels. Use a standard implementation and pin its version. If FVD is blocked, fall back to per-frame KID with Inception features and record the fallback in the config.
- **Sensor W1:** per channel, normalised by the real std. This is the workshop metric.
- **Sensor PSD distance.**

**Generic joint metric:** RBF-kernel MMD (median heuristic) on the concatenation [flattened 16×16 video, gyro, torque].

**Physical metrics:**
- the three diagnostics from section 1.4;
- the invalid-frame fraction;
- the floors (true θ; pixels → θ on clean, uint8, and σ = 0.02/0.05/0.1 noise).

**Training-dynamics checkpoints (joint):** every 2k steps, sample 256 with DDIM 50 and log every metric above. FVD at checkpoints may use 256 samples, and the log should say so.

## 4. Notebooks (one per Colab session, so they can run in parallel)
- `nb0_benchmark.ipynb` — CPU is fine. Builds the data cache and floors table, runs the corruption self-test and the noise-robustness sweep, writes `floors.json` and the self-test table.
- `nb1_train_joint.ipynb` — `joint_s0`, with checkpoint evaluation. **Launch first.**
- `nb2_train_video_only.ipynb` — `video_only_s0`.
- `nb3_train_sensor_only.ipynb` — `sensor_only_s0`.
- `nb4_eval.ipynb` — loads the final weights and samples every variant (including `cascaded`, `independent` and `shuffle_oracle`). It then:
  - computes all metrics with CIs;
  - writes `results.json` plus a markdown table;
  - writes the figures as PDF and PNG.
- `nb5_train_joint_seeds.ipynb` — `joint_s1` and `joint_s2`. Optional.

Every notebook must:
- check for a GPU (except nb0);
- mount Drive;
- clone the repo (or pull if it already exists);
- skip runs whose `log.json` exists;
- finish with a summary cell that prints its key numbers.

## 5. Experiments → paper claims

| id | experiment | claim it supports |
|---|---|---|
| E1 | floors + corruption table + noise sweep | the audit is valid: tiny floors, and each check catches its failure mode |
| E2 | `joint_s0` checkpoints: FVD, W1 vs the 3 diagnostics | marginal / visual metrics saturate while physics keeps improving (the workshop result, now on video) |
| E3 | **new:** the windowed/global ratio over checkpoints, against the ratio on real data | global balance converges while local violations persist (cancellation in learned models, not just a constructed corruption) |
| E4 | joint vs independent vs cascaded at the final checkpoint | similar FVD/W1, physics differing by orders of magnitude; cascaded is the realistic baseline |
| E5 | shuffle oracle | marginals are identical by construction while the physics is destroyed |
| E6 | MMD vs diagnostics: how many samples each needs to separate the variants (sample sizes 64, 128, 256, 512, 1024) | a generic joint metric is far less sample-efficient or fails |
| E7 | per-sample detection AUROC of each diagnostic, on real vs mispaired/lagged/corrupted clips and on real vs generated samples | the audit works per sample, not just on average |

## 6. Go/no-go — Sun 8 PM ET
- **Go if** `joint_s0` reaches global energy within about 10× of the pixel floor, **and** joint vs independent has disjoint CIs on the physics metrics with a gap of at least 10×, **and** at least one of E3, E4 (cascaded) or E6 shows something FVD/W1/MMD miss.
- **No-go if** the joint model can't produce valid frames (more than 5% invalid) or the physics never gets near the floor. In that case, don't submit; take the work to ICML 2027 / TMLR instead.

## 7. Schedule (ET)
- **Sat evening:** Claude Code builds `physvid.py`, the models and nb0–nb3, and pushes. Run nb0, then launch nb1, nb2 and nb3.
- **Sun morning:** check the logs. Launch nb5 if GPU time is available. Claude Code builds nb4.
- **Sun afternoon:** run nb4, then go/no-go at 8 PM.
- **Mon:** figures and writing. Submit Monday evening, well before 6:59 AM Tue.
- **By Nov 23 (supplementary):** extra seeds, code, more samples.
