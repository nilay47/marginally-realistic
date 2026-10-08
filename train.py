"""
    python train.py --gate     # ~3 min. Must print PASS.
    python train.py            # ~25 min on a T4

Changed from v1: training is counted in optimizer STEPS, not epochs. v1 ran
~3,100 steps, which is roughly an order of magnitude short for a DDPM and is
why samples were garbage. Default is now 30,000 steps with EMA and a cosine
LR decay.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from ddpm import EMA, Diffusion, UNet1D
from metrics import evaluate
from sim import T_LEN, project, projection_matrix, simulate



def normalise(d):
    mu = d.mean(axis=(0, 2), keepdims=True)
    sd = d.std(axis=(0, 2), keepdims=True)
    return (d - mu) / sd, mu, sd


def fit(data_n, channels, args, device, tag, base=64, on_ckpt=None):
    model = UNet1D(channels=channels, base=base).to(device)
    ema = EMA(model, args.ema)
    diff = Diffusion(args.diff_steps, device)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, args.warmup))
        * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    X = torch.tensor(data_n, device=device)
    n, log, run = X.shape[0], [], 0.0

    for step in range(1, args.steps + 1):
        model.train()
        idx = torch.randint(0, n, (args.batch,), device=device)
        loss = diff.loss(model, X[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); ema.update(model)
        run += loss.item()

        last = step == args.steps
        if on_ckpt and (step % args.ckpt_every == 0 or last):
            avg = run / min(step, args.ckpt_every); run = 0.0
            k = args.n_sample if last else args.n_ckpt
            rec = on_ckpt(
                lambda m: diff.sample(ema.shadow, m, channels, T_LEN, device)
                              .cpu().numpy(), step, avg, k)
            if rec:
                log.append(rec)
        elif step % args.ckpt_every == 0:
            print(f"  [{tag}] step {step:6d}  loss {run/args.ckpt_every:.4f}")
            run = 0.0

    def sampler(k):
        return diff.sample(ema.shadow, k, channels, T_LEN, device).cpu().numpy()

    return sampler, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--diff_steps", type=int, default=250)
    ap.add_argument("--ckpt_every", type=int, default=2500)
    ap.add_argument("--n_ckpt", type=int, default=512)
    ap.add_argument("--n_sample", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    tagdir = out if args.seed == 0 else out / f"seed{args.seed}"
    tagdir.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")

    if args.gate:
        # A short run on the REAL task. The previous gate trained on a swiss
        # roll, which is structurally unlike this data (2 channels, length 16,
        # constant in time) and failed twice while the actual model was
        # visibly learning. Testing the thing you care about beats a proxy.
        print("gate: 3000 steps on the real task, want w1_mean < 0.5")
        data = simulate(8000)
        dn, mu, sd = normalise(data)
        real = data[:512]
        args.steps, args.ckpt_every, args.n_ckpt = 3000, 1500, 512
        got = {}

        def gate_ckpt(sample, step, loss, k):
            m = evaluate(real, (sample(k) * sd + mu).astype(np.float32))
            got["w1"] = m["w1_mean"]
            print(f"  step {step:5d}  loss {loss:.4f}  w1 {m['w1_mean']:.4f}"
                  f"  r_energy {m['r_energy']:.4f}")
            return m

        fit(dn, 4, args, dev, "gate", on_ckpt=gate_ckpt)
        ok = got.get("w1", 9e9) < 0.5
        print(f"\n  GATE: w1_mean = {got.get('w1', float('nan')):.4f} "
              f"({'PASS' if ok else 'FAIL -- stop, send me the output'})")
        raise SystemExit(0 if ok else 1)

    print("simulating...")
    data = simulate(args.n, seed=args.seed)
    dn, mu, sd = normalise(data)
    real = data[:args.n_sample]
    np.save(tagdir / "real.npy", real)

    def denorm(s):
        return (s * sd + mu).astype(np.float32)

    results, finals = {}, {}

    print("\n=== joint ===")

    def joint_ckpt(sample, step, loss, k):
        m = evaluate(real[:k], denorm(sample(k)))
        m.update(epoch=step, loss=loss)
        print(f"  [joint] step {step:6d}  loss {loss:.4f}  "
              f"w1 {m['w1_mean']:.4f}  r_energy {m['r_energy']:.4f}  "
              f"r_dyn {m['r_dyn']:.4f}")
        return m

    joint_sampler, results["joint"] = fit(dn, 4, args, dev, "joint",
                                          on_ckpt=joint_ckpt)
    finals["joint"] = denorm(joint_sampler(args.n_sample))

    P = projection_matrix()
    finals["joint_proj"] = project(finals["joint"], P)
    results["joint_proj"] = [{**evaluate(real, finals["joint_proj"]),
                              "epoch": args.steps}]

    print("\n=== independent (4 separate 1-channel models) ===")
    chans = []
    for c in range(4):
        s, _ = fit(dn[:, c:c + 1], 1, args, dev, f"indep-{c}", base=32)
        chans.append(s(args.n_sample))
    finals["independent"] = denorm(np.concatenate(chans, axis=1))
    results["independent"] = [{**evaluate(real, finals["independent"]),
                               "epoch": args.steps}]

    for k, v in finals.items():
        np.save(tagdir / f"samples_{k}.npy", v)
    with open(tagdir / "log.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    fl = results["joint"][-1]
    print("\n--- FINAL ---")
    print(f"  real-data floor  r_energy {fl['floor_r_energy']:.5f}  "
          f"r_dyn {fl['floor_r_dyn']:.5f}  r_kin {fl['floor_r_kin']:.5f}")
    for k in ("independent", "joint", "joint_proj"):
        m = results[k][-1]
        print(f"  {k:15s}  r_energy {m['r_energy']:.5f}  "
              f"r_dyn {m['r_dyn']:.5f}  r_kin {m['r_kin']:.5f}  "
              f"w1 {m['w1_mean']:.5f}")


if __name__ == "__main__":
    main()
