#!/usr/bin/env python3
"""BoTorch reference baseline for the beta search mfbo_actors.cc runs.

Same task, same search space, same batch/round schedule as mfbo_actors.cc:
every evaluation is a full latentflow1.py run (no cheap tier), 3 seed
points + 10 rounds of 3 candidates each = 33 full runs, log10(beta) in
[-8, 1] normalized to u in [0,1]. The only thing this script swaps out is
*how* the surrogate and acquisition are computed:

  mfbo_actors.cc (custom C++)            mfbo_botorch.py (this file)
  ----------------------------       ----------------------------
  mfbo::MFBO::gp() -- hand-rolled    botorch.models.SingleTaskGP, fit via
  RBF kernel + Cholesky solve        fit_gpytorch_mll (GPyTorch under the
                                      hood)
  a(u) = best - mean(u) + std(u)     qLogExpectedImprovement (default) or
  (ad hoc exploit/explore score)     qKnowledgeGradient (--acquisition kg)
  propose_batch(): rank candidates   optimize_acqf(..., q=batch): jointly
  on a 200-pt grid, greedily keep    optimizes q points at once against a
  ones >= min_gap apart              batch acquisition function, which
                                      values the q points jointly instead
                                      of by a hand-tuned distance rule
  CAF manager/worker actors, one     concurrent.futures.ThreadPoolExecutor,
  std::thread per job, round-robin   one thread per job, same round-robin
  CUDA_VISIBLE_DEVICES assignment    CUDA_VISIBLE_DEVICES assignment
  bo.record(u, fidelity, y)          train_X / train_Y tensors, GP refit
                                      from scratch each round

Run:
  ../.venv/bin/python3 ./mfbo_botorch.py --workers 3 --gpus 2 --acquisition ei
"""
import argparse
import itertools
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
from botorch.acquisition import qKnowledgeGradient, qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# ----- search space: identical mapping to mfbo_actors.cc / mfbo.cc -------

LOG10_BETA_LO = -8.0
LOG10_BETA_HI = 1.0


def log10_beta(u: float) -> float:
    return LOG10_BETA_LO + (LOG10_BETA_HI - LOG10_BETA_LO) * u


def denorm(u: float) -> float:
    return 10.0 ** log10_beta(u)


# ----- one full latentflow1.py run, memoized like mfbo_actors.cc's cache ------

_mmd_cache: dict[float, float] = {}


def run_latentflow(beta: float, gpu_id: int | None, tag_suffix: str) -> float:
    key = round(beta * 1e8)
    if key in _mmd_cache:
        return _mmd_cache[key]

    env = None
    if gpu_id is not None:
        import os
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "../.venv/bin/python3", "./latentflow1.py", "--quiet",
        "--beta", str(beta),
        "--tag-suffix", tag_suffix,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"latentflow1.py failed (rc={result.returncode}) for beta={beta:.4e}",
              file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    mmd = float(result.stdout.strip())
    _mmd_cache[key] = mmd
    return mmd


# ----- "round": dispatch a batch of jobs to a thread pool, like mfbo_actors's -
# ----- manager actor dispatching to worker actors, one std::thread each. --

def run_round(us: list[float], num_gpus: int, tag_suffix: str) -> list[float]:
    """Evaluate every u in `us` concurrently, GPU-round-robined. Returns
    the measured MMD for each u, in the same order as `us`."""
    gpu_cycle = itertools.cycle(range(num_gpus)) if num_gpus > 0 else itertools.repeat(None)
    with ThreadPoolExecutor(max_workers=len(us)) as pool:
        futures = [
            pool.submit(run_latentflow, denorm(u), next(gpu_cycle), tag_suffix)
            for u in us
        ]
        return [f.result() for f in futures]


# ----- GP surrogate + acquisition -----------------------------------------
# train_Y stores -log10(mmd) so BoTorch's maximize-by-default acquisition
# functions look for the *lowest* MMD, matching mfbo::MFBO minimizing
# log10(mmd) directly.

def fit_model(train_X: torch.Tensor, train_Y: torch.Tensor) -> SingleTaskGP:
    model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def propose_batch(model: SingleTaskGP, train_Y: torch.Tensor, batch: int,
                   acquisition: str) -> torch.Tensor:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    if acquisition == "kg":
        # One-step lookahead: explicitly estimates the expected improvement
        # to the posterior mean at the incumbent after observing the
        # candidate. Principled, but far more expensive per proposal than
        # qLogExpectedImprovement (nested fantasy-model optimization) --
        # this is the acquisition function referenced in the paper's
        # "Validation" section.
        acqf = qKnowledgeGradient(model, num_fantasies=32)
    else:
        # qLogExpectedImprovement: the standard, cheap batch acquisition
        # function. Jointly scores a q-batch against the *joint* posterior,
        # which is what gives batch diversity here instead of mfbo_actors.cc's
        # hand-written min-gap rule -- two near-duplicate candidates add
        # almost no joint improvement over either alone, so the optimizer
        # naturally spreads the batch out.
        acqf = qLogExpectedImprovement(model, best_f=train_Y.max())

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=batch,
        num_restarts=10,
        raw_samples=256,
    )
    return candidates.detach().squeeze(-1)  # shape (batch,)


# ----- driver ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3,
                     help="candidates per round / parallel threads (default 3, matches mfbo_actors)")
    ap.add_argument("--gpus", type=int, default=2,
                     help="GPUs to round-robin workers across, 0 disables pinning (default 2)")
    ap.add_argument("--rounds", type=int, default=10,
                     help="acquisition rounds after the seed (default 10, matches mfbo_actors)")
    ap.add_argument("--acquisition", choices=["ei", "kg"], default="ei",
                     help="qLogExpectedImprovement (fast, default) or qKnowledgeGradient (slow, principled)")
    args = ap.parse_args()

    batch = max(2, args.workers)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wall_start = time.monotonic()
    print(f"mfbo_botorch run {ts} started (acquisition={args.acquisition})", file=sys.stderr)

    us: list[float] = []
    mmds: list[float] = []
    order: list[int] = []
    approx_before: list[float] = []  # GP's pred for each u, using the model as of *start of round*

    def record_round(round_us: list[float], round_mmds: list[float], before: list[float]):
        start = len(us)
        us.extend(round_us)
        mmds.extend(round_mmds)
        approx_before.extend(before)
        order.extend(range(start, start + len(round_us)))
        for u, mmd in zip(round_us, round_mmds):
            print(f"log10_beta={log10_beta(u):.4f}  beta={denorm(u):.4e}  MMD={mmd:.4e}",
                  file=sys.stderr)

    # Seed round: same spread-of-batch-points seed as mfbo_actors.cc, no model yet.
    seed_us = [i / (batch - 1) if batch > 1 else 0.5 for i in range(batch)]
    seed_mmds = run_round(seed_us, args.gpus, "_botorch")
    record_round(seed_us, seed_mmds, [float("nan")] * batch)

    train_X = torch.tensor(us, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.tensor(
        [-torch.log10(torch.tensor(max(m, 1e-6))).item() for m in mmds],
        dtype=torch.double,
    ).unsqueeze(-1)

    for r in range(args.rounds):
        model = fit_model(train_X, train_Y)

        round_us = propose_batch(model, train_Y, batch, args.acquisition).tolist()
        round_us = [min(max(u, 0.0), 1.0) for u in round_us]  # numerical safety

        # "before" prediction: what the round-start model thinks each
        # candidate will score, purely informational (matches the
        # gp_pred_log10_mmd_before column mfbo_actors.cc's CSV writes).
        with torch.no_grad():
            post = model.posterior(torch.tensor(round_us, dtype=torch.double).unsqueeze(-1))
            before = (-post.mean.squeeze(-1)).tolist()

        print(f"--- round {r} : {len(round_us)} full runs ---", file=sys.stderr)
        round_mmds = run_round(round_us, args.gpus, "_botorch")
        record_round(round_us, round_mmds, before)

        train_X = torch.tensor(us, dtype=torch.double).unsqueeze(-1)
        train_Y = torch.tensor(
            [-torch.log10(torch.tensor(max(m, 1e-6))).item() for m in mmds],
            dtype=torch.double,
        ).unsqueeze(-1)

    best_i = min(range(len(mmds)), key=lambda i: mmds[i])
    print(f"\nbest  log10_beta = {log10_beta(us[best_i]):.4f}  "
          f"beta = {denorm(us[best_i]):.4e}  MMD = {mmds[best_i]:.4e}")

    # Final model + 50-point prediction grid, same shape as mfbo_actors.cc's CSV.
    final_model = fit_model(train_X, train_Y)
    grid = [i / 49.0 for i in range(50)]
    with torch.no_grad():
        post = final_model.posterior(torch.tensor(grid, dtype=torch.double).unsqueeze(-1))
        pred_log10_mmd = (-post.mean.squeeze(-1)).tolist()
        pred_sd = post.variance.squeeze(-1).sqrt().tolist()

    runs_path = Path(f"mfbo_botorch_runs_{ts}.csv")
    pred_path = Path(f"mfbo_botorch_pred_{ts}.csv")

    with runs_path.open("w") as f:
        f.write("order,log10_beta,beta,mmd,log10_mmd,gp_pred_log10_mmd_before\n")
        for i, u in enumerate(us):
            import math
            log10_mmd = math.log10(max(mmds[i], 1e-6))
            f.write(f"{order[i]},{log10_beta(u)},{denorm(u)},{mmds[i]},{log10_mmd},{approx_before[i]}\n")

    with pred_path.open("w") as f:
        f.write("log10_beta,beta,pred_mmd,pred_log10_mmd,log10_mmd_sd\n")
        for i, u in enumerate(grid):
            f.write(f"{log10_beta(u)},{denorm(u)},{10.0 ** pred_log10_mmd[i]},"
                     f"{pred_log10_mmd[i]},{pred_sd[i]}\n")

    for src, dst in [(runs_path, Path("mfbo_botorch_runs.csv")),
                      (pred_path, Path("mfbo_botorch_pred.csv"))]:
        dst.write_text(src.read_text())
    print(f"\nwrote {runs_path}, {pred_path} "
          f"(and mfbo_botorch_runs.csv / mfbo_botorch_pred.csv)")

    png_path = f"mfbo_botorch_{ts}.png"
    rc = subprocess.run(
        ["../.venv/bin/python3", "./plot_mfbo.py", str(runs_path), str(pred_path),
         png_path, "mfbo_botorch"],
    ).returncode
    if rc == 0:
        Path(png_path).replace(png_path)  # no-op, keeps symmetry with the C++ drivers
        import shutil
        shutil.copy(png_path, "mfbo_botorch.png")
        print(f"wrote {png_path} (and mfbo_botorch.png)")
    else:
        print(f"plot_mfbo.py exited with {rc} -- CSVs are still there, plot them manually")

    secs = time.monotonic() - wall_start
    print(f"mfbo_botorch run {ts} finished in {secs:.0f} s ({len(us)} full runs)")


if __name__ == "__main__":
    main()
