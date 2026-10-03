# MFBO: tuning β with actor-parallel Bayesian optimization

This directory tunes the β-VAE + flow-matching pipeline's `beta`
hyperparameter by minimizing MMD (see `mfbo_technique.tex` for the full
write-up). It's a Gaussian-process Bayesian optimizer (`mfbo.h`) wrapped
in a handful of drivers that differ in how they run the actual training
jobs. **`mfbo_actors` is the main entry point** — it's the only driver
that trains in parallel across GPUs via CAF actors, and it's the one the
other drivers in this directory exist to be compared against.

If you just want to run the optimizer, skip to
["Running the main driver"](#4-running-the-main-driver-mfbo_actors).

## 1. Prerequisites: CAF (the C++ Actor Framework)

`mfbo_actors` (and `mfbo_caf`, `test_actor`) link against **CAF's >=1.0
API**. Most systems only ship an older CAF, so you'll likely need to
build it yourself first. This is a repo-wide, one-time setup already
documented in the [top-level README](../README.md#prerequisites-installing-the-caf-framework)
— follow that section, then come back here. (`mfbo`, the sequential
baseline, has no CAF dependency and skips this entirely.)

Once CAF is built, either pass its location on every `make` invocation:

```bash
make CAF_PREFIX=$HOME/caf-install CAF_LIBDIR=lib mfbo-actors
```

or edit the `CAF_PREFIX`/`CAF_LIBDIR` defaults at the top of
[`Makefile`](Makefile) so a plain `make` picks it up.

## 2. Python environment

Training jobs shell out to `../.venv/bin/python3 ./latentflow1.py`. The
venv at the repo root already has `torch`, `torchvision`, `numpy`, and
`matplotlib` (needed by every driver below to train and to render
graphs). The only extra dependency is for the BoTorch baseline:

```bash
../.venv/bin/pip install botorch   # only needed for mfbo_botorch.py
```

(pulls in `gpytorch`, `scipy`, and `scikit-learn` as transitive deps).

If you're using a different interpreter, swap `../.venv/bin/python3` for
your own path in `mfbo_actors.cc`, `mfbo.cc`, `mfbo_caf.cc`, and
`mfbo_botorch.py` before building.

## 3. Build

```bash
cd actors
make mfbo-actors   # just the main driver
# or
make all           # test_actor, mfbo, mfbo_caf, and mfbo_actors
```

This produces the `mfbo_actors` binary (and whichever others you asked
for) in this directory.

## 4. Running the main driver (`mfbo_actors`)

```bash
./mfbo_actors --workers 3 --gpus 2
```

What it does: searches `log10(beta)` over `[-8, 1]` (i.e. `beta` from
`1e-8` to `10`) with a Gaussian-process surrogate. It seeds with a batch
of full training runs spread across the range (batch size = `--workers`,
minimum 2), then runs 10 acquisition rounds — each round proposes a new
batch of that many candidate betas at once (spread out so they don't
duplicate each other or past runs), spawns one CAF worker actor per
candidate, round-robins each worker onto the next of `--gpus` GPUs via
`CUDA_VISIBLE_DEVICES`, and blocks until the whole round reports back
before fitting the next round. Total training runs = batch size × 11 (1
seed round + 10 acquisition rounds); with the defaults below that's
3 × 11 = 33 full runs.

| Flag | Default | Meaning |
|---|---|---|
| `--workers`, `-w` | `3` | candidates per round = parallel worker actors spawned per round |
| `--gpus`, `-g` | `2` | local GPUs to round-robin workers across (`0` disables pinning) |

(`--betas`, `--port`, `--host`, `--server-mode` also show up in `--help`
— they're inherited from the shared CLI config class used by `test_actor`
and `mfbo_caf`, and `mfbo_actors` ignores them.)

A full default run (33 full training runs, 3-way parallel) takes
somewhere in the 14–16 minute range on this machine, versus roughly 39
minutes for the same 33 runs one at a time (`mfbo`, below).

At the end it prints the best `beta`/MMD found, plus a 50-point predicted
MMD-vs-beta curve, to stdout/stderr.

### Generating the graph

`mfbo_actors` renders its own graph automatically on exit — no extra step
needed. It writes:

- `mfbo_actors_runs_<timestamp>.csv` — one row per real training run (β, MMD, GP's prediction just before that run)
- `mfbo_actors_pred_<timestamp>.csv` — the GP's final 50-point prediction curve (mean ± 1σ)
- `mfbo_actors_<timestamp>.png` — the plotted comparison of the two above

plus stable `mfbo_actors_runs.csv` / `mfbo_actors_pred.csv` /
`mfbo_actors.png` copies (no timestamp) that always point at the most
recent run, so you can glance at `mfbo_actors.png` without hunting for
the latest timestamp.

If you ever need to re-plot without re-running (e.g. after editing
`plot_mfbo.py`, or to compare two old CSVs), call the plotting script
directly:

```bash
../.venv/bin/python3 ./plot_mfbo.py mfbo_actors_runs.csv mfbo_actors_pred.csv out.png mfbo_actors
```

Arguments are positional and all optional: `[runs.csv] [pred.csv]
[out.png] [driver_name]` (the last is just the plot title). The x-axis is
`beta` on a log scale; points are colored by the order that run
completed, with the best result starred.

## 5. The other drivers (for comparison, not required)

| Driver | Build | What it is |
|---|---|---|
| `mfbo` | `make mfbo` / `./mfbo` | Sequential baseline: identical GP/acquisition logic and schedule as `mfbo_actors`, but no actors — one full run at a time, in-process. Exists to measure what parallelism buys you. `./mfbo --synthetic` instead runs a quick analytic test function (no training, no CAF) as a regression check on the GP/acquisition math itself. |
| `mfbo_caf` | `make mfbo-caf` / `./mfbo_caf --workers 4 --gpus 2` | True two-tier multi-fidelity variant: a genuine cheap/short training run as the low-fidelity tier (unlike `mfbo_actors`, where "low fidelity" is just a free GP prediction), AR1-corrected against full runs. 6 rounds, top ~1/4 of each batch promoted to a full run. Prints progress/results to the console only — it does not write CSVs or a graph. |
| `mfbo_botorch.py` | (no build) `../.venv/bin/python3 ./mfbo_botorch.py --workers 3 --gpus 2` | Runs the exact same task/schedule as `mfbo_actors`, but with BoTorch's `SingleTaskGP` + `qLogExpectedImprovement` instead of our hand-rolled GP/acquisition — a library-quality baseline. Writes `mfbo_botorch_*` CSVs/PNG the same way. |

`mfbo`, `mfbo_actors`, and `mfbo_botorch.py` all write CSVs in the same
schema (`mfbo_caf` doesn't), so `plot_mfbo.py` and the comparison tables
in `mfbo_technique.tex` work on any of those three.

## 6. Where results are written

Every driver's outputs land in this directory as `<driver>_runs*.csv`,
`<driver>_pred*.csv`, and `<driver>*.png`. Training itself also leaves
per-β artifacts here (`latent_vectors_beta<β>*.npy`,
`latent_labels_beta<β>*.npy`, etc.) tagged by β and driver so concurrent
runs never clobber each other.

See `mfbo_technique.tex` for the full method writeup and the actual
timing/quality comparison across all of the drivers above.
