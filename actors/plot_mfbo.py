#!/usr/bin/env python3
"""Plot an mfbo_hf run: high-fidelity runs (the real latentflow1.py evals)
vs. the GP's low-fidelity prediction curve + 1-sigma band.

Reads the two CSVs mfbo_hf writes on exit:
  mfbo_hf_runs.csv  -- one row per actual high-fidelity run
  mfbo_hf_pred.csv  -- the 50-point GP prediction grid

Usage:
  ../.venv/bin/python3 ./plot_mfbo.py [runs.csv] [pred.csv] [out.png]

x-axis is beta on a LOG scale (equivalently log10 beta). That is the space
the optimizer actually searches, and it keeps the small-beta decades
readable. The normalized u in [0,1] is an internal coordinate -- not shown.
"""
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs_csv = sys.argv[1] if len(sys.argv) > 1 else "mfbo_hf_runs.csv"
pred_csv = sys.argv[2] if len(sys.argv) > 2 else "mfbo_hf_pred.csv"
out_png  = sys.argv[3] if len(sys.argv) > 3 else "mfbo_hf.png"


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


runs = load(runs_csv)
pred = load(pred_csv)

# --- prediction curve (low fidelity) ---
pb  = np.array([float(r["beta"]) for r in pred])
pm  = np.array([float(r["pred_mmd"]) for r in pred])
pl  = np.array([float(r["pred_log10_mmd"]) for r in pred])
ps  = np.array([float(r["log10_mmd_sd"]) for r in pred])
band_lo = 10.0 ** (pl - ps)
band_hi = 10.0 ** (pl + ps)

# --- actual runs (high fidelity) ---
rb   = np.array([float(r["beta"]) for r in runs])
rm   = np.array([float(r["mmd"]) for r in runs])
rord = np.array([int(r["order"]) for r in runs])

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.fill_between(pb, band_lo, band_hi, alpha=0.18, color="tab:blue",
                label="GP ±1σ  (low-fidelity uncertainty)")
ax.plot(pb, pm, color="tab:blue", lw=2,
        label="GP mean  (low-fidelity prediction)")

sc = ax.scatter(rb, rm, c=rord, cmap="viridis", s=70, zorder=5,
                edgecolor="black", linewidth=0.6,
                label="high-fidelity runs (colour = order)")

best = int(np.argmin(rm))
ax.scatter([rb[best]], [rm[best]], marker="*", s=320, color="crimson",
           zorder=6, edgecolor="black", linewidth=0.6,
           label=f"best: β={rb[best]:.4g}, MMD={rm[best]:.3g}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("β  (log scale)")
ax.set_ylabel("MMD  (log scale)")
ax.set_title("mfbo_hf: high-fidelity runs vs GP low-fidelity prediction")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="best", fontsize=9)

cb = fig.colorbar(sc, ax=ax)
cb.set_label("run order (0 = first)")

fig.tight_layout()
fig.savefig(out_png, dpi=130)
print(f"wrote {out_png}  ({len(runs)} runs, {len(pred)} prediction points)")
