"""Experiment 1: how does a random world live and die?

Fill a grid at random, press play, and wait until it settles into a
repeating pattern. How long does that take, how much is left, and what is
the leftover "ash" made of?

1. Final density and lifetime against the initial density (128 x 128 torus).
2. Lifetime against grid size (16 ... 256) at 35 % initial density.
3. A census of the objects left behind on 256 x 256 worlds.

    python experiments/soups.py          # run + plot
    python experiments/soups.py --quick  # smoke test
    python experiments/soups.py --plot   # plot saved results
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np

from common import FIGURES, RESULTS, load_json, parallel_map, save_json  # noqa: E402

from lifelab.analysis import census, find_cycles
from lifelab.engine import random_grid

CONFIG = {
    "full": dict(
        densities=np.round(np.arange(0.05, 0.96, 0.05), 2),
        density_size=128,
        density_reps=32,
        sizes={16: 1024, 32: 512, 64: 256, 128: 96, 256: 48},
        size_density=0.35,
        max_steps=60_000,
        batch=16,
    ),
    "quick": dict(
        densities=np.array([0.1, 0.35, 0.6]),
        density_size=32,
        density_reps=4,
        sizes={16: 16, 32: 8},
        size_density=0.35,
        max_steps=5_000,
        batch=8,
    ),
}


def run_batch(job):
    kind, size, density, reps, max_steps, seed, do_census = job
    grids = random_grid((reps, size, size), density, np.random.default_rng(seed))
    res = find_cycles(grids, max_steps=max_steps)
    counts = Counter()
    if do_census:
        for g, settled in zip(res.final, res.settled):
            if settled:
                counts.update(census(g))
    return dict(kind=kind, size=size, initial_density=float(density),
                transient=res.transient.tolist(), period=res.period.tolist(),
                final_density=res.density.tolist(), census=dict(counts))


def run(mode):
    cfg = CONFIG[mode]
    jobs, seed = [], 9000
    for p in cfg["densities"]:
        reps = cfg["density_reps"]
        for start in range(0, reps, cfg["batch"]):
            seed += 1
            jobs.append(("density", cfg["density_size"], float(p), min(cfg["batch"], reps - start),
                         cfg["max_steps"], seed, False))
    for size, reps in cfg["sizes"].items():
        per = max(cfg["batch"], 4096 // size)  # bigger batches for small worlds
        for start in range(0, reps, per):
            seed += 1
            jobs.append(("size", size, cfg["size_density"], min(per, reps - start), cfg["max_steps"],
                         seed, size == max(cfg["sizes"])))
    jobs.sort(key=lambda j: -j[1])
    start = time.time()
    parts = parallel_map(run_batch, jobs, label="soup batches")
    print(f"done in {time.time() - start:.0f}s")
    out = {"config": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cfg.items()},
           "density_sweep": {}, "size_sweep": {}, "census": Counter()}
    for part in parts:
        key = "density_sweep" if part["kind"] == "density" else "size_sweep"
        k = str(part["initial_density"] if part["kind"] == "density" else part["size"])
        slot = out[key].setdefault(k, dict(transient=[], period=[], final_density=[]))
        for f in ("transient", "period", "final_density"):
            slot[f].extend(part[f])
        out["census"].update(part["census"])
    out["census"] = dict(out["census"].most_common())
    name = "soups" if mode == "full" else "soups_quick"
    save_json(out, f"{name}.json")
    summarise(name)


def summarise(name="soups"):
    d = load_json(f"{name}.json")
    print("initial density -> final density, median lifetime (unsettled)")
    for k, v in d["density_sweep"].items():
        t = np.array(v["transient"])
        print(f"  {float(k):.2f} -> {np.mean(v['final_density']):.4f}, "
              f"median T = {np.median(t[t >= 0]) if (t >= 0).any() else float('nan'):.0f} ({(t < 0).sum()})")
    print("grid size -> median lifetime")
    for k, v in d["size_sweep"].items():
        t = np.array(v["transient"])
        print(f"  {k:>4s}: median T = {np.median(t[t >= 0]):.0f}, mean = {t[t >= 0].mean():.0f}, "
              f"unsettled = {(t < 0).sum()} / {len(t)}")
    total = sum(d["census"].values())
    print(f"census: {total} objects")
    for name_, n in list(d["census"].items())[:15]:
        print(f"  {name_:24s} {n:6d}  {100 * n / total:5.1f}%")


def plot(name="soups"):
    from plots import plot_soups

    plot_soups(load_json(f"{name}.json"), FIGURES / ("soups.png" if name == "soups" else "soups_quick.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    mode = "quick" if args.quick else "full"
    name = "soups" if mode == "full" else "soups_quick"
    if not args.plot:
        run(mode)
    try:
        plot(name)
    except ImportError as exc:
        print(f"(skipping plot: {exc})")
