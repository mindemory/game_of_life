"""Experiment 4: what if the clock is not perfect?

Conway's Life updates every cell at exactly the same moment. Real systems
(neurons, cells, people) rarely share a perfect clock. Here each cell is
updated with probability sigma at every tick ("synchrony"); sigma = 1 is
ordinary Life.

Blok & Bergersen (1999) found a sharp phase transition: below a critical
synchrony sigma_c of about 0.906 Life no longer freezes into sparse debris but
turns into a dense, endlessly churning "labyrinth". This script measures the
steady-state density and activity across sigma for several grid sizes and
fits the critical point.

    python experiments/asynchrony.py          # run + plot
    python experiments/asynchrony.py --quick  # smoke test
    python experiments/asynchrony.py --plot   # plot saved results
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy import optimize

from common import FIGURES, RESULTS, load_json, parallel_map, save_json  # noqa: E402

from lifelab.engine import random_grid, step

CONFIG = {
    "full": dict(
        sigmas=np.round(np.r_[np.arange(0.50, 0.85, 0.05), np.arange(0.85, 0.95, 0.005),
                              np.arange(0.95, 1.0001, 0.01)], 4),
        sizes={64: 8, 128: 4, 256: 2},
        steps=8000,
        measure=3000,
        record_every=20,
    ),
    # Longer runs near the transition, to check how much finite run time biases sigma_c.
    "refine": dict(
        sigmas=np.round(np.r_[np.arange(0.75, 0.89, 0.01), np.arange(0.89, 0.9301, 0.0025)], 4),
        sizes={256: 2},
        steps=30000,
        measure=10000,
        record_every=50,
    ),
    "quick": dict(
        sigmas=np.array([0.6, 0.85, 0.9, 0.95, 1.0]),
        sizes={32: 2},
        steps=600,
        measure=200,
        record_every=20,
    ),
}


def run_point(job):
    sigma, size, replicas, steps, measure, every, seed = job
    rng = np.random.default_rng(seed)
    grids = random_grid((replicas, size, size), 0.5, rng)
    cells = size * size
    density_ts, activity_ts = [], []
    dens_acc = np.zeros(replicas)
    act_acc = np.zeros(replicas)
    for t in range(1, steps + 1):
        new = step(grids, "B3/S23", "wrap", synchrony=sigma, rng=rng)
        if t > steps - measure or t % every == 0:
            flips = (new != grids).reshape(replicas, -1).sum(1) / cells
            dens = new.reshape(replicas, -1).sum(1) / cells
            if t > steps - measure:
                dens_acc += dens
                act_acc += flips
            if t % every == 0:
                density_ts.append(dens.mean())
                activity_ts.append(flips.mean())
        grids = new
    return dict(sigma=float(sigma), size=size, replicas=replicas,
                density=(dens_acc / measure).tolist(),
                activity=(act_acc / measure).tolist(),  # flips per cell per tick
                density_ts=density_ts, activity_ts=activity_ts)


def critical_fit(sig, rho):
    """Fit rho = rho0 + A * (sigma_c - sigma)^beta for sigma < sigma_c (rho0 otherwise)."""
    sig = np.asarray(sig, float)
    rho = np.asarray(rho, float)

    def model(s, rho0, amp, sc, beta):
        return rho0 + amp * np.clip(sc - s, 0, None) ** beta

    sel = sig >= 0.75
    top = sig >= np.quantile(sig, 0.8)  # the frozen side, wherever the sweep ends
    p0 = [float(np.clip(np.median(rho[top]), 0, 0.099)), 0.5, 0.91, 0.6]
    params, cov = optimize.curve_fit(model, sig[sel], rho[sel], p0=p0,
                                     bounds=([0, 0, 0.85, 0.1], [0.1, 10, 0.99, 2.0]), maxfev=20000)
    err = np.sqrt(np.diag(cov))
    return dict(rho0=params[0], amp=params[1], sigma_c=params[2], beta=params[3],
                rho0_err=err[0], sigma_c_err=err[2], beta_err=err[3])


def run(mode):
    cfg = CONFIG[mode]
    jobs = []
    seed = 500
    for size, reps in cfg["sizes"].items():
        for s in cfg["sigmas"]:
            seed += 1
            jobs.append((float(s), size, reps, cfg["steps"], cfg["measure"], cfg["record_every"], seed))
    jobs.sort(key=lambda j: -j[1])
    start = time.time()
    points = parallel_map(run_point, jobs, label="synchrony points")
    print(f"done in {time.time() - start:.0f}s")
    out = dict(steps=cfg["steps"], measure=cfg["measure"], record_every=cfg["record_every"],
               points=sorted(points, key=lambda p: (p["size"], p["sigma"])))
    name = {"full": "asynchrony", "quick": "asynchrony_quick", "refine": "asynchrony_refine"}[mode]
    save_json(out, f"{name}.json")
    summarise(name)


def summarise(name="asynchrony"):
    data = load_json(f"{name}.json")
    fits = {}
    for size in sorted({p["size"] for p in data["points"]}):
        pts = [p for p in data["points"] if p["size"] == size]
        sig = np.array([p["sigma"] for p in pts])
        rho = np.array([np.mean(p["density"]) for p in pts])
        act = np.array([np.mean(p["activity"]) for p in pts]) / sig  # flips per *updated* cell
        try:
            fit_rho = critical_fit(sig, rho)
            fit_act = critical_fit(sig, act)
        except (RuntimeError, ValueError) as exc:
            print(f"  L={size}: fit failed ({exc})")
            continue
        fits[str(size)] = dict(density=fit_rho, activity=fit_act)
        print(f"  L={size:4d}: sigma_c = {fit_rho['sigma_c']:.4f} +/- {fit_rho['sigma_c_err']:.4f} "
              f"(density), {fit_act['sigma_c']:.4f} +/- {fit_act['sigma_c_err']:.4f} (activity); "
              f"beta = {fit_rho['beta']:.3f}; frozen density = {fit_rho['rho0']:.4f}")
    data["fits"] = fits
    save_json(data, f"{name}.json")


def plot(name="asynchrony"):
    from plots import plot_asynchrony

    refine = load_json("asynchrony_refine.json") if name == "asynchrony" and (RESULTS / "asynchrony_refine.json").exists() else None
    plot_asynchrony(load_json(f"{name}.json"),
                    FIGURES / ("asynchrony.png" if name == "asynchrony" else "asynchrony_quick.png"), refine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--refine", action="store_true", help="long runs near the transition (L=256)")
    args = parser.parse_args()
    if args.refine:
        run("refine")
        raise SystemExit
    mode = "quick" if args.quick else "full"
    name = "asynchrony" if mode == "full" else "asynchrony_quick"
    if not args.plot:
        run(mode)
    try:
        plot(name)
    except ImportError as exc:
        print(f"(skipping plot: {exc})")
