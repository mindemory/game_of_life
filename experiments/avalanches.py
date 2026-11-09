"""Experiment 3: is the Game of Life critical?

Bak, Chen & Creutz (1989) proposed that Life "self-organises" into a
critical state: poke a settled world and the disturbance (an avalanche)
can be any size, with power-law statistics like sandpiles or earthquakes.
Bennett & Bourzutschky (1991) and Blok & Bergersen (1997) argued that the
power law has a cutoff, so Life is only *near* critical.

Protocol (Blok & Bergersen 1997):

1. Start from a random soup (50 % alive) and let it settle.
2. Flip one random cell, chosen among live cells and their neighbours.
3. Run until the world repeats a state. The avalanche *lifetime* T is the
   number of generations before it entered its new cycle. The avalanche
   *area* A is the number of cells that ever differed from the world it
   would have been without the flip.
4. Repeat from the new settled state, many thousands of times.

This is done on square grids of several sizes, with wrap-around (torus) and
with dead ("cold") boundaries.

    python experiments/avalanches.py            # run + plot
    python experiments/avalanches.py --quick    # small smoke test
    python experiments/avalanches.py --plot     # plot saved results only
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy import optimize

from common import FIGURES, RESULTS, load_json, parallel_map, save_json  # noqa: E402  (sets sys.path)

from lifelab.analysis import state_key
from lifelab.engine import neighbor_counts, random_grid, step

RULE = "B3/S23"
MAX_BACKGROUND_PERIOD = 64


def settle(grid, boundary, max_steps=500_000):
    """Run until a state repeats. Returns (state on the cycle, period)."""
    seen = {state_key(grid): 0}
    for t in range(1, max_steps + 1):
        grid = step(grid, RULE, boundary)
        key = state_key(grid)
        if key in seen:
            return grid, t - seen[key]
        seen[key] = t
    return grid, None


def cycle_states(grid, period, boundary):
    """All states of the cycle starting at ``grid`` (None if the cycle is long)."""
    if period is None or period > MAX_BACKGROUND_PERIOD:
        return None
    states = [grid]
    for _ in range(period - 1):
        states.append(step(states[-1], RULE, boundary))
    return np.stack(states)


def run_chain(job):
    """One chain of avalanches on one grid. Returns per-avalanche arrays."""
    size, boundary, n_avalanches, seed, max_steps = job
    rng = np.random.default_rng(seed)
    grid, period = settle(random_grid((size, size), 0.5, rng), boundary)
    background = cycle_states(grid, period, boundary)
    lifetimes = np.zeros(n_avalanches, np.int64)
    areas = np.full(n_avalanches, -1, np.int64)
    periods = np.zeros(n_avalanches, np.int64)
    densities = np.zeros(n_avalanches)
    censored = np.zeros(n_avalanches, bool)
    restarts = np.zeros(n_avalanches, bool)
    for a in range(n_avalanches):
        if not grid.any():
            # Single flips cannot bring an empty world back to life, so a small
            # world that dies out is restarted from a fresh soup (and flagged).
            restarts[a] = True
            grid, period = settle(random_grid((size, size), 0.5, rng), boundary)
            background = cycle_states(grid, period, boundary)
        near = (grid > 0) | (neighbor_counts(grid, boundary) > 0)
        candidates = np.flatnonzero(near)
        site = rng.choice(candidates) if len(candidates) else rng.integers(grid.size)
        x = grid.copy()
        x.flat[site] ^= 1
        seen = {state_key(x): 0}
        touched = (x != background[0]) if background is not None else None
        lifetime = new_period = None
        for t in range(1, max_steps + 1):
            x = step(x, RULE, boundary)
            if background is not None:
                touched |= x != background[t % len(background)]
            key = state_key(x)
            first = seen.get(key)
            if first is not None:
                lifetime, new_period = first, t - first
                break
            seen[key] = t
        if lifetime is None:  # very rare: give up on this one and let it settle
            censored[a] = True
            lifetime = max_steps
            x, new_period = settle(x, boundary)
        lifetimes[a] = lifetime
        periods[a] = new_period or -1
        if touched is not None:
            areas[a] = int(touched.sum())
        grid = x
        densities[a] = grid.mean()
        background = cycle_states(grid, new_period, boundary)
    return dict(size=size, boundary=boundary, seed=seed, lifetime=lifetimes, area=areas,
                period=periods, density=densities, censored=censored, restart=restarts)


# ----------------------------------------------------------------- fitting
def fit_cutoff_power_law(t, tmin, x0=None):
    """Maximum-likelihood fit of a power law with an exponential cutoff.

    Uses the survival function of Blok & Bergersen (1997):
    ``P(T >= t) = (t / tmin)^(1-b) * exp(-(t - tmin) / tc)`` for ``t >= tmin``.
    Returns ``(b, tc, loglik, n)``.
    """
    t = np.asarray(t, float)
    t = t[t >= tmin]

    def nll(params):
        b, log_tc = params
        tc = np.exp(log_tc)
        if b <= 1.0:
            return np.inf
        log_s = (1 - b) * np.log(t / tmin) - (t - tmin) / tc
        hazard = (b - 1) / t + 1 / tc
        return -(log_s + np.log(hazard)).sum()

    starts = [x0] if x0 is not None else [[b0, np.log(tc0)] for b0 in (1.2, 1.5, 2.0)
                                           for tc0 in (100.0, 1000.0, 10000.0)]
    best = None
    for start in starts:
        res = optimize.minimize(nll, start, method="Nelder-Mead",
                                options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
        if best is None or res.fun < best.fun:
            best = res
    b, log_tc = best.x
    return float(b), float(np.exp(log_tc)), float(-best.fun), int(len(t))


def fit_pure_power_law(t, tmin):
    """MLE of a pure power law (continuous approximation): b, loglik."""
    t = np.asarray(t, float)
    t = t[t >= tmin]
    b = 1 + len(t) / np.log(t / tmin).sum()
    loglik = len(t) * np.log((b - 1) / tmin) - b * np.log(t / tmin).sum()
    return float(b), float(loglik)


def bootstrap(t, tmin, n_boot=200, seed=0):
    """95% percentile intervals for (b, tc) from resampled avalanches."""
    rng = np.random.default_rng(seed)
    t = np.asarray(t)
    b, tc, _, _ = fit_cutoff_power_law(t, tmin)
    x0 = [b, np.log(tc)]
    stats = [fit_cutoff_power_law(rng.choice(t, len(t)), tmin, x0)[:2] for _ in range(n_boot)]
    stats = np.array(stats)
    return np.percentile(stats, [2.5, 97.5], axis=0)


# ------------------------------------------------------------------ main
CONFIG = {
    "full": dict(
        sizes={64: 30_000, 128: 20_000, 256: 12_000, 512: 4_000},
        chains={64: 4, 128: 6, 256: 8, 512: 8},
        boundaries=("wrap", "dead"),
        burn_in=50,
        max_steps=200_000,
    ),
    "quick": dict(
        sizes={32: 400, 64: 200},
        chains={32: 2, 64: 2},
        boundaries=("wrap", "dead"),
        burn_in=10,
        max_steps=20_000,
    ),
}


def run(mode):
    cfg = CONFIG[mode]
    jobs = []
    seed = 1000
    for size, total in cfg["sizes"].items():
        for boundary in cfg["boundaries"]:
            n_chain = cfg["chains"][size]
            for c in range(n_chain):
                seed += 1
                jobs.append((size, boundary, total // n_chain, seed, cfg["max_steps"]))
    # Big grids first so the slowest jobs start early.
    jobs.sort(key=lambda j: -j[0])
    start = time.time()
    chains = parallel_map(run_chain, jobs, label="avalanche chains")
    print(f"simulated {sum(len(c['lifetime']) for c in chains)} avalanches in {time.time() - start:.0f}s")
    arrays = {}
    for key in ("lifetime", "area", "period", "density", "censored", "restart"):
        arrays[key] = np.concatenate([c[key] for c in chains])
    arrays["size"] = np.concatenate([np.full(len(c["lifetime"]), c["size"]) for c in chains])
    arrays["boundary"] = np.concatenate([np.full(len(c["lifetime"]), c["boundary"] == "dead") for c in chains])
    arrays["chain"] = np.concatenate([np.full(len(c["lifetime"]), i) for i, c in enumerate(chains)])
    arrays["index"] = np.concatenate([np.arange(len(c["lifetime"])) for c in chains])
    np.savez_compressed(RESULTS / f"avalanches{'' if mode == 'full' else '_quick'}.npz", **arrays)
    summarise(mode, cfg["burn_in"])


def load(mode="full"):
    data = np.load(RESULTS / f"avalanches{'' if mode == 'full' else '_quick'}.npz")
    return {k: data[k] for k in data.files}


def summarise(mode, burn_in, tmin=10):
    d = load(mode)
    keep = d["index"] >= burn_in
    summary = {"tmin": tmin, "burn_in": burn_in, "fits": []}
    for size in np.unique(d["size"]):
        for dead in (False, True):
            sel = keep & (d["size"] == size) & (d["boundary"] == dead)
            if not sel.any():
                continue
            t = d["lifetime"][sel]
            b, tc, ll_cut, n_tail = fit_cutoff_power_law(t, tmin)
            b_pure, ll_pure = fit_pure_power_law(t, tmin)
            ci = bootstrap(t, tmin, n_boot=100 if mode == "full" else 10)
            area = d["area"][sel]
            area = area[area >= 0]
            entry = dict(
                size=int(size), boundary="dead" if dead else "wrap", n=int(sel.sum()), n_tail=n_tail,
                b=b, tc=tc, b_ci=ci[:, 0].tolist(), tc_ci=ci[:, 1].tolist(),
                b_pure=b_pure, loglik_ratio=ll_cut - ll_pure,
                mean_lifetime=float(t.mean()), median_lifetime=float(np.median(t)),
                max_lifetime=int(t.max()), censored=int(d["censored"][sel].sum()),
                mean_area=float(area.mean()) if len(area) else None,
                density=float(d["density"][sel].mean()),
                restarts=int(d["restart"][sel].sum()) if "restart" in d else 0,
            )
            summary["fits"].append(entry)
            print(f"L={size:4d} {entry['boundary']:4s} n={entry['n']:6d}  b={b:.3f} "
                  f"[{ci[0,0]:.3f},{ci[1,0]:.3f}]  tc={tc:8.1f} [{ci[0,1]:.0f},{ci[1,1]:.0f}]  "
                  f"mean T={t.mean():7.1f}  max T={t.max()}  density={entry['density']:.4f}")
    save_json(summary, f"avalanches{'' if mode == 'full' else '_quick'}.json")


def plot(mode="full"):
    from plots import plot_avalanches

    plot_avalanches(load(mode), load_json(f"avalanches{'' if mode == 'full' else '_quick'}.json"),
                    FIGURES / ("avalanches.png" if mode == "full" else "avalanches_quick.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tiny run for testing")
    parser.add_argument("--plot", action="store_true", help="only re-draw figures from saved results")
    parser.add_argument("--summarise", action="store_true", help="only refit saved results")
    args = parser.parse_args()
    mode = "quick" if args.quick else "full"
    if args.summarise:
        summarise(mode, CONFIG[mode]["burn_in"])
    elif not args.plot:
        run(mode)
    if not args.summarise:
        try:
            plot(mode)
        except ImportError as exc:  # plotting module not written yet
            print(f"(skipping plot: {exc})")
