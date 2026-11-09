"""Experiment 2: how special is Conway's rule?

A Life-like rule is a choice of birth counts B and survival counts S. There
are 2**18 of them; leaving out the B0 rules (where empty space flashes on)
leaves 2**17 = 131,072. This script runs every one of them on the same
random soup and records how each world ends:

* dies out          - every cell is dead
* freezes           - settles into a still picture (period 1)
* oscillates        - settles into a repeating cycle (period > 1)
* never settles     - still changing after the time budget

plus the transient length (generations before settling), the final density
and the activity (fraction of cells flipping per generation). It also
runs Conway's rule and its 17 "one-change mutants" side by side.

    python experiments/rule_space.py              # full survey + plots
    python experiments/rule_space.py --sample 2000  # random subset
    python experiments/rule_space.py --plot       # plot saved results
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import FIGURES, RESULTS, load_json, parallel_map, save_json  # noqa: E402

from lifelab.analysis import find_cycles
from lifelab.engine import random_grid
from lifelab.rules import CONWAY, NAMED_RULES, Rule

SIZE = 64
DENSITY = 0.35
MAX_STEPS = 2000
SEED = 2026
CHUNK = 256
LONG_TRANSIENT = 300  # generations; "long-lived" rules churn at least this long before settling


def soup(size=SIZE, density=DENSITY, seed=SEED):
    return random_grid((size, size), density, np.random.default_rng(seed))


def survey_chunk(job):
    codes, max_steps = job
    start = soup()
    luts = np.stack([Rule.from_code(int(c)).lut for c in codes])
    grids = np.broadcast_to(start, (len(codes), *start.shape))
    res = find_cycles(grids, luts=luts, max_steps=max_steps)
    return dict(code=np.asarray(codes), transient=res.transient, period=res.period,
                density=res.density, activity=res.activity)


def classify(density, period):
    """Behaviour class from the final density and period (-1 = never settled)."""
    out = np.full(len(period), "never settles", dtype=object)
    out[period == 1] = "freezes"
    out[period > 1] = "oscillates"
    out[(period > 0) & (density == 0)] = "dies out"
    return out


def run(sample=None):
    codes = np.array([r.code for r in Rule.all_rules()])
    if sample:
        rng = np.random.default_rng(0)
        codes = np.sort(rng.choice(codes, sample, replace=False))
        codes = np.union1d(codes, [Rule.parse(r).code for r in NAMED_RULES.values() if "0" not in Rule.parse(r).notation.split("/")[0]])
    # Shuffle so chunks have a similar mix of fast and slow rules.
    order = np.random.default_rng(1).permutation(len(codes))
    chunks = [codes[order[i: i + CHUNK]] for i in range(0, len(codes), CHUNK)]
    start = time.time()
    parts = parallel_map(survey_chunk, [(c, MAX_STEPS) for c in chunks], label="rule chunks")
    print(f"surveyed {len(codes)} rules in {time.time() - start:.0f}s")
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(data["code"])
    data = {k: v[order] for k, v in data.items()}
    rules = [Rule.from_code(int(c)) for c in data["code"]]
    data["lam"] = np.array([r.langton_lambda for r in rules])
    name = "rule_space" if not sample else "rule_space_sample"
    np.savez_compressed(RESULTS / f"{name}.npz", **data)
    summarise(name)


def load(name="rule_space"):
    d = np.load(RESULTS / f"{name}.npz")
    return {k: d[k] for k in d.files}


def summarise(name="rule_space"):
    d = load(name)
    cls = classify(d["density"], d["period"])
    n = len(cls)
    summary = {"n_rules": n, "size": SIZE, "density": DENSITY, "max_steps": MAX_STEPS,
               "seed": SEED, "long_transient": LONG_TRANSIENT, "classes": {}}
    for c in ("dies out", "freezes", "oscillates", "never settles"):
        summary["classes"][c] = int((cls == c).sum())
    settled = d["period"] > 0
    long_lived = settled & (d["transient"] >= LONG_TRANSIENT) & (d["density"] > 0)
    summary["long_lived"] = int(long_lived.sum())
    summary["long_lived_sparse"] = int((long_lived & (d["density"] < 0.1)).sum())
    life = np.flatnonzero(d["code"] == CONWAY.code)
    if len(life):
        i = life[0]
        summary["life"] = dict(transient=int(d["transient"][i]), period=int(d["period"][i]),
                               density=float(d["density"][i]), activity=float(d["activity"][i]),
                               lam=float(d["lam"][i]), cls=str(cls[i]))
        # Life's percentile among settled rules by transient length
        summary["life"]["transient_rank_pct"] = float((d["transient"][settled] < d["transient"][i]).mean() * 100)
    sparse_long = np.flatnonzero(long_lived & (d["density"] < 0.1))
    top = sparse_long[np.argsort(-d["transient"][sparse_long])][:40]
    summary["longest_sparse"] = [dict(rule=Rule.from_code(int(d["code"][i])).notation,
                                      transient=int(d["transient"][i]), period=int(d["period"][i]),
                                      density=float(d["density"][i])) for i in top]
    named = {}
    for label, text in NAMED_RULES.items():
        code = Rule.parse(text).code
        j = np.flatnonzero(d["code"] == code)
        if len(j):
            j = j[0]
            named[label] = dict(rule=text, cls=str(cls[j]), transient=int(d["transient"][j]),
                                period=int(d["period"][j]), density=float(d["density"][j]),
                                activity=float(d["activity"][j]), lam=float(d["lam"][j]))
    summary["named"] = named
    # Behaviour by Langton's lambda
    bins = np.linspace(0, 1, 21)
    idx = np.clip(np.digitize(d["lam"], bins) - 1, 0, len(bins) - 2)
    summary["by_lambda"] = []
    for b in range(len(bins) - 1):
        sel = idx == b
        if not sel.any():
            continue
        row = dict(lo=float(bins[b]), hi=float(bins[b + 1]), n=int(sel.sum()))
        for c in ("dies out", "freezes", "oscillates", "never settles"):
            row[c] = float((cls[sel] == c).mean())
        row["long_lived"] = float(long_lived[sel].mean())
        summary["by_lambda"].append(row)
    save_json(summary, f"{name}.json")
    for c, k in summary["classes"].items():
        print(f"  {c:14s} {k:7d} ({100 * k / n:5.1f}%)")
    print(f"  long-lived (transient >= {LONG_TRANSIENT}): {summary['long_lived']} "
          f"({100 * summary['long_lived'] / n:.2f}%), sparse: {summary['long_lived_sparse']}")
    if "life" in summary:
        print("  Life:", summary["life"])


def mutant_worlds(size=128, steps=4000, seed=7):
    """Run Life and its 17 one-change mutants from the same soup."""
    start = random_grid((size, size), DENSITY, np.random.default_rng(seed))
    rules = [("Life", CONWAY)] + CONWAY.mutants()
    luts = np.stack([r.lut for _, r in rules])
    res = find_cycles(np.broadcast_to(start, (len(rules), size, size)), luts=luts, max_steps=steps)
    cls = classify(res.density, res.period)
    worlds = []
    for i, (label, rule) in enumerate(rules):
        worlds.append(dict(label=label, rule=rule.notation, name=rule.name, cls=str(cls[i]),
                           transient=int(res.transient[i]), period=int(res.period[i]),
                           density=float(res.density[i]), activity=float(res.activity[i])))
    return worlds, res.final, start


def plot(name="rule_space"):
    from plots import plot_rule_space

    worlds, finals, _ = mutant_worlds()
    save_json(worlds, "life_mutants.json")
    plot_rule_space(load(name), load_json(f"{name}.json"), worlds, finals,
                    FIGURES / ("rule_space.png" if name == "rule_space" else "rule_space_sample.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=None, help="survey a random subset of rules")
    parser.add_argument("--plot", action="store_true", help="only re-draw figures from saved results")
    args = parser.parse_args()
    name = "rule_space" if not args.sample else "rule_space_sample"
    if not args.plot:
        run(args.sample)
    try:
        plot(name)
    except ImportError as exc:
        print(f"(skipping plot: {exc})")
