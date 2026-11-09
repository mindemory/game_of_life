"""Shared helpers for the experiment scripts (paths, parallel runs, saving)."""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# Make `import lifelab` work even without `pip install -e .`
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def n_workers() -> int:
    """Worker processes to use (override with the LIFELAB_WORKERS variable)."""
    default = max(1, (os.cpu_count() or 2) - 1)
    return int(os.environ.get("LIFELAB_WORKERS", default))


def parallel_map(fn: Callable, jobs: Iterable, label: str = "jobs") -> list:
    """Run ``fn`` over ``jobs`` in worker processes, keeping the input order."""
    jobs = list(jobs)
    start = time.time()
    out = [None] * len(jobs)
    with ProcessPoolExecutor(n_workers()) as pool:
        futures = {pool.submit(fn, job): i for i, job in enumerate(jobs)}
        for done, fut in enumerate(_as_completed(futures), start=1):
            out[futures[fut]] = fut.result()
            print(f"  [{label}] {done}/{len(jobs)} done ({time.time() - start:.0f}s)", flush=True)
    return out


def _as_completed(futures):
    from concurrent.futures import as_completed

    return as_completed(futures)


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(obj, name: str) -> Path:
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2, cls=_Encoder) + "\n")
    print(f"saved {path.relative_to(ROOT)}")
    return path


def load_json(name: str):
    return json.loads((RESULTS / name).read_text())
