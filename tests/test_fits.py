"""The fitting routines used by the experiments recover known parameters."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from asynchrony import critical_fit  # noqa: E402
from avalanches import fit_cutoff_power_law, fit_pure_power_law  # noqa: E402


def sample_cutoff_power_law(b, tc, tmin, n, rng):
    """Inverse-transform samples from P(T >= t) = (t/tmin)^(1-b) exp(-(t-tmin)/tc)."""
    grid = np.logspace(np.log10(tmin), np.log10(tmin + 60 * tc), 200_000)
    survival = (grid / tmin) ** (1 - b) * np.exp(-(grid - tmin) / tc)
    u = rng.random(n)
    return np.interp(u, survival[::-1], grid[::-1])


def test_cutoff_power_law_recovers_parameters():
    rng = np.random.default_rng(0)
    t = sample_cutoff_power_law(1.35, 1500.0, 10, 40_000, rng)
    b, tc, _, n = fit_cutoff_power_law(t, 10)
    assert n == len(t)
    assert b == pytest.approx(1.35, abs=0.02)
    assert tc == pytest.approx(1500, rel=0.12)


def test_pure_power_law_recovers_exponent():
    rng = np.random.default_rng(1)
    t = 10 * (1 - rng.random(50_000)) ** (-1 / 0.5)  # b = 1.5, tmin = 10
    b, _ = fit_pure_power_law(t, 10)
    assert b == pytest.approx(1.5, abs=0.02)


def test_critical_fit_recovers_transition():
    sigma = np.linspace(0.5, 1.0, 81)
    rho = 0.027 + 0.6 * np.clip(0.906 - sigma, 0, None) ** 0.6
    rho += np.random.default_rng(2).normal(0, 0.001, len(rho))
    fit = critical_fit(sigma, rho)
    assert fit["sigma_c"] == pytest.approx(0.906, abs=0.004)
    assert fit["beta"] == pytest.approx(0.6, abs=0.08)
    assert fit["rho0"] == pytest.approx(0.027, abs=0.002)
