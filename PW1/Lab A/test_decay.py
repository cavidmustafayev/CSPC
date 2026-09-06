"""
Tests for the decay simulation.

One complete test is given as a model. Add the two tests described in the
lab handout (a negative-rate test, and a test against the analytical law).
Run with:  pytest -v
"""

import numpy as np
import pytest
from decay import simulate, simulate_loop


def test_starts_at_N0():
    # at time zero, no atoms have decayed yet
    assert simulate(1000, 0.4)[0] == 1000


# TODO 1: test_rejects_negative_rate
def test_rejects_negative_rate():
    with pytest.raises(ValueError):
        simulate(1000, -0.4)


# TODO 2: test_matches_law
def test_matches_law():
    N0 = 1000
    lam = 0.4
    dt = 0.05  # matching step size in decay.py
    t = 1.0    # 20 steps * 0.05 = 1.0s
    idx = 20

    runs = [simulate(N0, lam)[idx] for _ in range(500)]
    avg_N = np.mean(runs)

    # Group explicitly with parentheses: ((1 - (lam * dt)) ** idx)
    p_step = lam * dt
    expected = N0 * ((1 - p_step) ** idx)

    assert avg_N == pytest.approx(expected, rel=0.05)