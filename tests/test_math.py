"""Unit tests for the mathematical primitives of e-means.

These check the paper's equations against the *shipped* official coefficients
in ``faiss.contrib.e_means.base``. Where the shipped coefficients differ from
the paper's Table 2, we assert the shipped values and document the difference.
"""
import math

import numpy as np
import pytest

from faiss.contrib.e_means import base as B


def test_scheduled_alpha_endpoints():
    alpha0 = 0.4
    M = 100
    # Equation 5: alpha(0) = alpha0, alpha(M) = eps * alpha0 with eps = 0.1
    assert B._scheduled_alpha(alpha0, 0, M) == pytest.approx(alpha0, rel=1e-12)
    assert B._scheduled_alpha(alpha0, M, M) == pytest.approx(0.1 * alpha0, rel=1e-12)
    # midpoint: cos(pi/2) = 0 -> alpha0 * (eps + (1-eps)/2)
    mid = B._scheduled_alpha(alpha0, M // 2, M)
    assert mid == pytest.approx(alpha0 * (0.1 + 0.9 * 0.5), rel=1e-9)


def test_scheduled_alpha_matches_closed_form():
    alpha0, M = 0.37, 250
    for step in (0, 1, 7, 123, 249, 250):
        frac = min(step, M) / M
        expected = alpha0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        assert B._scheduled_alpha(alpha0, step, M) == pytest.approx(expected, rel=1e-12)


def test_round_power_of_two():
    assert B._round_power_of_two(1.0) == 1
    assert B._round_power_of_two(3.0) == 4
    assert B._round_power_of_two(5.0) == 4
    assert B._round_power_of_two(6.0) == 8
    assert B._round_power_of_two(1000.0) == 1024


def test_shipped_scaling_law_coefficients():
    # Shipped fits in faiss.contrib.e_means.base (differ from paper Table 2).
    assert B._BATCH_LAWS["throughput"] == (0.035134, 0.134749, 0.876473, 0.813344)
    assert B._BATCH_LAWS["quality"] == (0.00439175, 0.134749, 0.876473, 0.813344)
    assert B._ALPHA_LAW == (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)
    # quality law is exactly one eighth of throughput before rounding
    assert B._BATCH_LAWS["quality"][0] == pytest.approx(B._BATCH_LAWS["throughput"][0] / 8)


def test_step_law_clamped():
    # alpha0 is clamped to [1e-4, 1.0]
    for n, k, T, B_ in [
        (1_000_000, 64, 5, 1024),
        (10, 10_000_000, 1, 1),
        (1e8, 4, 25, 10**7),
    ]:
        assert 1e-4 <= B._step_law(n, k, T, B_) <= 1.0
    # a configuration that saturates the upper clamp
    assert B._step_law(100, 1, 1, 10**12) == 1.0


def test_recipe_batch_is_power_of_two_and_capped():
    for n, k, T in [(10_000, 64, 5), (100_000, 256, 3), (1_000_000, 1024, 2)]:
        for law in ("throughput", "quality"):
            b, a0 = B._resolve_recipe(n, k, T, law, None, None)
            assert b == int(2 ** round(math.log2(b))) or b == 1
            assert b <= n
            assert 1e-4 <= a0 <= 1.0


def test_epoch_rows_fractional():
    # whole passes, then a rounded fractional prefix
    rows = B._epoch_rows(1000, 2.5)
    assert rows == [1000, 500, 1000]
    assert B._epoch_rows(1000, 3) == [1000, 1000, 1000]
