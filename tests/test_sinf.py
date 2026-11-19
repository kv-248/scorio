"""Regression tests for sequential-inference helpers."""

from __future__ import annotations

import pytest

from scorio.sinf import ci_from_mu_sigma


def test_clipped_interval_cannot_invert_when_mean_is_outside_clip() -> None:
    """A mean outside the clip must collapse to the nearest bound, not invert."""
    above = ci_from_mu_sigma(2.0, 0.01, clip=(0.0, 1.0))
    below = ci_from_mu_sigma(-1.0, 0.01, clip=(0.0, 1.0))
    point = ci_from_mu_sigma(1.5, 0.0, clip=(0.0, 1.0))

    assert above == pytest.approx((1.0, 1.0))
    assert below == pytest.approx((0.0, 0.0))
    assert point == pytest.approx((1.0, 1.0))
    assert above[0] <= above[1]
    assert below[0] <= below[1]
    assert point[0] <= point[1]


def test_clip_still_truncates_only_the_endpoint_that_leaves_the_range() -> None:
    lo, hi = ci_from_mu_sigma(0.0, 0.1, clip=(0.0, 1.0))

    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.1959963984540054)
    assert lo <= hi


def test_inverted_clip_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="clip"):
        ci_from_mu_sigma(0.5, 0.1, clip=(1.0, 0.0))
