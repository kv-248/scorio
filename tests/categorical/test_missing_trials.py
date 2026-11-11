"""Missing trials must use a real rubric score, not a phantom zero."""

from __future__ import annotations

import numpy as np
import pytest

from scorio.categorical.evaluate import _scores_to_R, evaluate_schema
from scorio.categorical.io import _rows_to_columns
from scorio.eval import bayes


def test_scores_to_R_does_not_invent_a_zero_category() -> None:
    """Unfilled cells take the lowest observed score, not a literal 0."""
    scored = [
        ("p1", 0, 1.0),
        ("p1", 1, 0.5),
        ("p2", 0, 1.0),
    ]
    R, w = _scores_to_R(scored)
    assert w.tolist() == [0.5, 1.0]
    assert R.tolist() == [[1, 0], [1, 0]]


def test_scores_to_R_uses_a_negative_floor_when_that_is_the_lowest_score() -> None:
    """A literal 0 is not the floor when the rubric itself goes below zero."""
    R, w = _scores_to_R(
        [
            ("p1", 0, -1.0),
            ("p1", 1, 1.0),
            ("p2", 0, 1.0),
        ]
    )
    assert w.tolist() == [-1.0, 1.0]
    assert R.tolist() == [[0, 1], [1, 0]]


def test_missing_trial_uses_schema_floor_when_that_outcome_was_not_observed() -> None:
    """Bayes@N imputes the worst defined rubric outcome for a missing trial.

    Every observed completion is correct and high-confidence, so the only
    score in the data is 1.0. ``Confident & Correct`` can still return 0.1
    (confidently wrong). The absent trial has to receive that floor. Padding
    with 0.0 invents a category the rubric does not have and lowers the
    posterior mean from 0.6625 to 0.625.
    """
    rows = []
    for problem, n_trials in (("p1", 2), ("p2", 1)):
        for trial in range(n_trials):
            rows.append(
                {
                    "model": "m",
                    "problem": problem,
                    "trial": trial,
                    "is_correct": 1,
                    "completion_avg_logprob": 1.0,
                }
            )
    columns = _rows_to_columns(rows)
    mu, _sigma = evaluate_schema(columns, "Confident & Correct")["m"]

    expected_mu, _expected_sigma = bayes(
        np.array([[1, 1], [1, 0]]),
        np.array([0.1, 1.0]),
    )
    assert mu == pytest.approx(expected_mu)
    assert expected_mu == pytest.approx(0.6625)
