"""
Proves the rolling-accuracy circuit breaker and the model-registry lift gate —
see ../BUILD_LOG.md Issue #2 for why "beats 50%" is not the same as "beats this
fold's own majority-class baseline."
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import (
    ROLLING_WINDOW,
    ROLLING_HOLD_THRESHOLD,
    MIN_LIFT_OVER_BASELINE,
    ModelRegistry,
    check_qa_hold,
)


def test_qa_hold_false_on_insufficient_data():
    outcomes = [True] * (ROLLING_WINDOW - 1)
    assert check_qa_hold(outcomes) is False


def test_qa_hold_true_when_rolling_accuracy_below_threshold():
    n_correct = int(ROLLING_WINDOW * (ROLLING_HOLD_THRESHOLD - 0.05))
    outcomes = [True] * n_correct + [False] * (ROLLING_WINDOW - n_correct)
    assert check_qa_hold(outcomes) is True


def test_qa_hold_false_when_rolling_accuracy_above_threshold():
    n_correct = int(ROLLING_WINDOW * (ROLLING_HOLD_THRESHOLD + 0.15))
    outcomes = [True] * n_correct + [False] * (ROLLING_WINDOW - n_correct)
    assert check_qa_hold(outcomes) is False


def test_qa_hold_only_looks_at_most_recent_window():
    # A bad start followed by a fully-recovered recent window should NOT hold.
    outcomes = [False] * 20 + [True] * ROLLING_WINDOW
    assert check_qa_hold(outcomes) is False


@pytest.mark.xfail(
    strict=True,
    reason="a model that only matches its fold's own majority-class rate has "
    "learned nothing — raw accuracy alone must not clear the gate",
)
def test_raw_accuracy_alone_is_not_a_valid_gate():
    """
    Demonstrates the exact mistake this pipeline refuses to make: a model
    scoring 58% raw accuracy on a fold whose majority class is already 58%
    has zero lift and must NOT be registered as active — but a naive gate
    that only checks "accuracy > 0.5" would pass it. This test intentionally
    uses that naive (wrong) gate to prove it's wrong.
    """
    validation = {"mean_accuracy": 0.58, "mean_baseline": 0.58, "mean_lift": 0.0}
    naive_gate_passes = validation["mean_accuracy"] > 0.50  # the wrong check
    assert not naive_gate_passes, "a model with zero lift over baseline must not pass any gate"


def test_registry_rejects_model_below_lift_gate(tmp_path):
    registry = ModelRegistry(path=tmp_path / "registry.json")
    active = registry.register(
        version="test-v0",
        validation={"mean_accuracy": 0.58, "mean_baseline": 0.58, "mean_lift": 0.0},
    )
    assert active is False
    assert registry.active_model_version() is None


def test_registry_accepts_model_above_lift_gate(tmp_path):
    registry = ModelRegistry(path=tmp_path / "registry.json")
    active = registry.register(
        version="test-v1",
        validation={
            "mean_accuracy": 0.55,
            "mean_baseline": 0.50,
            "mean_lift": MIN_LIFT_OVER_BASELINE + 0.01,
        },
    )
    assert active is True
    assert registry.active_model_version() == "test-v1"
