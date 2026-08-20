"""
Proves the walk-forward-gap bug and its fix structurally, not just by observing
an accuracy number. A row's target at index T is computed from row T+LOOKAHEAD —
so a split "leaks" if any training row's target lookahead window reaches into
the test set. See ../BUILD_LOG.md Issue #1.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline import LOOKAHEAD_DAYS, walk_forward_splits
from before.naive_split import naive_walk_forward_splits

N_ROWS = 400


def _has_leakage(train_idx, test_idx, lookahead: int) -> bool:
    """A leak exists if the last training row's target window reaches test_idx."""
    if len(train_idx) == 0 or len(test_idx) == 0:
        return False
    last_train_target_reaches = max(train_idx) + lookahead
    return last_train_target_reaches >= min(test_idx)


@pytest.mark.xfail(
    strict=True,
    reason="naive split has no gap — the last LOOKAHEAD_DAYS rows of train have "
    "targets computed from data inside the test window",
)
def test_naive_split_has_no_target_leakage():
    for train_idx, test_idx in naive_walk_forward_splits(N_ROWS, n_splits=4):
        assert not _has_leakage(train_idx, test_idx, LOOKAHEAD_DAYS), (
            "naive split leaked target information across the train/test boundary"
        )


def test_gap_split_has_no_target_leakage():
    folds = list(walk_forward_splits(N_ROWS, n_splits=4, gap=LOOKAHEAD_DAYS))
    assert len(folds) > 0, "expected at least one valid fold"
    for train_idx, test_idx in folds:
        assert not _has_leakage(train_idx, test_idx, LOOKAHEAD_DAYS), (
            "gap split leaked target information — gap is not large enough"
        )


def test_gap_split_folds_are_chronological_and_non_overlapping():
    folds = list(walk_forward_splits(N_ROWS, n_splits=4, gap=LOOKAHEAD_DAYS))
    for train_idx, test_idx in folds:
        assert max(train_idx) < min(test_idx), "train must fully precede test"
    for i in range(len(folds) - 1):
        assert folds[i][0][-1] <= folds[i + 1][0][-1], "train windows must expand, not shrink"
