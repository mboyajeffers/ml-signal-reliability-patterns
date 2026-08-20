"""
The naive, pre-fix version — kept in the repo, not deleted, so the leakage it
causes can be proven directly rather than just described. See ../BUILD_LOG.md
Issue #1 and ../pipeline.py's `walk_forward_splits()` for the fix.

Looks like a reasonable walk-forward split at a glance: expanding train window,
fixed test window, no shuffling, chronological order preserved. The bug is what
it's missing, not what it does wrong.
"""

from __future__ import annotations

import numpy as np


def naive_walk_forward_splits(n_rows: int, n_splits: int = 4, min_train: int = 120):
    """
    Same shape as pipeline.py's walk_forward_splits() — expanding train, fixed
    test window — but with NO gap between train end and test start.
    """
    usable = n_rows - min_train
    test_size = usable // n_splits

    for fold in range(n_splits):
        train_end = min_train + fold * test_size
        test_start = train_end  # <-- the bug: test starts immediately where train ends
        test_end = test_start + test_size
        if test_end > n_rows:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, min(test_end, n_rows))
        yield train_idx, test_idx
