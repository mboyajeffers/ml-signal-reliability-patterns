# Build Log — Signal Model

Postmortem-style log, not a changelog: each entry is Issue / Impact / Root Cause / Fix / Prevention, the same shape a real incident review uses. See `../EVALUATION_RUBRIC.md` for what this log is demonstrating structurally (rubric #9 — incident-response discipline).

---

## Issue #1 — Walk-forward split leaked target information across the train/test boundary

**Found:** while writing the first version of the walk-forward validator, structurally checking each fold's boundary rather than just trusting the split looked chronological (`before/naive_split.py` — kept, not deleted).

**Impact:** the target for row T is computed from row T + `LOOKAHEAD_DAYS` (`compute_target()` in `pipeline.py`). A walk-forward split with no gap between train end and test start puts training rows whose target was computed from data *inside the test window* directly into the training set. The model doesn't need to "cheat" on the test set directly — it's trained on rows that already encode test-window information through their labels. This is the exact same class of bug documented in the private methodology this repo's technique is modeled on (a gap sized to the lookahead window, there called `TimeSeriesSplit(..., gap=7)`), rebuilt fresh here rather than copied.

**Root cause:** the naive split (`naive_walk_forward_splits`) is chronological and non-shuffled — the two properties that make a split *look* correct at a glance — but doesn't account for the lookahead window baked into every label near a fold boundary.

**Fix:** `walk_forward_splits()` in `pipeline.py` inserts an explicit `gap` (defaults to `LOOKAHEAD_DAYS`) between the end of train and the start of test, so no training row's label window can reach into the test set.

**Prevention:** `tests/test_walkforward_no_leakage.py` proves this structurally, not by observing an accuracy delta — it directly checks, for every fold, whether any training row's label window overlaps the test window. One test (`xfail`, `strict=True`) proves the naive split leaks on every fold; one proves the gapped split never does.

---

## Issue #2 — Raw accuracy is not a valid promotion gate

**Found:** running real walk-forward validation against live SPY data (`python pipeline.py --live --ticker SPY`, 2026-08-20) to generate the first registry entry — not a synthetic example.

**Real, unmodified output from that run:**

```
mean_accuracy: 0.489   mean_baseline: 0.629   mean_lift: -0.140
Registered SPY-v1 — active=False
```

**Impact:** this result is the whole point of this issue, not an inconvenience to hide. Individual fold baselines (majority-class rate) ranged from 0.51 to 0.74 — during a fold where the market trended strongly, "predict UP every day" alone scores 74%. A gate that only checked `accuracy > 0.50` would have looked at this model's individual fold accuracies (0.43–0.55) and, depending on which fold, could still pass some of them — a model that is materially *worse* than doing nothing would clear a naive gate. `test_raw_accuracy_alone_is_not_a_valid_gate` proves this directly using this repo's own numbers: 0.58 raw accuracy against a 0.58 baseline is zero lift, not a working model, even though `0.58 > 0.50`.

**Root cause:** accuracy is not translation-invariant to class balance. A single scalar number without its baseline is not evaluable.

**Fix:** `ModelRegistry.register()` gates on `mean_lift` (accuracy minus that fold's own majority-class baseline, averaged across folds) against `MIN_LIFT_OVER_BASELINE`, never on raw accuracy alone. The SPY-v1 run above is registered — every run is, win or lose, nothing is hidden — but `active=False`. No signal is ever issued from a model that isn't active (see `live/daily_predict.py`).

**Prevention:** `tests/test_qa_gate_circuit_breaker.py::test_registry_rejects_model_below_lift_gate` and `::test_registry_accepts_model_above_lift_gate` pin both directions of this behavior directly against `ModelRegistry`.

---

## What this pipeline does *not* claim

The 5-feature technical-only set here does not currently clear its own gate on the single tickers tested (SPY, AAPL, MSFT, QQQ — all negative lift, logged honestly, not cherry-picked). That is a disclosed, real limitation, not a hidden one — and it is itself evidence the gate works: a system that always finds a way to pass its own gate isn't a gate. Building a model that reliably clears this bar (broader feature set, fundamentals, a larger pooled ticker universe) is future work, explicitly out of scope for this pass — see the top-level README's "What this is / isn't."

## Version History

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-08-20 | Initial build — 2 issues logged, fixed, and test-pinned. First live validation run (SPY, 2y) registered honestly as inactive. |
