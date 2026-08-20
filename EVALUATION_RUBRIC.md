# Evaluation Rubric

This repo is built to satisfy a specific, explicit rubric — the one an institutional ML/quant reviewer (bank, fund, or a company running production models on real money) actually screens for, not "does the accuracy number look good." Naming it here, rather than leaving it implicit, is itself part of the point — see the sibling repo [`pipeline-reliability-patterns`](https://github.com/mboyajeffers/pipeline-reliability-patterns) for the same doctrine applied to data-engineering reliability instead of modeling.

A model that reports a clean accuracy number and nothing else has proven almost nothing. All nine axes below are demonstrated directly — in code, in tests, in a real (and honestly negative) live validation run — not asserted in prose.

| # | What's screened for | Where it's demonstrated |
|---|---|---|
| 1 | **No lookahead / walk-forward validation discipline** — the single most common way quant/ML systems silently lie to themselves | `model/BUILD_LOG.md` Issue #1; `model/tests/test_walkforward_no_leakage.py` proves the leak structurally, not by observing a suspicious accuracy number |
| 2 | **Model governance** — a model that hasn't earned it is never marked active | `model/pipeline.py`'s `ModelRegistry` — every run is registered, win or lose; only lift-over-baseline above a hard threshold sets `active=True` |
| 3 | **Drift / circuit breaker discipline** — a model that degrades in production gets caught, not silently trusted forever | `model/pipeline.py`'s `check_qa_hold()` — rolling-accuracy circuit breaker, test-pinned in both directions |
| 4 | **Explainability** — a prediction with no reasoning attached isn't reviewable | `live/daily_predict.py` attaches `feature_importances()` to every issued prediction |
| 5 | **Risk-adjusted evaluation, not raw accuracy** — the specific, well-known way accuracy alone misleads under class imbalance | `model/BUILD_LOG.md` Issue #2 — a real run where 0.58 accuracy against a 0.58 baseline is correctly treated as a failed model, not a working one |
| 6 | **Data integrity** — real data, sourced and attributed, no fabrication | `model/pipeline.py:fetch_daily_bars()` — live Yahoo Finance, same public source used elsewhere in this account's `financial-data-engineering` and `market-pulse` |
| 7 | **Reproducibility** — a fixed seed and a documented data window, not a lucky run | `SignalModel(random_state=42)`; every registry entry stores its exact validation window and result |
| 8 | **Honest framing under scrutiny** — the single failure mode this whole repo exists to avoid repeating | Every number in this repo is labeled backtested/walk-forward, never implied live-traded, until `live/track_record.jsonl` actually accumulates real dated predictions — see README |
| 9 | **Incident-response / postmortem discipline** — diagnose, document, fix, prevent | `model/BUILD_LOG.md`, same Issue/Impact/Root Cause/Fix/Prevention format as the sibling repo |

## How the tests are structured

Same convention as `pipeline-reliability-patterns`: every fixed bug is proven twice.

1. A test marked `@pytest.mark.xfail(strict=True)` runs against the pre-fix behavior (`model/before/naive_split.py`, or a deliberately naive gate check inline in the test itself) — it fails for the specific documented reason. If it ever unexpectedly passes, `strict=True` fails the suite.
2. A normal passing test proves the real fix.

Run it: `pytest -v` from the repo root — no network required, fully deterministic.

## What this repo is / isn't

**Is:** a small, real, walk-forward-validated binary direction model, built to institutional governance standards (hard gate, drift circuit breaker, explainability, honest reporting) and run live on a daily cron so its track record is dated and third-party-verifiable, not a claimed backtest.

**Isn't:** a framework, investment/trading advice, a claim that any specific employer's or private system's actual architecture looks like this, or a claim that the model currently works. As of the first live run (2026-08-20, SPY, 2yr window), it does not clear its own gate — disclosed in full in `model/BUILD_LOG.md` Issue #2, not hidden. That's the rubric working, not a bug in the repo.
