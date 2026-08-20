# Signal Model

**What it does:** predicts whether SPY's close price will be higher `LOOKAHEAD_DAYS` (3) trading days from now, using 5 standard technical features computed from live daily OHLCV (Yahoo Finance, no API key).

**Architecture, in 5 lines:** `fetch_daily_bars()` hits the network → `compute_features()` builds RSI-14/SMA-ratio/volatility/volume-ratio/momentum → `compute_target()` labels each row from its own future (with a documented lookahead-leakage hazard, see `BUILD_LOG.md` Issue #1) → `walk_forward_splits()` validates with an explicit gap → `ModelRegistry` gates promotion on lift over baseline, never raw accuracy (Issue #2). I/O and logic are separate so the logic is unit-testable without a network call.

**What this proves — see `BUILD_LOG.md` for the full postmortem on each:**
1. **No-lookahead walk-forward validation** — a gap between train and test sized to the label's own lookahead window, proven structurally, not by trusting a clean-looking split.
2. **Governance that rejects real models, honestly** — the first live run (SPY, 2yr, 2026-08-20) registered as `active=False`. That result is reported here in full, not replaced with a better-looking one.

**Run it yourself:**
```bash
pip install -r ../requirements.txt
pytest ../model/tests -v         # deterministic, no network required
python pipeline.py --live --ticker SPY   # hits the real Yahoo Finance endpoint
```

**Extending this (explicitly out of scope for this pass):**
- The current single-ticker, 5-feature setup doesn't clear its own gate — a pooled multi-ticker cross-sectional training set (more rows, more regime diversity) or a broader feature set (fundamentals, macro) is the natural next iteration, not attempted here so this stays a small, honestly-scoped demonstration rather than a moving target.
- Wiring `live/daily_predict.py`'s output into an actual paper-trading order (via a `BrokerInterface`-style abstraction) is a clean, separate addition — this repo deliberately stops at "signal," not "trade."

**What this is / isn't:** a small demonstration of ML validation and governance discipline — not a trading system, not investment advice, not a claim that this is any specific employer's or private system's actual architecture, and not (currently) a model that has earned production status by its own stated rule.
