"""
Runs once per day via GitHub Actions cron (see ../.github/workflows/daily_track_record.yml).

Re-validates and re-registers the model against live data, then appends exactly
one record to track_record.jsonl — a real prediction if a model cleared the gate
that day, an honest "no active model" record if it didn't. Every day is logged,
not just the good ones — that's what makes the log a track record instead of a
highlight reel.

Deliberately does not place trades or touch a broker API — see the top-level
README's "Personal-use note." This is a signal generator, not a trading bot.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from pipeline import (  # noqa: E402
    LOOKAHEAD_DAYS,
    ModelRegistry,
    compute_features,
    compute_target,
    fetch_daily_bars,
    walk_forward_validate,
)
from observability import emit  # noqa: E402

TICKER = "SPY"
TRACK_RECORD_PATH = Path(__file__).parent / "track_record.jsonl"
REGISTRY_PATH = Path(__file__).parent.parent / "model" / "model_registry.json"


def run() -> dict:
    bars = fetch_daily_bars(TICKER)
    featured = compute_target(compute_features(bars))

    emit("validation_started", ticker=TICKER, rows=len(bars))
    validation = walk_forward_validate(featured)

    registry = ModelRegistry(path=REGISTRY_PATH)
    version = f"{TICKER}-{date.today().isoformat()}"
    active = registry.register(version=version, validation=validation)

    record = {
        "date": date.today().isoformat(),
        "ticker": TICKER,
        "model_version": version,
        "active": active,
        "mean_accuracy": validation["mean_accuracy"],
        "mean_baseline": validation["mean_baseline"],
        "mean_lift": validation["mean_lift"],
    }

    if active:
        latest = featured.dropna(subset=["rsi_14", "sma20_ratio", "volatility_10d", "volume_ratio_20d", "momentum_5d"]).iloc[[-1]]
        from pipeline import SignalModel, FEATURE_COLS

        clean_train = featured.dropna(subset=FEATURE_COLS + ["target"])
        model = SignalModel().fit(clean_train, clean_train["target"])
        prob_up = float(model.predict_proba(latest)[0])
        record["prediction"] = "UP" if prob_up >= 0.5 else "DOWN"
        record["probability"] = prob_up
        record["horizon_days"] = LOOKAHEAD_DAYS
        record["feature_importances"] = model.feature_importances()
        emit("prediction_issued", **{k: record[k] for k in ("prediction", "probability")})
    else:
        record["prediction"] = None
        emit("no_active_model", reason="gate not cleared", mean_lift=validation["mean_lift"])

    with TRACK_RECORD_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")

    return record


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
