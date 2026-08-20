"""
Signal model pipeline: fetch -> features -> walk-forward validate -> gate -> predict.

I/O (fetch_daily_bars) and logic (everything else) are deliberately separate so the
model logic is unit-testable without a network call — same separation used elsewhere
in this account's public pipelines.

Scope: a small, honest, real binary direction classifier (does the close price
LOOKAHEAD_DAYS from now finish higher than today's close). Not a trading system,
not investment advice — see README.md "What this is / isn't".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

LOOKAHEAD_DAYS = 3
GAP_DAYS = LOOKAHEAD_DAYS  # gap must be >= lookahead or target leakage is possible — see BUILD_LOG Issue #1
REGISTRY_PATH = Path(__file__).parent / "model_registry.json"


def fetch_daily_bars(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Live daily OHLCV via yfinance. No API key required."""
    import yfinance as yf

    df = yf.Ticker(ticker).history(period=period, interval="1d")
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index.name = "date"
    return df.reset_index()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Small, honest feature set (5 features) — deliberately not the exact feature
    list/count of any private system. Real, standard technical features only.
    """
    out = df.copy()
    ret = out["close"].pct_change()

    # RSI-14 (Wilder's smoothing)
    delta = out["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    out["sma20_ratio"] = out["close"] / out["close"].rolling(20).mean()
    out["volatility_10d"] = ret.rolling(10).std()
    out["volume_ratio_20d"] = out["volume"] / out["volume"].rolling(20).mean()
    out["momentum_5d"] = out["close"].pct_change(5)

    return out


FEATURE_COLS = ["rsi_14", "sma20_ratio", "volatility_10d", "volume_ratio_20d", "momentum_5d"]


def compute_target(df: pd.DataFrame, lookahead: int = LOOKAHEAD_DAYS) -> pd.DataFrame:
    """
    Binary target: 1 if close price `lookahead` trading days ahead is higher than
    today's close, else 0. NaN (dropped later) for the last `lookahead` rows —
    there is no future data for them yet.

    IMPORTANT: must only be called on chronologically sorted data. The target for
    row T is computed from row T + lookahead — this is exactly the information
    that must never be visible to a model trained on data "as of" row T. See
    BUILD_LOG Issue #1 for what happens when a train/test split doesn't respect this.
    """
    out = df.copy()
    forward_close = out["close"].shift(-lookahead)
    out["target"] = (forward_close > out["close"]).astype(float)
    out.loc[forward_close.isna(), "target"] = np.nan
    return out


def walk_forward_splits(n_rows: int, n_splits: int = 4, gap: int = GAP_DAYS, min_train: int = 120):
    """
    Real walk-forward validation: expanding train window, gap between train end
    and test start, fixed-size test window. Yields (train_idx, test_idx) arrays.

    The gap exists specifically so that no row in the training set has a target
    value computed from data inside the test window — see BUILD_LOG Issue #1.
    """
    usable = n_rows - min_train - gap
    if usable < n_splits:
        raise ValueError(f"Not enough rows ({n_rows}) for {n_splits} walk-forward folds")
    test_size = usable // n_splits

    for fold in range(n_splits):
        train_end = min_train + fold * test_size
        test_start = train_end + gap
        test_end = test_start + test_size
        if test_end > n_rows:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, min(test_end, n_rows))
        yield train_idx, test_idx


@dataclass
class SignalModel:
    """Thin wrapper around GradientBoostingClassifier — feature importances exposed directly."""

    random_state: int = 42
    model: GradientBoostingClassifier = field(init=False)
    feature_cols: list[str] = field(default_factory=lambda: list(FEATURE_COLS))
    is_fitted: bool = False

    def __post_init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=self.random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SignalModel":
        self.model.fit(X[self.feature_cols], y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_cols])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_cols])[:, 1]

    def feature_importances(self) -> dict[str, float]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return dict(sorted(
            zip(self.feature_cols, self.model.feature_importances_.tolist()),
            key=lambda kv: -kv[1],
        ))


def walk_forward_validate(
    df: pd.DataFrame, n_splits: int = 4, gap: int = GAP_DAYS
) -> dict:
    """
    Runs walk-forward validation, returns per-fold accuracy, the majority-class
    baseline for each fold, and the mean lift over baseline.

    Reporting raw accuracy alone is exactly the mistake this function refuses to
    make — see BUILD_LOG Issue #2. A model that only matches its fold's own
    majority-class rate has learned nothing.
    """
    clean = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
    fold_results = []

    for train_idx, test_idx in walk_forward_splits(len(clean), n_splits=n_splits, gap=gap):
        train, test = clean.iloc[train_idx], clean.iloc[test_idx]
        if train["target"].nunique() < 2 or len(test) == 0:
            continue

        model = SignalModel().fit(train, train["target"])
        preds = model.predict(test)
        accuracy = float((preds == test["target"].to_numpy()).mean())
        baseline = float(max(test["target"].mean(), 1 - test["target"].mean()))

        fold_results.append({
            "accuracy": accuracy,
            "baseline": baseline,
            "lift": accuracy - baseline,
            "n_test": len(test),
        })

    if not fold_results:
        raise ValueError("No valid folds produced — insufficient data")

    return {
        "folds": fold_results,
        "mean_accuracy": float(np.mean([f["accuracy"] for f in fold_results])),
        "mean_baseline": float(np.mean([f["baseline"] for f in fold_results])),
        "mean_lift": float(np.mean([f["lift"] for f in fold_results])),
    }


# --- Model registry: minimal, file-backed, hard gate before "active" ---

MIN_LIFT_OVER_BASELINE = 0.03  # model must beat its own fold's majority-class rate by 3pp, not just beat 50%


class ModelRegistry:
    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = path
        self._data = json.loads(path.read_text()) if path.exists() else {"models": []}

    def register(self, version: str, validation: dict, gate: float = MIN_LIFT_OVER_BASELINE) -> bool:
        """Registers a model. `active` is only ever True if it clears the lift gate."""
        active = validation["mean_lift"] >= gate
        entry = {
            "version": version,
            "mean_accuracy": validation["mean_accuracy"],
            "mean_baseline": validation["mean_baseline"],
            "mean_lift": validation["mean_lift"],
            "gate": gate,
            "active": active,
        }
        self._data["models"].append(entry)
        self.path.write_text(json.dumps(self._data, indent=2))
        return active

    def active_model_version(self) -> Optional[str]:
        active = [m for m in self._data["models"] if m["active"]]
        return active[-1]["version"] if active else None


# --- QA gate: rolling-accuracy circuit breaker ---

ROLLING_WINDOW = 10
ROLLING_HOLD_THRESHOLD = 0.45


def check_qa_hold(recent_outcomes: list[bool], window: int = ROLLING_WINDOW, threshold: float = ROLLING_HOLD_THRESHOLD) -> bool:
    """
    Returns True if predictions should be HELD (paused) — rolling accuracy over
    the most recent `window` resolved predictions has dropped below `threshold`.
    Returns False (safe to predict) if there isn't yet a full window of data —
    never holds on insufficient data, only on demonstrated underperformance.
    """
    if len(recent_outcomes) < window:
        return False
    recent = recent_outcomes[-window:]
    rolling_accuracy = sum(recent) / len(recent)
    return rolling_accuracy < threshold


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Fetch real data and run validation")
    parser.add_argument("--ticker", default="SPY")
    args = parser.parse_args()

    if args.live:
        bars = fetch_daily_bars(args.ticker)
        featured = compute_target(compute_features(bars))
        result = walk_forward_validate(featured)
        print(json.dumps(result, indent=2))
        registry = ModelRegistry()
        active = registry.register(version=f"{args.ticker}-v1", validation=result)
        print(f"Registered {args.ticker}-v1 — active={active}")
