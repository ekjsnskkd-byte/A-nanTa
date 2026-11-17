"""Optional real Chronos-2 via AutoGluon. No dependency or weights at core install."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ananta.contracts import Candle, Forecast, validate_history

FREQUENCIES = {"M1": "min", "M5": "5min", "M15": "15min", "H1": "h", "H4": "4h", "D1": "D"}


class AutoGluonChronos2:
    """Zero-shot Chronos-2 adapter; fit() must use only historical training bars.

    fit() prepares AutoGluon's predictor and may download official model weights.
    Explicitly disabled until the caller installs the optional forecast extra.
    """

    expert_id = "chronos2"

    def __init__(self, horizon: int, model_dir: str | Path = "artifacts/chronos2") -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = horizon
        self.model_dir = str(model_dir)
        self._predictor = None

    @staticmethod
    def _frame(history: Sequence[Candle]):
        validate_history(history)
        try:
            import pandas as pd
            from autogluon.timeseries import TimeSeriesDataFrame
        except ImportError as exc:
            raise RuntimeError("Install optional dependencies: pip install -e '.[forecast]'") from exc
        rows = [{"item_id": bar.symbol, "timestamp": bar.timestamp.replace(tzinfo=None),
                 "target": bar.close} for bar in history]
        return TimeSeriesDataFrame.from_data_frame(pd.DataFrame(rows))

    def fit(self, train_history: Sequence[Candle]) -> None:
        """Initialize a pretrained zero-shot predictor using a training-only window."""
        self._frame(train_history)
        from autogluon.timeseries import TimeSeriesPredictor

        self._predictor = TimeSeriesPredictor(
            prediction_length=self.horizon, target="target", path=self.model_dir,
            eval_metric="WQL",
        ).fit(self._frame(train_history), presets="chronos2")

    def forecast(self, history: Sequence[Candle], horizon: int) -> Forecast:
        if self._predictor is None:
            raise RuntimeError("Call fit(train_history) before forecasting")
        if horizon != self.horizon:
            raise ValueError("horizon differs from the prepared predictor")
        validate_history(history)
        latest = history[-1]
        if latest.timeframe not in FREQUENCIES:
            raise ValueError("unsupported timeframe")
        result = self._predictor.predict(self._frame(history))
        predicted_price = float(result["mean"].iloc[-1])
        expected_return = predicted_price / latest.close - 1.0
        return Forecast(self.expert_id, latest.symbol, latest.timeframe,
                        latest.timestamp, horizon, expected_return)
