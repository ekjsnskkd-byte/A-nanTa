from datetime import datetime, timedelta, timezone

import pytest

from ananta.contracts import Candle, Forecast, validate_history


def bar(index: int, close: float = 100.0) -> Candle:
    return Candle(datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index),
                  "SYNTHUSD", "M1", close, close + 0.2, close - 0.2, close)


def test_valid_history():
    validate_history([bar(0), bar(1)])


def test_reject_out_of_order():
    with pytest.raises(ValueError, match="increasing"):
        validate_history([bar(1), bar(0)])


def test_reject_naive_timestamp():
    with pytest.raises(ValueError, match="timezone"):
        Candle(datetime(2026, 1, 1), "X", "M1", 10, 11, 9, 10)


def test_reject_crossed_quote():
    with pytest.raises(ValueError, match="non-crossed"):
        Candle(datetime(2026, 1, 1, tzinfo=timezone.utc), "X", "M1",
               10, 11, 9, 10, bid=10.1, ask=10.0)


def test_reject_nonfinite_forecast():
    with pytest.raises(ValueError):
        Forecast("x", "X", "M1", bar(0).timestamp, 1, float("nan"))
