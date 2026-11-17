from datetime import datetime, timedelta, timezone

import pytest

from ananta.contracts import Forecast
from ananta.decision import PolicyLimits, assess_forecast
from ananta.microstructure import Quote

NOW = datetime(2026, 11, 17, tzinfo=timezone.utc)


def forecast(expected=0.01):
    return Forecast("mock", "EURUSD", "M1", NOW, 1, expected)


def test_policy_proposes_no_order():
    quote = Quote(NOW, "EURUSD", 100, 100.01)
    answer = assess_forecast(forecast(), quote)
    assert answer.action == "BUY"
    assert answer.research_only is True


def test_weak_edge_abstains():
    q = Quote(NOW, "EURUSD", 100, 100.03)
    result = assess_forecast(forecast(0.00001), q)
    assert result.action == "NO_TRADE"
    assert result.reason == "INSUFFICIENT_NET_EDGE"


def test_stale_forecast_abstains():
    q = Quote(NOW + timedelta(minutes=30), "EURUSD", 100, 100.01)
    assert assess_forecast(forecast(), q).reason == "STALE_FORECAST"


def test_wide_spread_abstains():
    q = Quote(NOW, "EURUSD", 100, 102)
    assert assess_forecast(forecast(), q).reason == "EXCESSIVE_SPREAD"


def test_forecast_future_abstains():
    q = Quote(NOW - timedelta(seconds=1), "EURUSD", 100, 100.01)
    assert assess_forecast(forecast(), q).reason == "QUOTE_BEFORE_FORECAST"


def test_mismatch_rejected():
    with pytest.raises(ValueError, match="mismatch"):
        assess_forecast(forecast(), Quote(NOW, "GBPUSD", 100, 100.01))


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        PolicyLimits(max_spread_bps=float("nan"))
