from datetime import datetime, timezone

import pytest

from ananta.microstructure import L2Snapshot, Level, Quote

NOW = datetime(2026, 11, 17, tzinfo=timezone.utc)


def test_quote_spread():
    quote = Quote(NOW, "EURUSD", 1.10, 1.101)
    assert quote.spread_bps == pytest.approx((0.001 / 1.1005) * 10000)


def test_crossed_quote_rejected():
    with pytest.raises(ValueError, match="non-crossed"):
        Quote(NOW, "EURUSD", 1.12, 1.11)


def test_l2_imbalance_only_from_supplied_levels():
    book = L2Snapshot(NOW, "BTCUSD", "TEST_EXCHANGE",
                      (Level(100, 3), Level(99, 1)),
                      (Level(101, 1), Level(102, 1)))
    assert book.imbalance(depth=2) == pytest.approx(1 / 3)


def test_invalid_book_rejected():
    with pytest.raises(ValueError, match="crossed or locked"):
        L2Snapshot(NOW, "BTCUSD", "TEST_EXCHANGE",
                   (Level(101, 3),), (Level(100, 1),))
    with pytest.raises(ValueError, match="bid levels"):
        L2Snapshot(NOW, "BTCUSD", "TEST_EXCHANGE",
                   (Level(99, 1), Level(100, 1)), (Level(101, 1),))
