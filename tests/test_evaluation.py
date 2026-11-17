import pytest

from ananta.evaluation import realized_net_return, walk_forward_splits


def test_embargo_prevents_overlap():
    splits = list(walk_forward_splits(100, train_bars=40, test_bars=10,
                                     embargo_bars=5, step_bars=10))
    assert len(splits) == 5
    for train, test in splits:
        assert train.stop + 5 == test.start
        assert len(test) == 10


def test_buy_pays_spread():
    result = realized_net_return("BUY", 100.0, 100.2, 100.1, 100.3)
    assert result < 0


def test_short_pays_spread():
    result = realized_net_return("SELL", 100.0, 100.2, 100.1, 100.3)
    assert result < 0


def test_fee_reduces_net_return():
    initial = realized_net_return("BUY", 100, 100.1, 101, 101.1)
    charged = realized_net_return("BUY", 100, 100.1, 101, 101.1,
                                  commission_bps=10, slippage_bps=5)
    assert charged == pytest.approx(initial - 15 / 10000)
