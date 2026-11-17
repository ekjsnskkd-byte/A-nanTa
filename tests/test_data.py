from pathlib import Path

import pytest

from ananta.data import parse_utc_timestamp, read_ohlcv_csv

ROOT = Path(__file__).resolve().parents[1]


def test_csv_is_valid():
    data = read_ohlcv_csv(ROOT / "examples" / "synthetic_m1.csv")
    assert len(data) == 15
    assert data[-1].symbol == "SYNTHUSD"
    assert data[0].timestamp.tzinfo is not None


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError, match="UTC offset"):
        parse_utc_timestamp("2026-01-05T00:00:00")


def test_missing_csv_column_rejected(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("timestamp,close\n2026-01-01T00:00:00Z,1\n")
    with pytest.raises(ValueError, match="requires columns"):
        read_ohlcv_csv(path)
