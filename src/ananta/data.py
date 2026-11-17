"""CSV adapter; no network access and no Dukascopy dependency."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from ananta.contracts import Candle, validate_history


def parse_utc_timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("timestamps require explicit UTC offset or Z")
    return dt.astimezone(timezone.utc)


def read_ohlcv_csv(path: str | Path) -> list[Candle]:
    required = {"timestamp", "symbol", "timeframe", "open", "high", "low", "close"}
    rows: list[Candle] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV requires columns {sorted(required)}")
        for n, row in enumerate(reader, start=2):
            try:
                rows.append(
                    Candle(
                        timestamp=parse_utc_timestamp(row["timestamp"]),
                        symbol=row["symbol"].strip(),
                        timeframe=row["timeframe"].strip(),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("volume") or 0.0),
                        bid=float(row["bid"]) if row.get("bid") else None,
                        ask=float(row["ask"]) if row.get("ask") else None,
                    )
                )
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"invalid CSV row {n}: {exc}") from exc
    validate_history(rows)
    return rows
