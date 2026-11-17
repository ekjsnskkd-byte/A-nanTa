# Market Data Strategy — Dukascopy excluded

## Source candidates (not yet connected)

| Market | Primary candidate | Fallback / validation | Cautions |
|---|---|---|---|
| FX major/minor | HistData tick and MT5 broker | TrueFX / FXCM | Volume, UTC offset, depth, account spread |
| XAUUSD, XAGUSD | MT5 broker ticks + HistData if available | independent broker | Contract size, session hours, broker feeds |
| Index & energy CFDs | MT5 broker history | licensed index/future proxy | Synthetic CFD prices vary by broker |
| Crypto spot/perpetual | CryptoHFTData + exchange public archives | OKX / Binance | L2 book availability, delistings, symbol mapping |
| Final execution calibration | Target broker's MT5 bid/ask / fills | trade/deal reports | Historical quotes ≠ actual fills |

Links: https://www.histdata.com/ | https://www.truefx.com/ | https://github.com/fxcm/MarketData | https://www.cryptohftdata.com/ | https://www.okx.com/historical-data | https://github.com/binance/binance-public-data | https://www.mql5.com/en/docs/python_metatrader5

These are **candidates**, not guarantees of free bulk access, completeness or commercial rights. Confirm APIs/ToS/redistribution limits before downloading. No Dukascopy collector should be added.

## Canonical storage specification

```text
data/
  raw/<provider>/<asset>/<symbol>/<yyyy>/<mm>/*         # immutable, ignored
  processed/<schema_version>/<venue>/<symbol>/*.parquet # immutable, ignored
  quality/<snapshot_id>/*.json                          # ignored, or sanitized summary
  manifests/<snapshot_id>.json                          # hashes / source / license / cutoffs
```

Proposed fields: `event_time_utc`, `available_at_utc`, `ingested_at_utc`, `venue`, `provider`, `instrument_id`, `symbol`, `bid`, `ask`, `bid_size`, `ask_size`, `last_price`, `last_size`, `source_flags`. Fields may be unavailable; never silently fill missing L2/tick measurements.

OHLCV training windows derive only from **closed** bars. Historical timezone conversion must explicitly handle provider conventions, DST and trading calendars; never infer a timezone from a naked timestamp.

## Ingestion quality gates

- All timestamps parsed with explicit offset, converted to UTC.
- Reject duplicates or out-of-order records; record gap statistics.
- Verify positivity, crossed quotes, spread spikes, weekend/holiday sessions and market-specific precision.
- Normalize contract specifications without conflating spot, future, CFD and perpetual instruments.
- Keep source-specific data separated until feed bias has been measured.
- Preserve **point-in-time availability** for features/labels and snapshot immutable splits.
- Broker fills, financing, swaps, rollovers, fees and margin schedules are a distinct dataset.
- Validate economic usefulness with per-symbol/timeframe coverage, quote quality and latency reports.

`src/ananta/data.py` currently implements a **single-symbol sorted CSV** contract and is intentionally not a production-scale data lake.
