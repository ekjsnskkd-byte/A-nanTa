"""Common expert interface: adapters must never inspect future observations."""

from __future__ import annotations

from typing import Protocol, Sequence

from ananta.contracts import Candle, Forecast


class Expert(Protocol):
    expert_id: str

    def forecast(self, history: Sequence[Candle], horizon: int) -> Forecast:
        """Forecast using only bars at or before history[-1].timestamp."""
        ...
