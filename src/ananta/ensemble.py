"""Transparent weighted ensemble prototype, not a trained meta-learner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from ananta.contracts import Candle, Forecast, validate_history
from ananta.experts.base import Expert
from ananta.regime import RegimeHeuristic

DEFAULT_WEIGHTS = {
    "trend": {"momentum": 0.8, "mean_reversion": 0.2},
    "range": {"momentum": 0.2, "mean_reversion": 0.8},
    "volatile": {"momentum": 0.2, "mean_reversion": 0.8},
    "unknown": {"momentum": 0.5, "mean_reversion": 0.5},
}


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    action: str
    regime: str
    horizon: int
    expected_return: float
    candidate_gross_bps: float
    candidate_net_edge_bps: float
    normalized_weights: Mapping[str, float]
    forecasts: tuple[Forecast, ...]


class WeightedEnsemble:
    def __init__(
        self,
        experts: Sequence[Expert],
        weights_by_regime: Mapping[str, Mapping[str, float]] | None = None,
        regime_detector: RegimeHeuristic | None = None,
        min_net_edge_bps: float = 2.0,
    ) -> None:
        self.experts = list(experts)
        self.weights = weights_by_regime or DEFAULT_WEIGHTS
        self.detector = regime_detector or RegimeHeuristic()
        self.min_net_edge_bps = min_net_edge_bps
        ids = [expert.expert_id for expert in self.experts]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("experts must have unique non-empty IDs")
        if min_net_edge_bps < 0 or not math.isfinite(min_net_edge_bps):
            raise ValueError("invalid minimum net edge")

    def forecast(
        self, history: Sequence[Candle], horizon: int = 1, round_trip_cost_bps: float = 0.0
    ) -> EnsembleResult:
        validate_history(history)
        if horizon < 1 or not math.isfinite(round_trip_cost_bps) or round_trip_cost_bps < 0:
            raise ValueError("invalid horizon or cost")
        regime = self.detector.detect(history)
        configured = self.weights.get(regime)
        if not configured:
            raise ValueError(f"no weights defined for regime {regime}")
        raw = {expert.expert_id: configured.get(expert.expert_id, 0.0)
               for expert in self.experts}
        if any(not math.isfinite(w) or w < 0 for w in raw.values()):
            raise ValueError("weights must be finite and nonnegative")
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("at least one expert must have positive weight")
        weights = {name: value / total for name, value in raw.items()}
        predictions: list[Forecast] = []
        combined = 0.0
        latest = history[-1]
        for expert in self.experts:
            prediction = expert.forecast(history, horizon)
            if (prediction.expert_id != expert.expert_id or
                    prediction.symbol != latest.symbol or
                    prediction.timeframe != latest.timeframe or
                    prediction.observed_at != latest.timestamp or
                    prediction.horizon != horizon):
                raise ValueError(f"unaligned prediction from {expert.expert_id}")
            predictions.append(prediction)
            combined += weights[expert.expert_id] * prediction.expected_return
        gross_bps = combined * 10_000.0
        candidate_edge = abs(gross_bps) - round_trip_cost_bps
        action = "NO_TRADE"
        if candidate_edge > self.min_net_edge_bps:
            action = "BUY" if gross_bps > 0 else "SELL"
        return EnsembleResult(action, regime, horizon, combined, gross_bps,
                              candidate_edge, weights, tuple(predictions))
