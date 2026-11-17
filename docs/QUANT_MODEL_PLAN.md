# Quant Modeling Plan — Prediction, Regime, Entry/Exit

**Recorded:** 2026-11-17
**Status:** Approved research direction only; not yet implemented or benchmarked.
**Scope:** Forex, CFDs, metals and crypto; asset-agnostic mathematical approaches fit to each market's historical data.
**Priority:** Prediction > Market Regime Intelligence > Entry/Exit Intelligence.
**Out of scope for this plan:** building a dedicated AI capital manager, RL trading controller, or making EXAONE a mandatory centerpiece. Ordinary order decisions, position limits, margin handling and execution can be deterministic logic.

## Selected Quant research stack (5 complementary components)

| Responsibility | Selected model/method | Implementation reference | Main observable output |
|---|---|---|---|
| Directional/return prediction | Markov-Switching Autoregression (MS-AR) | https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression.html | Regime-conditional forecast of next-period returns / direction |
| Latent trend prediction | Kalman State-Space / Local Linear Trend | https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html | Filtered latent trend and forward return forecast |
| Market regime classification | Statistical Jump Model (SJM) | https://github.com/Yizhan-Oliver-Shu/jump-models | Online inferred market state and transition signals |
| Volatility prediction | EGARCH | https://arch.readthedocs.io/en/stable/univariate/forecasting.html | Conditional volatility forecast; simulation where analytical multi-step forecast is unavailable |
| Entry/Exit assessment | First-Passage probabilities + Stochastic Optimal Stopping | Quant stochastic-process methods; solver must be implemented and validated | Probability of TP before SL, expected holding time, estimated net utility of enter/wait/exit |

These five are **not five ready-made pretrained checkpoints**. MS-AR, Kalman, Jump Model and EGARCH need fitting to historical data; First-Passage/Optimal Stopping need a payoff model, boundary conditions and validated implementation. No claim of accuracy, profitability or universal superiority follows from their selection.

## Proposed prediction-only data flow

```text
Market-specific data (time-aligned OHLCV, Bid/Ask and available ticks)
  -> point-in-time features and historical fitting windows
  -> MS-AR: directional return forecast
  -> Kalman: latent state / trend forecast
  -> Jump Model: filtered ONLINE regime state (no future smoothing)
  -> EGARCH: forecast conditional volatility
  -> First-Passage + Optimal Stopping: TP/SL order, wait/enter/exit utility
  -> time-aligned quantified outputs and uncertainty
  -> simple deterministic trade logic and independent risk checks
```

EXAONE and other foundation models are **optional standalone experts** if separately proven useful. This plan does not replace the existing seven-model registry or modify active selection automatically.

## Research acceptance gates

1. **Input contracts:** define normalized returns, features available as of cutoff, quote conventions and supported horizons; broker symbol mappings are explicit.
2. **Leakage control:** fit estimators and scalers inside chronological training folds. For MS-AR and SJM never use smoothed retrospective states as live inputs; use filtered/predicted states with observed data only.
3. **Direction:** test signed hit rate, balanced accuracy, Brier / calibration and return forecasts against random walk, always-up/down and zero-return baselines.
4. **Regime:** measure state persistence, switch delay, stability and conditional performance; latent regime labels require post-fit interpretation and are not predetermined Trend/Sideway labels.
5. **Volatility:** EGARCH predicts conditional variance/volatility, not direction; validate QLIKE or realized-volatility errors on forward samples and use simulation for unsupported multi-step analytical cases.
6. **Entry/Exit:** estimate first-hit probability, expected time to hit each barrier and net expected payoff including bid/ask, spread, slippage, swaps/funding and execution delay; test both sides and no-trade.
7. **Model comparison:** chronological walk-forward / untouched holdouts by symbol and timeframe; run component ablations to test incremental **net** benefit and computation latency.
8. **Operational guard:** no live orders or autonomous leverage from research code. Risk and execution stay deterministic and separately verified.

## Implementation sequence (not authorization to implement yet)

- P1: market data features and chronology contracts.
- P2: MS-AR, Kalman and SJM fitted research adapters using only historical windows.
- P3: EGARCH forecast and calibrated horizon alignment.
- P4: First-Passage / Optimal Stopping solver with broker cost accounting.
- P5: walk-forward comparison / ablation and clearly reproducible results.
- P6: optionally connect useful outputs to the separate ensemble layer.

## Scope controls

Do not add more candidate models, a challenger roster, RL portfolio managers or additional framework dependencies without a separate instruction. This file is a **saved plan**, not a claim that the system or its checkpoints were installed, trained or tested.
