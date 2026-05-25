# Bitcoin Portfolio Insurance

A dynamic risk budgeting strategy that protects a Bitcoin portfolio against drawdowns exceeding a defined maximum threshold. This project is the live RQA-1 component of [The Self-Driving Portfolio](https://github.com/geraledesma/self-driving-portfolio).

---

## Strategy

Replace a static CPPI allocator with **`drawdown_allocator()`** — a dynamic allocation approach that continuously rebalances between Bitcoin (PSP) and a short-duration safe asset (USDC/cash) to ensure the portfolio never falls below `(1 − max_drawdown) × peak_value`.

### Risk Budgeting Hierarchy

| Strategy | Description |
|---|---|
| Naive Fixed Mix | Static allocation (e.g., 80/20 BTC/cash); no adjustment over time |
| CPPI | Cushion = portfolio value − floor; risky assets ∝ cushion × multiplier |
| **Drawdown Allocator** | Tracks peak value; constrains loss to max_drawdown × peak; dynamic rebalancing |
| Dynamic Risk Budgeting | Liability-aware; stochastic modeling (advanced) |

---

## Implementation

- `drawdown_allocator()` — core strategy; monitors peak and adjusts BTC/cash allocation dynamically
- `bt_mix(btc, usdc, ...)` — backtester using the `erk` library (EDHEC Risk Institute)
- Benchmark: CPPI allocator on the same BTC/USDC universe

---

## Roadmap

- [ ] Implement `drawdown_allocator()` strategy
- [ ] Build `bt_mix(btc, usdc)` backtester
- [ ] Backtest drawdown allocator vs. CPPI benchmark
- [ ] Document results and compare to floor allocator
- [ ] Integrate with Hermes Agent execution stack (autonomous signal generation)

---

## Theoretical Foundation

Based on the dynamic risk budgeting framework from EDHEC Risk Institute:
- Coursera: *Introduction to Portfolio Construction and Analysis with Python* — Lab: Dynamic risk budgeting between PSP & LHP
- Parent project: [The Self-Driving Portfolio](https://github.com/geraledesma/self-driving-portfolio) — Andrew Ang / BlackRock MASS architecture

---

## Status

**Active development.** Personal sandbox. Not managing third-party capital.
