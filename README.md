# StatisticalArbitrage

A production-grade statistical arbitrage framework for cointegrated commodity pairs. Runs a full pipeline from raw market data through statistical testing, signal generation, risk management, vectorised backtesting, and paper-trade execution on Alpaca.

## Overview

The strategy exploits **mean-reversion of cointegrated spread** between two assets. When prices diverge beyond a statistically-derived threshold, the system enters a dollar-neutral position and exits when the spread reverts. Risk is managed through position sizing, drawdown limits, and a rolling regime filter that suspends trading when cointegration breaks down.

**Current pairs:**
| Key | Leg 1 (research) | Leg 2 (research) | Leg 1 (execution) | Leg 2 (execution) |
|-----|-----------------|-----------------|-------------------|-------------------|
| `gold_silver` | `/GC` Gold futures | `/SI` Silver futures | `GLD` | `SLV` |
| `wti_brent` | `/CL` WTI futures | `/BZ` Brent futures | `USO` | `BNO` |

Futures data comes from **Schwab** (research / calibration). Live execution uses **Alpaca paper trading** with the corresponding ETF proxies.

---

## Architecture

```
StatisticalArbitrage/
├── src/
│   ├── config.py              # Central Pydantic config — all parameters live here
│   ├── backtest.py            # Vectorised backtester: P&L, metrics, trade ledger
│   ├── signals.py             # SignalEngine: full pipeline + SignalEvent types
│   ├── data/
│   │   └── schwab_loader.py   # OAuth2 + OHLCV fetch + Parquet cache
│   ├── stats/
│   │   ├── stationarity.py    # ADF + KPSS tests (I(1) check)
│   │   ├── cointegration.py   # Engle-Granger + Johansen + rolling window
│   │   └── kalman.py          # Kalman filter for dynamic hedge ratio
│   ├── risk/
│   │   ├── sizer.py           # Target-volatility position sizer → TradeOrder
│   │   └── monitor.py         # Runtime risk monitor: drawdown, exposure, kill switch
│   └── execution/
│       └── alpaca_trader.py   # Alpaca paper trading execution layer
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_cointegration.ipynb # Cointegration research
│   ├── 03_kalman.ipynb        # Kalman filter calibration
│   └── 04_backtest.ipynb      # Full backtest walkthrough
├── configs/
│   └── params.yaml            # Override file for config parameters
├── run_smoke_tests.sh         # Runs all module-level smoke tests
└── requirements.txt
```

### Data flow

```
SchwabLoader          AlpacaTrader (live prices)
     │                       │
     ▼                       ▼
  price series ──► SignalEngine ──► SignalResult
                       │
              ┌────────┼────────────┐
              │        │            │
              ▼        ▼            ▼
      Stationarity  Cointegration  KalmanHedgeRatio
      gate (ADF+    gate (EG +     (dynamic β)
      KPSS)         Johansen)
              └────────┴────────────┘
                       │
                   z-score + state machine
                       │
                  SignalEvent (+1 / -1 / 0)
                       │
               ┌───────┴───────┐
               ▼               ▼
         PositionSizer    Backtester
         (target vol)     (P&L + metrics)
               │
         TradeOrder
               │
         RiskMonitor
         (approve / block)
               │
         AlpacaTrader
         (submit orders)
```

---

## Signal Pipeline

Every call to `SignalEngine.run(y, x)` executes these steps in order:

### 1. Stationarity gate
Both legs must be **I(1)** (integrated of order 1): non-stationary in levels, stationary in first differences. Uses ADF + KPSS with configurable significance level (`adf_significance=0.05`).

### 2. Cointegration gate
The pair must be cointegrated via **Engle-Granger** AND **Johansen trace test**. A rolling window check (`coint_rolling_window=252` days) additionally verifies that at least `min_coint_fraction=70%` of trailing windows show cointegration.

### 3. Hedge ratio
Three methods, selectable per run:

| Method | Config key | Notes |
|--------|-----------|-------|
| **Kalman filter** *(default)* | `"kalman"` | Dynamic β(t), adapts to regime shifts. Built from first principles — no black-box libraries. |
| Rolling OLS | `"rolling_ols"` | β estimated over trailing `rolling_ols_window=60` days. |
| Static OLS | `"static_ols"` | Full-history OLS, fixed β. Fastest; good baseline. |

**Kalman state-space model:**
```
y(t) = β(t)·x(t) + ε(t),   ε(t) ~ N(0, Vt)
β(t) = β(t-1) + δ(t),       δ(t) ~ N(0, Wt)
```
Parameters: `kalman_delta=1e-4` (state noise), `kalman_vt=1e-3` (observation noise).

### 4. Spread & z-score
```
spread(t) = y(t) − β(t)·x(t)
z(t)      = (spread(t) − μ_w) / σ_w     # rolling window = 60 days
```

### 5. Spread quality gate
Checked per bar before any entry:
- **Half-life** of mean reversion must be in `[2, 40]` days (Kalman-estimated OU half-life)
- **Hurst exponent** must be `< 0.5` (confirms mean-reverting behaviour, not trending)

### 6. Rolling regime filter
A rolling Engle-Granger p-value is computed each bar. If `p > regime_coint_threshold=0.10`, the regime is considered broken — new entries are suppressed, but open positions are not forcibly closed (only a SUSPEND action does that).

### 7. Position state machine

| Condition | Action | New state |
|-----------|--------|-----------|
| z < −2.0 and gates ok | `ENTER` (+1 long spread) | LONG |
| z > +2.0 and gates ok | `ENTER` (−1 short spread) | SHORT |
| \|z\| < 0.5 | `EXIT` | FLAT |
| \|z\| > 3.5 | `STOP` | FLAT |
| regime broken | `SUSPEND` | FLAT |
| otherwise | `HOLD` or `WAIT` | unchanged |

---

## Backtester

`Backtester.run(signal_result)` prices every position using a **strictly causal P&L model** — both position and hedge ratio are lagged one bar:

```
unit_pnl(t) = position(t-1) · [ Δy(t) − β(t-1)·Δx(t) ]
gross_ret(t) = unit_pnl(t) / (|y(t-1)| + |β(t-1)·x(t-1)|)
cost_ret(t)  = slippage_bps/10000 · |Δposition(t)|
net_ret(t)   = gross_ret(t) − cost_ret(t)
```

**Metrics reported** (full period, in-sample, out-of-sample):

| Metric | Description |
|--------|-------------|
| Total return | Compounded net return |
| CAGR | Compound annual growth rate |
| Ann. vol | Annualised daily return std |
| Sharpe | `(CAGR − rf) / ann_vol` |
| Sortino | Uses downside deviation only |
| Max drawdown | Worst peak-to-trough |
| Calmar | `CAGR / |max_drawdown|` |
| Win rate | % of round-trips with positive P&L |
| Avg hold | Mean days per round-trip |
| Time in market | % of bars with open position |

Walk-forward split: **in-sample ≤ 2022-12-31**, **out-of-sample ≥ 2023-01-01** (configurable).

### Portfolio backtest
```python
port = Backtester().run_portfolio(
    {"gold_silver": res1, "wti_brent": res2},
    weights={"gold_silver": 0.6, "wti_brent": 0.4},  # optional; default equal weight
)
```

---

## Risk Management

### Position sizing (`sizer.py`)
Target-volatility sizing ensures each pair contributes the same annualised volatility to the portfolio:

```
raw_capital     = (target_vol / spread_vol) × equity × max_position_pct
capital_per_leg = clip(raw_capital, min_notional, max_notional)
```

Defaults: `target_annual_vol=10%`, `max_position_pct=20%` of NAV per pair, `slippage_bps=5`.

### Runtime monitor (`monitor.py`)
Every `TradeOrder` passes through `RiskMonitor.approve()` before execution:

- **Drawdown kill switch** — blocks all new entries if portfolio drawdown exceeds `max_drawdown_pct=15%`
- **Exposure cap** — total open notional cannot exceed `max_exposure_pct` of NAV
- **Duplicate guard** — rejects a second entry in the same pair while one is open
- **Pair cap** — limits maximum simultaneous open pairs
- **Audit trail** — full log of every fill, unrealised P&L, and exit

---

## Configuration

All parameters live in `src/config.py` as a Pydantic `Settings` object. Override any value via environment variables or `configs/params.yaml`.

```python
from src.config import get_config
cfg = get_config()

cfg.stats.zscore_entry        # 2.0
cfg.stats.zscore_exit         # 0.5
cfg.stats.zscore_stop         # 3.5
cfg.stats.hedge_method        # "kalman"
cfg.stats.zscore_window       # 60
cfg.stats.coint_rolling_window# 252
cfg.stats.min_halflife_days   # 2
cfg.stats.max_halflife_days   # 40
cfg.risk.target_annual_vol    # 0.10
cfg.risk.max_position_pct     # 0.20
cfg.risk.max_drawdown_pct     # 0.15
cfg.backtest.initial_capital  # 100_000
cfg.backtest.in_sample_end    # "2022-12-31"
cfg.backtest.out_sample_start # "2023-01-01"
cfg.backtest.risk_free_rate   # 0.05
```

---

## Setup

### Requirements
- Python 3.10+
- Schwab developer account (for futures data)
- Alpaca paper trading account (for execution)

### Installation

```bash
git clone https://github.com/raulghosh/StatisticalArbitrage.git
cd StatisticalArbitrage

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```env
SCHWAB_APP_KEY=your_schwab_app_key
SCHWAB_APP_SECRET=your_schwab_app_secret
SCHWAB_CALLBACK_URL=https://127.0.0.1:3000/api/schwab/callback

ALPACA_API_KEY_PAPER=your_alpaca_key
ALPACA_SECRET=your_alpaca_secret
```

On first run, `SchwabLoader` opens a browser for OAuth2 login. The token is cached in `.schwab_token.json` and refreshed silently on subsequent runs.

---

## Usage

### Research workflow

```python
from src.data.schwab_loader import SchwabLoader
from src.signals import SignalEngine
from src.backtest import Backtester

# 1. Fetch data
loader = SchwabLoader()
gc = loader.get_price_history("/GC", years=10)  # Gold futures
si = loader.get_price_history("/SI", years=10)  # Silver futures

# 2. Run signal engine
engine = SignalEngine(hedge_method="kalman", pair_key="gold_silver")
result = engine.run(gc["close"], si["close"], y_name="/GC", x_name="/SI")
result.summary()

# 3. Backtest
bt = Backtester().run(result)
bt.summary()

# Access the trade ledger
trades_df = bt.trades_dataframe()
```

### Live incremental update (for `continuous.py`)

```python
# Warm up on history
state = engine.init_state(historical_gc, historical_si)

# Each new bar
state, event = engine.update_one(state, latest_gc, latest_si, eg_pval_t=0.03)

if event.action == SignalAction.ENTER:
    order = sizer.size(event, leg1_price=gc_ask, leg2_price=si_ask)
    order = monitor.approve(order)
    if order.approved:
        trader.submit_pair_order(order)
```

### Smoke tests

```bash
bash run_smoke_tests.sh
```

Runs the `__main__` block of each module in sequence:

```
stationarity.py  ✅
cointegration.py ✅
signals.py       ✅
sizer.py         ✅
monitor.py       ✅
backtest.py      ✅
```

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_eda.ipynb` | Price series visualisation, correlation, spread inspection |
| `02_cointegration.ipynb` | ADF/KPSS/Johansen tests, rolling cointegration analysis |
| `03_kalman.ipynb` | Kalman filter calibration, β(t) vs static OLS comparison |
| `04_backtest.ipynb` | Full walk-forward backtest with P&L attribution |

---

## What's not yet built

| Component | File | Status |
|-----------|------|--------|
| Live trading loop | `src/continuous.py` | Stub — wires all modules together for daily execution |
| Alpaca execution | `src/execution/alpaca_trader.py` | Scaffold — order routing and position query implemented |

---

## Key design decisions

**No look-ahead in backtesting.** Both the position and hedge ratio are lagged one bar before computing P&L. The same z-score that triggers entry at bar *t* cannot contribute to that bar's return.

**Cointegration is the gatekeeper.** The system trades only while the statistical relationship holds. The rolling regime filter re-runs cointegration every bar and suspends entries when the relationship weakens — the strategy adapts dynamically rather than assuming a fixed relationship.

**Separation of concerns.** `SignalEngine` produces signals. `Backtester` prices them. `PositionSizer` sizes them. `RiskMonitor` approves them. `AlpacaTrader` executes them. None of these know about each other's internals.

**ETF proxies for execution.** Futures require margin accounts and carry roll costs. The strategy researches on futures (cleaner data, tighter spreads) but executes on liquid ETF proxies via Alpaca's zero-commission paper account.
