"""
backtest.py — Vectorised Pairs Backtester

Turns a SignalResult (from src.signals.SignalEngine.run) into a P&L curve,
risk-adjusted performance metrics, and a per-trade ledger.

Design principle (mirrors the rest of the project):
  The SignalEngine decides WHEN to be in the market (+1 / -1 / 0).
  This module decides WHAT THAT WAS WORTH — it never generates signals,
  it only prices the positions the engine produced.

P&L model (long spread = long y, short β·x)
-------------------------------------------
We hold `position` units of the spread.  A position opened at the close of
bar t-1 earns, into bar t, the dollar move of one share of y net of β shares
of x:

    unit_pnl(t) = position(t-1) · [ Δy(t) − β(t-1)·Δx(t) ]

Both the position and the hedge ratio are LAGGED by one bar, so the backtest
is strictly causal — no look-ahead from same-bar z-scores or β re-estimates.

To express this as a return we normalise by the gross dollar exposure that
was actually deployed at t-1:

    gross(t-1)   = |y(t-1)| + |β(t-1)·x(t-1)|
    gross_ret(t) = unit_pnl(t) / gross(t-1)

i.e. the return on the capital tied up in both legs while the trade is on.

Costs
-----
On every change in position (entry, exit, flip) both legs are traded.  The
slippage charged is `slippage_bps` of the traded notional.  Because our
return basis is the gross notional (both legs), the cost as a fraction of
that basis is:

    cost_ret(t) = (slippage_bps / 10_000) · |position(t) − position(t-1)|

    net_ret(t)  = gross_ret(t) − cost_ret(t)

The equity curve is the compounded net return of a book that allocates its
full gross notional to the pair whenever a position is open and sits in cash
(zero return) when flat.

Walk-forward
------------
config.backtest.in_sample_end / out_sample_start split the sample.  Metrics
are reported for the full period, in-sample, and out-of-sample so you can see
whether edge survives outside the fitting window.

Usage
-----
    from src.signals import SignalEngine
    from src.backtest import Backtester

    result = SignalEngine().run(df["/GC"], df["/SI"], y_name="/GC", x_name="/SI")
    bt     = Backtester().run(result)
    bt.summary()

    # Multi-pair, equal-weight portfolio
    port = Backtester().run_portfolio({"gold_silver": r1, "wti_brent": r2})
    port.summary()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_config
from src.signals import SignalResult

TRADING_DAYS = 252


# ══════════════════════════════════════════════════════════════════════════════
# Per-trade record
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """One round-trip: entry → exit."""
    pair_key      : str
    direction     : int          # +1 long spread | -1 short spread
    entry_date    : pd.Timestamp
    exit_date     : pd.Timestamp
    holding_days  : int
    entry_zscore  : float
    exit_zscore   : float
    pnl_return    : float         # compounded net return over the trade
    reason        : str           # 'exit' | 'stop' | 'open' (still open at end)

    def to_series(self) -> pd.Series:
        return pd.Series({
            "pair_key"     : self.pair_key,
            "direction"    : self.direction,
            "entry_date"   : self.entry_date,
            "exit_date"    : self.exit_date,
            "holding_days" : self.holding_days,
            "entry_zscore" : round(self.entry_zscore, 4),
            "exit_zscore"  : round(self.exit_zscore, 4),
            "pnl_return"   : round(self.pnl_return, 6),
            "reason"       : self.reason,
        })


# ══════════════════════════════════════════════════════════════════════════════
# Performance metrics for one window
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PerfMetrics:
    """Risk/return summary over a single date window."""
    label          : str
    start          : Optional[pd.Timestamp]
    end            : Optional[pd.Timestamp]
    n_days         : int
    total_return   : float
    cagr           : float
    ann_vol        : float
    sharpe         : float
    sortino        : float
    max_drawdown   : float
    calmar         : float
    time_in_market : float
    n_trades       : int
    win_rate       : float
    avg_trade_ret  : float
    avg_hold_days  : float
    turnover       : float

    def line(self) -> str:
        return (
            f"  {self.label:<12} | "
            f"ret {self.total_return:>7.1%} | "
            f"CAGR {self.cagr:>6.1%} | "
            f"vol {self.ann_vol:>5.1%} | "
            f"Sharpe {self.sharpe:>5.2f} | "
            f"Sortino {self.sortino:>5.2f} | "
            f"maxDD {self.max_drawdown:>6.1%} | "
            f"Calmar {self.calmar:>5.2f} | "
            f"trades {self.n_trades:>3} | "
            f"win {self.win_rate:>5.1%}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Full backtest result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    """
    Output of Backtester.run() / run_portfolio().

    Key attributes
    --------------
    net_returns  : pd.Series   — daily net return on gross notional
    equity       : pd.Series   — compounded equity curve ($)
    drawdown     : pd.Series   — running drawdown from peak
    trades       : list[Trade] — round-trip ledger
    full / in_sample / out_sample : PerfMetrics
    """
    label          : str
    initial_capital: float
    net_returns    : pd.Series
    gross_returns  : pd.Series
    cost_returns   : pd.Series
    position       : pd.Series
    equity         : pd.Series
    drawdown       : pd.Series
    trades         : List[Trade]
    full           : PerfMetrics
    in_sample      : Optional[PerfMetrics] = None
    out_sample     : Optional[PerfMetrics] = None

    # ── Convenience properties ──────────────────────────────────────────────
    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1]) if not self.equity.empty else self.initial_capital

    @property
    def total_return(self) -> float:
        return self.full.total_return

    @property
    def sharpe(self) -> float:
        return self.full.sharpe

    @property
    def max_drawdown(self) -> float:
        return self.full.max_drawdown

    def to_dataframe(self) -> pd.DataFrame:
        """Aligned equity/return/position frame."""
        return pd.DataFrame({
            "position"     : self.position,
            "gross_return" : self.gross_returns,
            "cost_return"  : self.cost_returns,
            "net_return"   : self.net_returns,
            "equity"       : self.equity,
            "drawdown"     : self.drawdown,
        })

    def trades_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_series() for t in self.trades])

    def summary(self) -> str:
        lines = [
            f"\n{'═'*98}",
            f"  Backtest — {self.label}",
            f"  initial ${self.initial_capital:,.0f}  →  final ${self.final_equity:,.0f}",
            f"{'═'*98}",
            self.full.line(),
        ]
        if self.in_sample is not None:
            lines.append(self.in_sample.line())
        if self.out_sample is not None:
            lines.append(self.out_sample.line())
        lines.append(f"{'═'*98}")
        output = "\n".join(lines)
        print(output)
        return output


# ══════════════════════════════════════════════════════════════════════════════
# Metric computation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _max_drawdown(equity: pd.Series) -> Tuple[pd.Series, float]:
    """Running drawdown series and the worst (most negative) value."""
    if equity.empty:
        return equity, 0.0
    peak = equity.cummax()
    dd   = equity / peak - 1.0
    return dd, float(dd.min())


def _annualised_return(net_ret: pd.Series) -> float:
    return float(net_ret.mean() * TRADING_DAYS) if not net_ret.empty else 0.0


def _sharpe(net_ret: pd.Series, rf: float) -> float:
    if net_ret.empty:
        return 0.0
    ann_ret = _annualised_return(net_ret)
    ann_vol = float(net_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return (ann_ret - rf) / ann_vol if ann_vol > 1e-12 else 0.0


def _sortino(net_ret: pd.Series, rf: float) -> float:
    if net_ret.empty:
        return 0.0
    ann_ret  = _annualised_return(net_ret)
    downside = net_ret[net_ret < 0]
    dvol     = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else 0.0
    return (ann_ret - rf) / dvol if dvol > 1e-12 else 0.0


def _compute_metrics(
    label    : str,
    net_ret  : pd.Series,
    position : pd.Series,
    trades   : List[Trade],
    rf       : float,
) -> PerfMetrics:
    """All risk/return stats for a single date window."""
    net_ret = net_ret.dropna()
    n_days  = int(len(net_ret))

    if n_days == 0:
        return PerfMetrics(
            label=label, start=None, end=None, n_days=0,
            total_return=0.0, cagr=0.0, ann_vol=0.0, sharpe=0.0, sortino=0.0,
            max_drawdown=0.0, calmar=0.0, time_in_market=0.0,
            n_trades=0, win_rate=0.0, avg_trade_ret=0.0, avg_hold_days=0.0,
            turnover=0.0,
        )

    equity        = (1.0 + net_ret).cumprod()
    total_return  = float(equity.iloc[-1] - 1.0)
    years         = n_days / TRADING_DAYS
    cagr          = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    ann_vol       = float(net_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
    _, max_dd     = _max_drawdown(equity)
    calmar        = cagr / abs(max_dd) if max_dd < -1e-12 else 0.0

    pos_win       = position.reindex(net_ret.index).fillna(0)
    time_in_mkt   = float((pos_win != 0).mean())
    turnover      = float(pos_win.diff().abs().fillna(pos_win.abs()).sum())

    win_trades    = [t for t in trades if t.pnl_return > 0]
    n_trades      = len(trades)
    win_rate      = len(win_trades) / n_trades if n_trades else 0.0
    avg_trade_ret = float(np.mean([t.pnl_return for t in trades])) if trades else 0.0
    avg_hold      = float(np.mean([t.holding_days for t in trades])) if trades else 0.0

    return PerfMetrics(
        label=label, start=net_ret.index[0], end=net_ret.index[-1], n_days=n_days,
        total_return=total_return, cagr=cagr, ann_vol=ann_vol,
        sharpe=_sharpe(net_ret, rf), sortino=_sortino(net_ret, rf),
        max_drawdown=max_dd, calmar=calmar, time_in_market=time_in_mkt,
        n_trades=n_trades, win_rate=win_rate, avg_trade_ret=avg_trade_ret,
        avg_hold_days=avg_hold, turnover=turnover,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Backtester
# ══════════════════════════════════════════════════════════════════════════════

class Backtester:
    """
    Vectorised pairs backtester.

    Parameters
    ----------
    initial_capital : float — starting NAV ($). None → config value.
    risk_free_rate  : float — annual rf for Sharpe/Sortino. None → config.
    slippage_bps    : float — per-side slippage on each rebalance. None → config.

    Methods
    -------
    run(signal_result)              → BacktestResult  (single pair)
    run_portfolio(results, weights) → BacktestResult  (equal/weighted blend)
    """

    def __init__(
        self,
        initial_capital : Optional[float] = None,
        risk_free_rate  : Optional[float] = None,
        slippage_bps    : Optional[float] = None,
    ) -> None:
        cfg = get_config()
        self.initial_capital = initial_capital if initial_capital is not None else cfg.backtest.initial_capital
        self.risk_free_rate  = risk_free_rate  if risk_free_rate  is not None else cfg.backtest.risk_free_rate
        self.slippage        = (slippage_bps if slippage_bps is not None else cfg.risk.slippage_bps) / 10_000

        self.in_sample_end    = pd.Timestamp(cfg.backtest.in_sample_end)
        self.out_sample_start = pd.Timestamp(cfg.backtest.out_sample_start)

        logger.info(
            "Backtester | capital=${:,.0f} | rf={:.1%} | slippage={:.1f}bps | "
            "IS≤{} | OOS≥{}",
            self.initial_capital, self.risk_free_rate, self.slippage * 10_000,
            self.in_sample_end.date(), self.out_sample_start.date(),
        )

    # ── Core return engine ──────────────────────────────────────────────────

    def _pair_returns(
        self, res: SignalResult
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Causal gross / cost / net return series for one pair.

        Returns (gross_ret, cost_ret, net_ret), all indexed by date.
        """
        y    = res.y.astype(float)
        x    = res.x.astype(float)
        beta = res.hedge_ratios.astype(float)
        pos  = res.position.astype(float)

        dy = y.diff()
        dx = x.diff()

        pos_lag  = pos.shift(1).fillna(0.0)
        beta_lag = beta.shift(1)

        # Dollar P&L per 1 share y / β shares x, position & β both lagged.
        unit_pnl = pos_lag * (dy - beta_lag * dx)

        # Gross notional deployed at t-1 (both legs).
        gross_lag = y.shift(1).abs() + (beta_lag * x.shift(1)).abs()
        gross_ret = (unit_pnl / gross_lag).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Cost: slippage on traded notional whenever position changes.
        dpos     = pos.diff().abs().fillna(pos.abs())
        cost_ret = self.slippage * dpos

        net_ret = (gross_ret - cost_ret).rename("net_return")
        return gross_ret.rename("gross_return"), cost_ret.rename("cost_return"), net_ret

    def _build_trades(
        self, res: SignalResult, net_ret: pd.Series
    ) -> List[Trade]:
        """Walk the position series into round-trip Trade records."""
        pos  = res.position.astype(int)
        z    = res.zscore
        stop = res.zscore_stop
        trades: List[Trade] = []

        in_pos    = 0
        entry_idx = None
        entry_z   = np.nan

        idx = pos.index
        for i in range(len(pos)):
            p = int(pos.iloc[i])
            if in_pos == 0 and p != 0:
                # Opened a position this bar.
                in_pos, entry_idx, entry_z = p, idx[i], float(z.iloc[i])
            elif in_pos != 0 and p != in_pos:
                # Closed (to flat) or flipped — book the round-trip.
                exit_idx = idx[i]
                exit_z   = float(z.iloc[i])
                window   = net_ret.loc[entry_idx:exit_idx]
                ret      = float((1.0 + window).prod() - 1.0)
                hold     = int(idx.get_loc(exit_idx) - idx.get_loc(entry_idx))
                reason   = "stop" if abs(exit_z) >= stop else "exit"
                trades.append(Trade(
                    pair_key=res.pair_key, direction=in_pos,
                    entry_date=entry_idx, exit_date=exit_idx,
                    holding_days=hold, entry_zscore=entry_z, exit_zscore=exit_z,
                    pnl_return=ret, reason=reason,
                ))
                # Handle a direct flip (rare): immediately re-enter.
                if p != 0:
                    in_pos, entry_idx, entry_z = p, idx[i], float(z.iloc[i])
                else:
                    in_pos, entry_idx, entry_z = 0, None, np.nan

        # Position still open at the end of the sample.
        if in_pos != 0 and entry_idx is not None:
            exit_idx = idx[-1]
            window   = net_ret.loc[entry_idx:exit_idx]
            ret      = float((1.0 + window).prod() - 1.0)
            hold     = int(idx.get_loc(exit_idx) - idx.get_loc(entry_idx))
            trades.append(Trade(
                pair_key=res.pair_key, direction=in_pos,
                entry_date=entry_idx, exit_date=exit_idx,
                holding_days=hold, entry_zscore=entry_z,
                exit_zscore=float(z.iloc[-1]), pnl_return=ret, reason="open",
            ))
        return trades

    def _assemble(
        self,
        label    : str,
        net_ret  : pd.Series,
        gross_ret: pd.Series,
        cost_ret : pd.Series,
        position : pd.Series,
        trades   : List[Trade],
    ) -> BacktestResult:
        """Build equity curve, windowed metrics, and the result object."""
        net_ret     = net_ret.fillna(0.0)
        equity      = self.initial_capital * (1.0 + net_ret).cumprod()
        drawdown, _ = _max_drawdown(equity)

        def window(lo: Optional[pd.Timestamp], hi: Optional[pd.Timestamp]) -> Tuple[pd.Series, List[Trade]]:
            mask = pd.Series(True, index=net_ret.index)
            if lo is not None:
                mask &= net_ret.index >= lo
            if hi is not None:
                mask &= net_ret.index <= hi
            sub_ret    = net_ret[mask]
            sub_trades = [t for t in trades if (lo is None or t.entry_date >= lo)
                          and (hi is None or t.entry_date <= hi)]
            return sub_ret, sub_trades

        rf = self.risk_free_rate

        full = _compute_metrics("FULL", net_ret, position, trades, rf)

        is_ret, is_trades = window(None, self.in_sample_end)
        in_sample = (_compute_metrics("IN-SAMPLE", is_ret, position, is_trades, rf)
                     if len(is_ret) else None)

        oos_ret, oos_trades = window(self.out_sample_start, None)
        out_sample = (_compute_metrics("OUT-SAMPLE", oos_ret, position, oos_trades, rf)
                      if len(oos_ret) else None)

        return BacktestResult(
            label=label, initial_capital=self.initial_capital,
            net_returns=net_ret, gross_returns=gross_ret.fillna(0.0),
            cost_returns=cost_ret.fillna(0.0), position=position,
            equity=equity, drawdown=drawdown, trades=trades,
            full=full, in_sample=in_sample, out_sample=out_sample,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self, res: SignalResult, label: Optional[str] = None) -> BacktestResult:
        """Backtest a single pair's SignalResult."""
        lbl = label or f"{res.y_name}/{res.x_name} [{res.hedge_method}]"

        if not res.passed_gate or res.position.empty:
            logger.warning("Backtest skipped — gate failed / no positions: {}", lbl)
            empty = pd.Series(dtype=float)
            flat  = _compute_metrics("FULL", empty, empty, [], self.risk_free_rate)
            return BacktestResult(
                label=lbl, initial_capital=self.initial_capital,
                net_returns=empty, gross_returns=empty, cost_returns=empty,
                position=res.position, equity=empty, drawdown=empty,
                trades=[], full=flat,
            )

        gross_ret, cost_ret, net_ret = self._pair_returns(res)
        trades = self._build_trades(res, net_ret)
        result = self._assemble(lbl, net_ret, gross_ret, cost_ret,
                                res.position.astype(float), trades)

        logger.info(
            "Backtest {} | total={:.1%} | Sharpe={:.2f} | maxDD={:.1%} | trades={}",
            lbl, result.total_return, result.sharpe,
            result.max_drawdown, result.full.n_trades,
        )
        return result

    def run_portfolio(
        self,
        results : Dict[str, SignalResult],
        weights : Optional[Dict[str, float]] = None,
        label   : str = "PORTFOLIO",
    ) -> BacktestResult:
        """
        Blend several pairs into one portfolio.

        Each pair contributes its net return series; the portfolio return is
        the weighted average (default: equal weight) across pairs. Trade
        ledgers are concatenated for aggregate win-rate stats.
        """
        tradable = {k: r for k, r in results.items() if r.passed_gate and not r.position.empty}
        if not tradable:
            logger.warning("Portfolio backtest — no tradable pairs")
            empty = pd.Series(dtype=float)
            flat  = _compute_metrics("FULL", empty, empty, [], self.risk_free_rate)
            return BacktestResult(
                label=label, initial_capital=self.initial_capital,
                net_returns=empty, gross_returns=empty, cost_returns=empty,
                position=empty, equity=empty, drawdown=empty, trades=[], full=flat,
            )

        if weights is None:
            w = {k: 1.0 / len(tradable) for k in tradable}
        else:
            tot = sum(weights[k] for k in tradable)
            w   = {k: weights[k] / tot for k in tradable}

        net_parts, gross_parts, cost_parts, pos_parts = {}, {}, {}, {}
        all_trades: List[Trade] = []
        for k, r in tradable.items():
            g, c, n = self._pair_returns(r)
            net_parts[k]   = n * w[k]
            gross_parts[k] = g * w[k]
            cost_parts[k]  = c * w[k]
            pos_parts[k]   = (r.position != 0).astype(float)
            all_trades.extend(self._build_trades(r, n))

        net_ret   = pd.DataFrame(net_parts).fillna(0.0).sum(axis=1).sort_index()
        gross_ret = pd.DataFrame(gross_parts).fillna(0.0).sum(axis=1).sort_index()
        cost_ret  = pd.DataFrame(cost_parts).fillna(0.0).sum(axis=1).sort_index()
        # Portfolio counts as "in market" if any pair holds a position.
        position  = (pd.DataFrame(pos_parts).fillna(0.0).sum(axis=1).sort_index() > 0).astype(float)

        all_trades.sort(key=lambda t: t.entry_date)
        result = self._assemble(label, net_ret, gross_ret, cost_ret, position, all_trades)

        logger.info(
            "Portfolio backtest | {} pairs | total={:.1%} | Sharpe={:.2f} | maxDD={:.1%}",
            len(tradable), result.total_return, result.sharpe, result.max_drawdown,
        )
        return result


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from src.signals import SignalEngine

    print("\n" + "=" * 60)
    print("  BACKTEST — SMOKE TEST")
    print("=" * 60)

    # ── Build a synthetic cointegrated pair ───────────────────────────────────
    # x = random walk; spread = mean-reverting OU; y = β·x + spread.
    rng   = np.random.default_rng(7)
    n     = 1500
    dates = pd.bdate_range("2018-01-01", periods=n)

    x_level   = 100 + np.cumsum(rng.normal(0, 1.0, n))
    beta_true = 0.8
    spread    = np.zeros(n)
    kappa, sigma = 0.05, 1.0          # OU mean-reversion speed / noise
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1 - kappa) + rng.normal(0, sigma)
    y_level = beta_true * x_level + spread

    y = pd.Series(y_level, index=dates, name="Y")
    x = pd.Series(x_level, index=dates, name="X")

    def overlay_zscore_positions(result, entry=2.0, exit_=0.5):
        """
        Replace the engine's gated positions with a clean ±entry / exit
        z-score rule so the BACKTESTER is exercised deterministically,
        independent of the half-life / regime gates (which depend on the
        chosen hedge method). This keeps the smoke test focused on P&L.
        """
        z   = result.zscore.fillna(0.0)
        pos = np.zeros(len(z), dtype=int)
        cur = 0
        for i, zt in enumerate(z.values):
            if cur == 0:
                if zt < -entry:  cur = 1
                elif zt > entry: cur = -1
            elif abs(zt) < exit_:
                cur = 0
            pos[i] = cur
        result.position = pd.Series(pos, index=z.index, name="position")
        pos_prev = result.position.shift(1).fillna(0).astype(int)
        result.trade_entries = (pos_prev == 0) & (result.position != 0)
        result.trade_exits   = (pos_prev != 0) & (result.position == 0)
        return result

    # ── Run the signal engine for a real spread, then overlay positions ───────
    print("\n── Generating spread (static_ols, gate=skip) + z-score overlay ──")
    res = SignalEngine(
        hedge_method="static_ols", pair_key="gold_silver", gate_mode="skip",
    ).run(y, x, y_name="Y", x_name="X")
    res = overlay_zscore_positions(res)
    print(f"  positions taken : {int(res.trade_entries.sum())} entries | "
          f"time in market {res.time_in_market:.0%}")

    # ── Test 1: single-pair backtest ──────────────────────────────────────────
    print("\n── Test 1: single-pair backtest ──")
    bt = Backtester(initial_capital=100_000).run(res, label="Y/X synthetic")
    bt.summary()

    assert not bt.equity.empty,                      "❌ Equity curve is empty"
    assert np.isfinite(bt.final_equity),             "❌ Final equity not finite"
    assert bt.full.n_trades > 0,                     "❌ Expected at least one trade"
    assert np.isfinite(bt.sharpe),                   "❌ Sharpe not finite"
    assert -1.0 <= bt.max_drawdown <= 0.0,           "❌ Drawdown out of range"
    assert len(bt.net_returns) == len(res.position), "❌ Return/position length mismatch"
    print("✅ PASS: single-pair backtest produced a valid equity curve")

    # ── Test 2: causality (no look-ahead) ─────────────────────────────────────
    print("\n── Test 2: causality / lagged exposure ──")
    # First bar can never earn P&L (no prior position).
    assert bt.net_returns.iloc[0] == 0.0, "❌ Non-zero P&L on first bar (look-ahead!)"
    # Idle bars (flat with no trade) must have exactly zero return.
    pos_lag = res.position.shift(1).fillna(0)
    dpos    = res.position.diff().abs().fillna(res.position.abs())
    idle    = (pos_lag == 0) & (dpos == 0)
    assert (bt.net_returns[idle].abs() < 1e-12).all(), "❌ P&L leaked on idle bars"
    print("✅ PASS: returns are strictly causal (lagged position & β)")

    # ── Test 3: costs reduce returns ──────────────────────────────────────────
    print("\n── Test 3: transaction costs drag on P&L ──")
    bt_nocost = Backtester(initial_capital=100_000, slippage_bps=0.0).run(res)
    print(f"  gross (0 bps)  : {bt_nocost.total_return:>7.2%}")
    print(f"  net  (cfg bps) : {bt.total_return:>7.2%}")
    assert bt_nocost.total_return >= bt.total_return - 1e-9, \
        "❌ Zero-cost return should be ≥ costed return"
    assert (bt.cost_returns >= -1e-12).all(), "❌ Cost returns should be non-negative"
    print("✅ PASS: slippage costs correctly drag on returns")

    # ── Test 4: trade ledger integrity ────────────────────────────────────────
    print("\n── Test 4: trade ledger ──")
    tdf = bt.trades_dataframe()
    print(f"  trades booked : {len(tdf)}")
    print(f"  win rate      : {bt.full.win_rate:.1%}")
    print(f"  avg hold      : {bt.full.avg_hold_days:.1f} days")
    assert len(tdf) == bt.full.n_trades,         "❌ Ledger length mismatch"
    assert (tdf["holding_days"] >= 0).all(),     "❌ Negative holding period"
    assert tdf["direction"].isin([-1, 1]).all(), "❌ Bad trade direction"
    print("✅ PASS: trade ledger is internally consistent")

    # ── Test 5: walk-forward windows ──────────────────────────────────────────
    print("\n── Test 5: in-sample / out-of-sample split ──")
    assert bt.in_sample is not None,  "❌ Missing in-sample metrics"
    assert bt.out_sample is not None, "❌ Missing out-of-sample metrics"
    assert bt.in_sample.n_days + bt.out_sample.n_days <= bt.full.n_days, \
        "❌ IS + OOS day count exceeds full sample"
    print(f"  IS  days: {bt.in_sample.n_days:>4} | OOS days: {bt.out_sample.n_days:>4}")
    print("✅ PASS: walk-forward windows computed")

    # ── Test 6: two-pair portfolio ────────────────────────────────────────────
    print("\n── Test 6: equal-weight portfolio ──")
    res2 = SignalEngine(
        hedge_method="static_ols", pair_key="wti_brent", gate_mode="skip",
    ).run(y * 1.01 + 2.0, x, y_name="Y2", x_name="X")
    res2 = overlay_zscore_positions(res2)
    port = Backtester(initial_capital=100_000).run_portfolio(
        {"gold_silver": res, "wti_brent": res2}
    )
    port.summary()
    assert not port.equity.empty,                  "❌ Portfolio equity empty"
    assert np.isfinite(port.sharpe),               "❌ Portfolio Sharpe not finite"
    assert port.full.n_trades >= bt.full.n_trades, "❌ Portfolio should aggregate trades"
    print("✅ PASS: portfolio blend produced a valid curve")

    print("\n" + "=" * 60)
    print("  ALL BACKTEST SMOKE TESTS PASSED ✅")
    print("=" * 60 + "\n")
