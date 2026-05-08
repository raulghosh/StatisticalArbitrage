"""
monitor.py — Runtime Risk Monitor

Tracks portfolio health in real time and approves or blocks
orders before they reach the execution layer.

Responsibilities:
  1. Drawdown tracking — current DD vs high-water mark
  2. Exposure limits   — total pair exposure as % of NAV
  3. Per-pair P&L      — unrealised and realised
  4. Kill switch       — blocks all new entries if max_drawdown breached
  5. Trade log         — full audit trail of all fills

Design:
  The monitor sits between the sizer and the execution layer.
  Every TradeOrder passes through monitor.approve() before
  being sent to AlpacaTrader.

  monitor.approve(order) → approved TradeOrder or blocked TradeOrder

Run directly:
    python -m src.risk.monitor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_config
from src.risk.sizer import TradeOrder
from src.signals.base import SignalAction


# ══════════════════════════════════════════════════════════════════════════════
# Position record
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OpenPosition:
    """
    Tracks a live pair position.

    Created when ENTER is executed.
    Closed when EXIT or STOP is executed.
    """
    pair_key       : str
    direction      : int                  # +1 long spread | -1 short spread
    entry_time     : datetime
    leg1_symbol    : str
    leg2_symbol    : str
    leg1_shares    : int
    leg2_shares    : int
    leg1_entry_px  : float
    leg2_entry_px  : float
    hedge_ratio    : float
    capital_per_leg: float

    exit_time      : Optional[datetime] = None
    leg1_exit_px   : float = 0.0
    leg2_exit_px   : float = 0.0
    realised_pnl   : float = 0.0
    exit_action    : str = ""             # "EXIT" | "STOP" | "SUSPEND"

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    def mark_to_market(
        self,
        leg1_price: float,
        leg2_price: float,
    ) -> float:
        """
        Unrealised P&L at current prices.

        Long spread:  profit when leg1 rises and/or leg2 falls
        Short spread: profit when leg1 falls and/or leg2 rises
        """
        leg1_pnl = self.leg1_shares * (leg1_price - self.leg1_entry_px)
        leg2_pnl = self.leg2_shares * (leg2_price - self.leg2_entry_px)

        if self.direction == 1:     # Long: long leg1, short leg2
            return leg1_pnl - leg2_pnl
        else:                       # Short: short leg1, long leg2
            return -leg1_pnl + leg2_pnl

    def close(
        self,
        leg1_exit_px : float,
        leg2_exit_px : float,
        exit_action  : str,
        exit_time    : Optional[datetime] = None,
    ) -> float:
        """Close the position and compute realised P&L."""
        self.exit_time   = exit_time or datetime.now(timezone.utc)
        self.leg1_exit_px = leg1_exit_px
        self.leg2_exit_px = leg2_exit_px
        self.exit_action  = exit_action
        self.realised_pnl = self.mark_to_market(leg1_exit_px, leg2_exit_px)
        return self.realised_pnl

    @property
    def hold_days(self) -> float:
        end = self.exit_time or datetime.now(timezone.utc)
        return (end - self.entry_time).total_seconds() / 86400

    def to_dict(self) -> dict:
        return {
            "pair_key"       : self.pair_key,
            "direction"      : self.direction,
            "entry_time"     : self.entry_time.isoformat(),
            "exit_time"      : self.exit_time.isoformat() if self.exit_time else None,
            "hold_days"      : round(self.hold_days, 2),
            "leg1_symbol"    : self.leg1_symbol,
            "leg2_symbol"    : self.leg2_symbol,
            "leg1_shares"    : self.leg1_shares,
            "leg2_shares"    : self.leg2_shares,
            "leg1_entry_px"  : round(self.leg1_entry_px, 4),
            "leg2_entry_px"  : round(self.leg2_entry_px, 4),
            "leg1_exit_px"   : round(self.leg1_exit_px, 4),
            "leg2_exit_px"   : round(self.leg2_exit_px, 4),
            "capital_per_leg": round(self.capital_per_leg, 2),
            "realised_pnl"   : round(self.realised_pnl, 2),
            "exit_action"    : self.exit_action,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio snapshot
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RiskSnapshot:
    """Current risk state — emitted daily by the monitor."""
    timestamp          : datetime
    equity             : float
    high_water_mark    : float
    current_drawdown   : float          # as fraction: 0.05 = 5% DD
    max_drawdown_seen  : float
    kill_switch_active : bool
    n_open_positions   : int
    total_exposure     : float          # total notional as % of equity
    realised_pnl_total : float
    unrealised_pnl     : float
    pair_exposures     : Dict[str, float]   # pair → notional $ in position

    @property
    def is_healthy(self) -> bool:
        return not self.kill_switch_active and self.current_drawdown < 0.10

    def summary(self) -> str:
        kill_str = "🔴 ACTIVE" if self.kill_switch_active else "🟢 off"
        lines = [
            f"\n{'─'*60}",
            f"  Risk Snapshot — {self.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            f"{'─'*60}",
            f"  Portfolio Equity     : ${self.equity:>12,.2f}",
            f"  High-Water Mark      : ${self.high_water_mark:>12,.2f}",
            f"  Current Drawdown     : {self.current_drawdown:>10.2%}",
            f"  Max Drawdown Seen    : {self.max_drawdown_seen:>10.2%}",
            f"  Kill Switch          : {kill_str}",
            f"",
            f"  Open Positions       : {self.n_open_positions}",
            f"  Total Exposure       : {self.total_exposure:>10.2%} of equity",
            f"  Realised P&L         : ${self.realised_pnl_total:>12,.2f}",
            f"  Unrealised P&L       : ${self.unrealised_pnl:>12,.2f}",
        ]
        if self.pair_exposures:
            lines.append(f"")
            lines.append(f"  Pair Exposures:")
            for pair, exp in self.pair_exposures.items():
                lines.append(f"    {pair:<20} : ${exp:>10,.2f}")
        lines.append(f"{'─'*60}")
        output = "\n".join(lines)
        print(output)
        return output


# ══════════════════════════════════════════════════════════════════════════════
# Risk Monitor
# ══════════════════════════════════════════════════════════════════════════════

class RiskMonitor:
    """
    Runtime risk monitor and order gate.

    All TradeOrders pass through approve() before execution.
    The monitor tracks positions, P&L, and drawdown and
    blocks orders when risk limits are breached.

    Parameters
    ----------
    initial_equity     : float  — starting portfolio NAV ($)
    max_drawdown_pct   : float  — kill switch threshold (default 0.15 = 15%)
    max_exposure_pct   : float  — max total notional as % NAV (default 0.40)
    max_pairs_open     : int    — max simultaneous open pairs (default 2)
    slippage_bps       : float  — slippage per side in basis points
    """

    def __init__(
        self,
        initial_equity   : float,
        max_drawdown_pct : Optional[float] = None,
        max_exposure_pct : float = 0.40,
        max_pairs_open   : int   = 2,
        slippage_bps     : Optional[float] = None,
    ) -> None:
        cfg = get_config().risk

        self.equity          = initial_equity
        self.high_water_mark = initial_equity
        self.max_dd_pct      = max_drawdown_pct or cfg.max_drawdown_pct
        self.max_exposure    = max_exposure_pct
        self.max_pairs_open  = max_pairs_open
        self.slippage_bps    = (slippage_bps or cfg.slippage_bps) / 10_000

        self._open_positions  : Dict[str, OpenPosition] = {}
        self._closed_positions: List[OpenPosition]      = []
        self._equity_curve    : List[tuple]              = [
            (datetime.now(timezone.utc), initial_equity)
        ]
        self._kill_switch     = False
        self._max_dd_seen     = 0.0

        logger.info(
            "RiskMonitor | equity=${:,.0f} | max_dd={:.0%} | "
            "max_exposure={:.0%} | max_pairs={}",
            initial_equity, self.max_dd_pct,
            self.max_exposure, self.max_pairs_open,
        )

    # ── Approve gate ───────────────────────────────────────────────────────

    def approve(self, order: TradeOrder) -> TradeOrder:
        """
        Gate every TradeOrder through risk checks.

        Checks (in order):
          1. Kill switch — block all entries if DD > threshold
          2. Max pairs   — don't open more pairs than allowed
          3. Max exposure — total notional cap
          4. Duplicate   — already have a position in this pair

        Exit / Stop / Suspend orders always pass through
        (you must always be able to exit a position).

        Returns the original order with approved=True/False.
        """
        action = order.signal_event.action

        # ── Exits always pass ──────────────────────────────────────────────
        if action in (SignalAction.EXIT, SignalAction.STOP, SignalAction.SUSPEND):
            order.approved    = True
            order.block_reason = ""
            return order

        # ── Non-entry passes silently ──────────────────────────────────────
        if action in (SignalAction.WAIT, SignalAction.HOLD):
            order.approved    = False
            order.block_reason = "no action"
            return order

        # ── Entry checks ───────────────────────────────────────────────────
        if action == SignalAction.ENTER:

            # Check 1: Kill switch
            if self._kill_switch:
                return self._block(order, "kill switch active — max drawdown breached")

            # Check 2: Already in this pair
            if order.pair_key in self._open_positions:
                return self._block(order, f"already have open position in {order.pair_key}")

            # Check 3: Max open pairs
            if len(self._open_positions) >= self.max_pairs_open:
                return self._block(
                    order,
                    f"max open pairs reached ({self.max_pairs_open})",
                )

            # Check 4: Exposure limit
            current_exposure = self._total_exposure()
            new_exposure     = current_exposure + order.total_notional
            if new_exposure / self.equity > self.max_exposure:
                return self._block(
                    order,
                    f"exposure limit: {new_exposure/self.equity:.1%} > {self.max_exposure:.0%}",
                )

            # Check 5: Minimum size
            if order.capital_per_leg < 100:
                return self._block(order, f"order too small: ${order.capital_per_leg:.0f}")

            order.approved    = True
            order.block_reason = ""
            logger.info(
                "APPROVED | {} | ${:,.0f}/leg | exposure={:.1%}",
                order.pair_key,
                order.capital_per_leg,
                new_exposure / self.equity,
            )
            return order

        return self._block(order, f"unknown action: {action}")

    @staticmethod
    def _block(order: TradeOrder, reason: str) -> TradeOrder:
        order.approved    = False
        order.block_reason = reason
        logger.debug("BLOCKED | {} | {}", order.pair_key, reason)
        return order

    # ── Position tracking ──────────────────────────────────────────────────

    def record_entry(
        self,
        order      : TradeOrder,
        leg1_symbol: str,
        leg2_symbol: str,
    ) -> OpenPosition:
        """
        Register a filled entry order.
        Call after AlpacaTrader confirms the fill.
        """
        pos = OpenPosition(
            pair_key        = order.pair_key,
            direction       = order.signal_event.direction,
            entry_time      = order.signal_event.timestamp,
            leg1_symbol     = leg1_symbol,
            leg2_symbol     = leg2_symbol,
            leg1_shares     = order.leg1_shares,
            leg2_shares     = order.leg2_shares,
            leg1_entry_px   = order.leg1_price,
            leg2_entry_px   = order.leg2_price,
            hedge_ratio     = order.hedge_ratio,
            capital_per_leg = order.capital_per_leg,
        )
        self._open_positions[order.pair_key] = pos
        logger.info(
            "Position opened | {} | dir={:+d} | ${:,.0f}/leg",
            order.pair_key, pos.direction, pos.capital_per_leg,
        )
        return pos

    def record_exit(
        self,
        pair_key    : str,
        leg1_exit_px: float,
        leg2_exit_px: float,
        action      : str,
    ) -> Optional[OpenPosition]:
        """
        Register a filled exit order and update equity.
        Call after AlpacaTrader confirms the fill.
        """
        pos = self._open_positions.pop(pair_key, None)
        if pos is None:
            logger.warning("record_exit called but no open position for {}", pair_key)
            return None

        # Apply slippage to exit prices
        sl = self.slippage_bps
        leg1_fill = leg1_exit_px * (1 - sl) if pos.direction == 1 else leg1_exit_px * (1 + sl)
        leg2_fill = leg2_exit_px * (1 + sl) if pos.direction == 1 else leg2_exit_px * (1 - sl)

        pnl = pos.close(leg1_fill, leg2_fill, action)
        self._closed_positions.append(pos)

        # Update equity and drawdown
        self.equity += pnl
        self._update_drawdown()
        self._equity_curve.append((datetime.now(timezone.utc), self.equity))

        logger.info(
            "Position closed | {} | action={} | pnl=${:,.2f} | "
            "hold={:.1f}d | equity=${:,.0f}",
            pair_key, action, pnl, pos.hold_days, self.equity,
        )
        return pos

    def update_equity(self, equity: float) -> None:
        """
        Update equity with latest mark-to-market value.
        Call daily with portfolio value from broker.
        """
        self.equity = equity
        self._update_drawdown()
        self._equity_curve.append((datetime.now(timezone.utc), equity))

    # ── Drawdown tracking ──────────────────────────────────────────────────

    def _update_drawdown(self) -> None:
        if self.equity > self.high_water_mark:
            self.high_water_mark = self.equity

        current_dd = (self.high_water_mark - self.equity) / self.high_water_mark
        self._max_dd_seen = max(self._max_dd_seen, current_dd)

        # Kill switch
        if current_dd > self.max_dd_pct and not self._kill_switch:
            self._kill_switch = True
            logger.critical(
                "KILL SWITCH ACTIVATED | drawdown={:.2%} > max={:.2%} | "
                "all new entries blocked",
                current_dd, self.max_dd_pct,
            )

        # Reset kill switch if equity recovers to 90% of HWM
        if self._kill_switch and current_dd < self.max_dd_pct * 0.5:
            self._kill_switch = False
            logger.info(
                "Kill switch RESET | drawdown recovered to {:.2%}",
                current_dd,
            )

    # ── Analytics ──────────────────────────────────────────────────────────

    def _total_exposure(self) -> float:
        """Total notional in open positions ($)."""
        return sum(
            p.capital_per_leg * 2
            for p in self._open_positions.values()
        )

    def get_snapshot(self, unrealised_pnl: float = 0.0) -> RiskSnapshot:
        """Return current risk state."""
        current_dd = (
            (self.high_water_mark - self.equity) / self.high_water_mark
            if self.high_water_mark > 0 else 0.0
        )
        pair_exposures = {
            k: v.capital_per_leg * 2
            for k, v in self._open_positions.items()
        }
        realised_total = sum(p.realised_pnl for p in self._closed_positions)

        return RiskSnapshot(
            timestamp          = datetime.now(timezone.utc),
            equity             = self.equity,
            high_water_mark    = self.high_water_mark,
            current_drawdown   = float(current_dd),
            max_drawdown_seen  = float(self._max_dd_seen),
            kill_switch_active = self._kill_switch,
            n_open_positions   = len(self._open_positions),
            total_exposure     = self._total_exposure() / self.equity if self.equity > 0 else 0,
            realised_pnl_total = realised_total,
            unrealised_pnl     = unrealised_pnl,
            pair_exposures     = pair_exposures,
        )

    def get_trade_log(self) -> pd.DataFrame:
        """All closed trades as a DataFrame."""
        if not self._closed_positions:
            return pd.DataFrame()
        return pd.DataFrame([p.to_dict() for p in self._closed_positions])

    def get_equity_curve(self) -> pd.Series:
        """Equity curve as a DatetimeIndex Series."""
        timestamps, values = zip(*self._equity_curve)
        return pd.Series(values, index=pd.DatetimeIndex(timestamps), name="equity")

    @property
    def kill_switch(self) -> bool:
        return self._kill_switch

    @property
    def n_open(self) -> int:
        return len(self._open_positions)

    # ── Performance summary ────────────────────────────────────────────────

    def performance_summary(self) -> dict:
        """Quick stats on closed trades."""
        trades = self.get_trade_log()
        if trades.empty:
            return {"message": "No closed trades yet."}

        winners = trades[trades["realised_pnl"] > 0]
        losers  = trades[trades["realised_pnl"] <= 0]

        return {
            "n_trades"          : len(trades),
            "n_winners"         : len(winners),
            "n_losers"          : len(losers),
            "hit_rate"          : round(len(winners) / len(trades), 4),
            "total_pnl"         : round(trades["realised_pnl"].sum(), 2),
            "avg_pnl_per_trade" : round(trades["realised_pnl"].mean(), 2),
            "avg_win"           : round(winners["realised_pnl"].mean(), 2) if len(winners) else 0,
            "avg_loss"          : round(losers["realised_pnl"].mean(), 2)  if len(losers)  else 0,
            "profit_factor"     : round(
                winners["realised_pnl"].sum() / abs(losers["realised_pnl"].sum()), 4
            ) if len(losers) and losers["realised_pnl"].sum() != 0 else float("inf"),
            "avg_hold_days"     : round(trades["hold_days"].mean(), 2),
            "max_drawdown"      : round(self._max_dd_seen, 4),
            "current_equity"    : round(self.equity, 2),
        }


# ── Run as module: python -m src.risk.monitor ─────────────────────────────
if __name__ == "__main__":
    from src.signals.base import PositionState, SignalAction, SignalEvent
    from src.risk.sizer import PositionSizer, TradeOrder
    from datetime import datetime, timezone

    print("\n" + "="*60)
    print("  RISK MONITOR — SMOKE TEST")
    print("  Simulating trade lifecycle with risk checks")
    print("="*60)

    monitor = RiskMonitor(
        initial_equity   = 100_000,
        max_drawdown_pct = 0.15,
        max_exposure_pct = 0.40,
        max_pairs_open   = 2,
    )
    sizer = PositionSizer(portfolio_equity=100_000, spread_annual_vol=0.15)

    def make_entry_order(pair: str, direction: int, strength: float) -> TradeOrder:
        event = SignalEvent(
            pair_key        = pair,
            action          = SignalAction.ENTER,
            direction       = direction,
            zscore          = -2.5 * direction,
            hedge_ratio     = 0.72,
            signal_strength = strength,
            spread_value    = -50.0,
            half_life_days  = 10.0,
            regime_ok       = True,
            coint_pvalue    = 0.02,
            prev_state      = PositionState.FLAT,
            new_state       = PositionState.LONG if direction == 1 else PositionState.SHORT,
            timestamp       = datetime.now(timezone.utc),
        )
        return sizer.size(event, leg1_price=185.0, leg2_price=22.0)

    # ── Test 1: Normal entry ───────────────────────────────────────────────
    print("\n[1] Normal entry — gold_silver:")
    order1 = make_entry_order("gold_silver", 1, 0.75)
    order1 = monitor.approve(order1)
    print(f"  Approved: {order1.approved}  |  Reason: '{order1.block_reason}'")
    if order1.approved:
        monitor.record_entry(order1, "GLD", "SLV")

    # ── Test 2: Duplicate position ────────────────────────────────────────
    print("\n[2] Duplicate position — gold_silver again:")
    order2 = make_entry_order("gold_silver", -1, 0.5)
    order2 = monitor.approve(order2)
    print(f"  Approved: {order2.approved}  |  Reason: '{order2.block_reason}'")

    # ── Test 3: Second pair ────────────────────────────────────────────────
    print("\n[3] Second pair — wti_brent:")
    order3 = make_entry_order("wti_brent", 1, 0.6)
    order3 = monitor.approve(order3)
    print(f"  Approved: {order3.approved}  |  Reason: '{order3.block_reason}'")
    if order3.approved:
        monitor.record_entry(order3, "USO", "BNO")

    # ── Test 4: Max pairs exceeded ────────────────────────────────────────
    print("\n[4] Third pair — exceeds max_pairs=2:")
    order4 = make_entry_order("fake_pair", 1, 0.8)
    order4 = monitor.approve(order4)
    print(f"  Approved: {order4.approved}  |  Reason: '{order4.block_reason}'")

    # ── Test 5: Close a position ───────────────────────────────────────────
    print("\n[5] Close gold_silver (profitable exit):")
    closed = monitor.record_exit("gold_silver", 187.0, 21.5, "EXIT")
    if closed:
        print(f"  Realised P&L: ${closed.realised_pnl:,.2f}  |  Hold: {closed.hold_days:.2f}d")

    # ── Test 6: Kill switch ────────────────────────────────────────────────
    print("\n[6] Simulating drawdown → kill switch:")
    monitor.update_equity(80_000)   # -20% drawdown → triggers kill switch
    order5 = make_entry_order("gold_silver", 1, 0.9)
    order5 = monitor.approve(order5)
    print(f"  Approved: {order5.approved}  |  Reason: '{order5.block_reason}'")

    # ── Snapshot ───────────────────────────────────────────────────────────
    print("\n[7] Risk snapshot:")
    monitor.get_snapshot().summary()

    # ── Performance ───────────────────────────────────────────────────────
    print("\n[8] Performance summary:")
    for k, v in monitor.performance_summary().items():
        print(f"  {k:<25}: {v}")