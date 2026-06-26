"""
sizer.py — Position Sizer

Translates a SignalEvent into a dollar-sized TradeOrder.

Sizing method: Target Volatility
  Capital per leg = (target_annual_vol / spread_annual_vol) * equity * max_position_pct
  Clipped to [min_notional, max_notional] guardrails.

  This ensures each pair contributes the same volatility to the portfolio
  regardless of how noisy its spread is.

TradeOrder is the canonical object passed to RiskMonitor.approve()
and then to AlpacaTrader.submit_pair_order().

Usage:
    from src.risk.sizer import PositionSizer
    sizer = PositionSizer(portfolio_equity=100_000, spread_annual_vol=0.15)
    order = sizer.size(signal_event, leg1_price=185.0, leg2_price=22.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.config import get_config
from src.signals import SignalAction, SignalEvent


# ══════════════════════════════════════════════════════════════════════════════
# Trade Order
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeOrder:
    """
    A sized, ready-to-route pair order.

    Produced by PositionSizer.size().
    Passed to RiskMonitor.approve() → AlpacaTrader.submit_pair_order().

    Attributes
    ----------
    pair_key        : str          — config pair key
    signal_event    : SignalEvent  — source signal (contains action/direction)
    capital_per_leg : float        — dollar amount to deploy in each leg
    leg1_shares     : int          — integer shares for leg1
    leg2_shares     : int          — integer shares for leg2
    leg1_price      : float        — reference price used for sizing
    leg2_price      : float        — reference price used for sizing
    hedge_ratio     : float        — β(t) from signal event
    total_notional  : float        — 2 × capital_per_leg (both legs)
    approved        : bool         — set by RiskMonitor.approve()
    block_reason    : str          — set by RiskMonitor when blocked
    """
    pair_key        : str
    signal_event    : SignalEvent
    capital_per_leg : float
    leg1_shares     : int
    leg2_shares     : int
    leg1_price      : float
    leg2_price      : float
    hedge_ratio     : float
    total_notional  : float
    approved        : bool = False
    block_reason    : str  = ""

    @property
    def action(self) -> SignalAction:
        return self.signal_event.action

    @property
    def direction(self) -> int:
        return self.signal_event.direction

    def __repr__(self) -> str:
        return (
            f"TradeOrder({self.pair_key} | {self.action.value} "
            f"dir={self.direction:+d} | "
            f"${self.capital_per_leg:,.0f}/leg | "
            f"approved={self.approved})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Position Sizer
# ══════════════════════════════════════════════════════════════════════════════

class PositionSizer:
    """
    Target-volatility position sizer for stat-arb pairs.

    Sizing formula
    --------------
    raw_capital = (target_vol / spread_vol) * equity * max_pos_pct
    capital_per_leg = clip(raw_capital, min_notional, max_notional)

    Where:
      target_vol   = cfg.risk.target_annual_vol  (default 10%)
      spread_vol   = annualised σ of the spread  (passed per-call)
      max_pos_pct  = cfg.risk.max_position_pct   (default 20% NAV per pair)
      equity       = current portfolio NAV

    Shares are integer-rounded (floor) for each leg:
      leg1_shares = floor(capital_per_leg / leg1_price)
      leg2_shares = floor(capital_per_leg * hedge_ratio / leg2_price)

    Parameters
    ----------
    portfolio_equity   : float  — current NAV ($)
    spread_annual_vol  : float  — annualised spread volatility (σ of daily
                                  spread returns × √252). If unknown, pass
                                  a conservative default (e.g. 0.20).
    target_annual_vol  : float  — target vol per pair. None → config value.
    max_position_pct   : float  — max NAV fraction per pair. None → config.
    min_notional       : float  — floor on capital_per_leg ($). Default 500.
    max_notional       : float  — ceiling on capital_per_leg ($). Default
                                  equity × max_position_pct.
    slippage_bps       : float  — applied to leg prices for sizing.
                                  None → config value.
    """

    def __init__(
        self,
        portfolio_equity  : float,
        spread_annual_vol : float = 0.20,
        target_annual_vol : Optional[float] = None,
        max_position_pct  : Optional[float] = None,
        min_notional      : float = 500.0,
        max_notional      : Optional[float] = None,
        slippage_bps      : Optional[float] = None,
    ) -> None:
        cfg = get_config().risk

        self.equity            = portfolio_equity
        self.spread_vol        = max(spread_annual_vol, 1e-6)  # avoid div/0
        self.target_vol        = target_annual_vol or cfg.target_annual_vol
        self.max_pos_pct       = max_position_pct  or cfg.max_position_pct
        self.min_notional      = min_notional
        self.max_notional      = max_notional or (portfolio_equity * self.max_pos_pct)
        self.slippage          = (slippage_bps or cfg.slippage_bps) / 10_000

        logger.info(
            "PositionSizer | equity=${:,.0f} | target_vol={:.0%} | "
            "spread_vol={:.0%} | max_pos={:.0%}",
            self.equity, self.target_vol,
            self.spread_vol, self.max_pos_pct,
        )

    def update_equity(self, equity: float) -> None:
        """Update NAV — call daily from continuous.py."""
        self.equity    = equity
        self.max_notional = equity * self.max_pos_pct

    def update_spread_vol(self, spread_annual_vol: float) -> None:
        """Update spread volatility estimate — call after each run()."""
        self.spread_vol = max(spread_annual_vol, 1e-6)

    def _raw_capital(self) -> float:
        """Target volatility sizing before guardrails."""
        return (self.target_vol / self.spread_vol) * self.equity * self.max_pos_pct

    def _calc_shares(
        self, notional: float, price: float, scale: float = 1.0
    ) -> int:
        """Integer shares = floor(notional × scale / price), min 1."""
        adjusted = notional * scale
        shares   = int(adjusted / price) if price > 0 else 0
        return max(shares, 1)

    def size(
        self,
        event      : SignalEvent,
        leg1_price : float,
        leg2_price : float,
    ) -> TradeOrder:
        """
        Size a TradeOrder from a SignalEvent.

        For EXIT / STOP / HOLD / WAIT events the notional is 0
        (execution uses existing position; shares not meaningful).

        Parameters
        ----------
        event      : SignalEvent — from SignalEngine
        leg1_price : float      — current leg1 ask/mid price
        leg2_price : float      — current leg2 ask/mid price

        Returns
        -------
        TradeOrder  (not yet approved — pass to RiskMonitor.approve())
        """
        # Non-entry events — pass through with zero size
        if event.action != SignalAction.ENTER:
            return TradeOrder(
                pair_key        = event.pair_key,
                signal_event    = event,
                capital_per_leg = 0.0,
                leg1_shares     = 0,
                leg2_shares     = 0,
                leg1_price      = leg1_price,
                leg2_price      = leg2_price,
                hedge_ratio     = event.hedge_ratio,
                total_notional  = 0.0,
                approved        = False,
                block_reason    = "",
            )

        # ── Target-vol sizing ──────────────────────────────────────────────
        raw_cap = self._raw_capital()
        capital = float(max(self.min_notional, min(raw_cap, self.max_notional)))

        # ── Apply slippage to prices ───────────────────────────────────────
        sl = self.slippage
        if event.direction == 1:   # Long spread: buy leg1, sell leg2
            p1_eff = leg1_price * (1 + sl)   # pay up to buy
            p2_eff = leg2_price * (1 - sl)   # receive less on short
        else:                       # Short spread: sell leg1, buy leg2
            p1_eff = leg1_price * (1 - sl)
            p2_eff = leg2_price * (1 + sl)

        # ── Share calculation ──────────────────────────────────────────────
        leg1_shares = self._calc_shares(capital, p1_eff, scale=1.0)
        leg2_shares = self._calc_shares(capital, p2_eff, scale=event.hedge_ratio)

        total_notional = capital * 2  # both legs

        logger.info(
            "PositionSizer | {} | cap/leg=${:,.0f} | β={:.4f} | "
            "L1: {} @ ${:.2f} | L2: {} @ ${:.2f}",
            event.pair_key, capital, event.hedge_ratio,
            leg1_shares, leg1_price, leg2_shares, leg2_price,
        )

        return TradeOrder(
            pair_key        = event.pair_key,
            signal_event    = event,
            capital_per_leg = capital,
            leg1_shares     = leg1_shares,
            leg2_shares     = leg2_shares,
            leg1_price      = leg1_price,
            leg2_price      = leg2_price,
            hedge_ratio     = event.hedge_ratio,
            total_notional  = total_notional,
        )

    def estimate_capital(self) -> float:
        """
        Capital per leg that would be deployed at current volatility.
        Useful for pre-trade checks in continuous.py.
        """
        return float(max(self.min_notional,
                         min(self._raw_capital(), self.max_notional)))


# ── Smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from datetime import datetime, timezone
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

    from src.signals import PositionState, SignalAction, SignalEvent

    print("\n" + "="*60)
    print("  POSITION SIZER — SMOKE TEST")
    print("="*60)

    def make_event(action: SignalAction, direction: int = 1) -> SignalEvent:
        return SignalEvent(
            pair_key        = "gold_silver",
            action          = action,
            direction       = direction,
            zscore          = -2.5 * direction,
            hedge_ratio     = 0.72,
            signal_strength = 0.80,
            spread_value    = -40.0,
            half_life_days  = 10.0,
            regime_ok       = True,
            quality_ok      = True,
            coint_pvalue    = 0.03,
            prev_state      = PositionState.FLAT,
            new_state       = PositionState.LONG if direction == 1 else PositionState.SHORT,
            timestamp       = datetime.now(timezone.utc),
        )

    sizer = PositionSizer(portfolio_equity=100_000, spread_annual_vol=0.15)

    # ── Test 1: ENTER order sizing ────────────────────────────────────────
    print("\n── Test 1: ENTER order ──")
    enter_event = make_event(SignalAction.ENTER, direction=1)
    order = sizer.size(enter_event, leg1_price=185.0, leg2_price=22.0)
    print(f"  {order}")
    print(f"  capital/leg : ${order.capital_per_leg:,.2f}")
    print(f"  leg1 shares : {order.leg1_shares}  @ ${order.leg1_price:.2f}")
    print(f"  leg2 shares : {order.leg2_shares}  @ ${order.leg2_price:.2f}  (β={order.hedge_ratio})")
    print(f"  total notional: ${order.total_notional:,.2f}")

    assert order.capital_per_leg > 0, "❌ Capital should be positive"
    assert order.leg1_shares >= 1,    "❌ Should have at least 1 share"
    assert order.leg2_shares >= 1,    "❌ Should have at least 1 share"
    assert order.total_notional == order.capital_per_leg * 2, "❌ Total notional mismatch"
    assert not order.approved,        "❌ Order should not be pre-approved"
    print("✅ PASS: ENTER order sized correctly")

    # ── Test 2: EXIT order (zero size) ────────────────────────────────────
    print("\n── Test 2: EXIT order ──")
    exit_event = make_event(SignalAction.EXIT)
    exit_order = sizer.size(exit_event, leg1_price=187.0, leg2_price=21.5)
    print(f"  {exit_order}")
    assert exit_order.capital_per_leg == 0.0, "❌ EXIT should have zero capital"
    assert exit_order.leg1_shares == 0, "❌ EXIT should have zero shares"
    print("✅ PASS: EXIT order has zero size")

    # ── Test 3: Vol scaling ───────────────────────────────────────────────
    print("\n── Test 3: High-vol pair gets smaller allocation ──")
    sizer_lv = PositionSizer(portfolio_equity=100_000, spread_annual_vol=0.05)
    sizer_hv = PositionSizer(portfolio_equity=100_000, spread_annual_vol=0.40)
    order_lv = sizer_lv.size(enter_event, 185.0, 22.0)
    order_hv = sizer_hv.size(enter_event, 185.0, 22.0)
    print(f"  Low-vol  (5%): ${order_lv.capital_per_leg:,.0f}/leg")
    print(f"  High-vol (40%): ${order_hv.capital_per_leg:,.0f}/leg")
    assert order_lv.capital_per_leg >= order_hv.capital_per_leg, \
        "❌ Low-vol pair should get same or larger allocation"
    print("✅ PASS: Vol scaling works correctly")

    # ── Test 4: Guardrails ────────────────────────────────────────────────
    print("\n── Test 4: Min/max notional guardrails ──")
    sizer_tiny = PositionSizer(portfolio_equity=1_000, spread_annual_vol=0.01,
                               min_notional=500)
    order_tiny = sizer_tiny.size(enter_event, 185.0, 22.0)
    print(f"  Tiny equity: capital/leg = ${order_tiny.capital_per_leg:,.0f}")
    assert order_tiny.capital_per_leg >= 500, "❌ Min notional guardrail failed"
    print("✅ PASS: Min notional guardrail applied")

    # ── Test 5: estimate_capital ──────────────────────────────────────────
    print("\n── Test 5: estimate_capital() ──")
    est = sizer.estimate_capital()
    print(f"  Estimated capital/leg: ${est:,.0f}")
    assert est > 0, "❌ estimate_capital() should return positive value"
    print("✅ PASS: estimate_capital() works")

    print("\n" + "="*60)
    print("  ALL SIZER SMOKE TESTS PASSED ✅")
    print("="*60 + "\n")
