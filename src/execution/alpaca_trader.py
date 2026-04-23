"""
alpaca_trader.py — Alpaca paper trading execution layer.

Responsibilities:
  1. Submit market / limit orders for ETF proxies
  2. Query positions and portfolio state
  3. Translate strategy signals → dollar-neutral order pairs
  4. Log all orders with full audit trail

Design principle:
  The strategy layer produces SIGNALS (+1, -1, 0).
  This module translates signals → shares → orders.
  It never makes trading decisions — it only executes.

Usage:
    from src.execution.alpaca_trader import AlpacaTrader
    trader = AlpacaTrader()
    trader.submit_pair_order("gold_silver", signal=1, capital=10_000)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from loguru import logger

from src.config import get_config


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PairOrder:
    """Represents a single pair trade — two legs submitted together."""
    pair_key: str
    signal: int                         # +1 = long spread, -1 = short spread
    leg1_symbol: str
    leg2_symbol: str
    leg1_side: OrderSide
    leg2_side: OrderSide
    leg1_shares: float
    leg2_shares: float
    leg1_price: float
    leg2_price: float
    notional: float                     # Dollar value per leg
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    leg1_order_id: Optional[str] = None
    leg2_order_id: Optional[str] = None
    status: str = "pending"             # pending | submitted | filled | rejected


@dataclass
class PortfolioSnapshot:
    """Current state of our stat-arb portfolio."""
    timestamp: datetime
    cash: float
    equity: float
    positions: Dict[str, float]         # symbol → market_value
    open_orders: int


# ══════════════════════════════════════════════════════════════════════════════
# Trader class
# ══════════════════════════════════════════════════════════════════════════════

class AlpacaTrader:
    """
    Paper trading execution for StatArb via Alpaca.

    All orders are MARKET orders at open — simplest and most realistic
    for a daily strategy. We'll add limit order support later.

    Signal convention:
        +1  → Long spread  = Buy  leg1, Sell leg2
        -1  → Short spread = Sell leg1, Buy  leg2
         0  → Exit all positions in this pair
    """

    def __init__(self) -> None:
        cfg = get_config()

        if not cfg.alpaca_api_key or not cfg.alpaca_secret:
            raise EnvironmentError(
                "ALPACA_API_KEY_PAPER and ALPACA_SECRET must be set in .env"
            )

        # Trading client (orders, positions, account)
        self.trading = TradingClient(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret,
            paper=True,  # Always paper
        )

        # Market data client (live quotes for share calculation)
        self.data = StockHistoricalDataClient(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret,
        )

        self.cfg = cfg
        self._order_log: List[PairOrder] = []

        logger.info("AlpacaTrader initialised | paper=True")
        self._log_account()

    # ── Account info ───────────────────────────────────────────────────────

    def _log_account(self) -> None:
        """Log account summary on startup."""
        acct = self.trading.get_account()
        logger.info(
            "Alpaca account | equity=${:.2f} | cash=${:.2f} | buying_power=${:.2f}",
            float(acct.equity), float(acct.cash), float(acct.buying_power),
        )

    def get_portfolio(self) -> PortfolioSnapshot:
        """Return current portfolio state."""
        acct = self.trading.get_account()
        positions = self.trading.get_all_positions()
        open_orders = self.trading.get_orders()

        pos_dict = {
            p.symbol: float(p.market_value)
            for p in positions
        }

        return PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            cash=float(acct.cash),
            equity=float(acct.equity),
            positions=pos_dict,
            open_orders=len(open_orders),
        )

    # ── Live quotes ────────────────────────────────────────────────────────

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch latest ask price for a list of ETF symbols.

        Returns dict: { symbol: price }
        """
        req = StockLatestQuoteRequest(symbol_or_symbols=symbols)
        quotes = self.data.get_stock_latest_quote(req)

        prices = {}
        for sym, quote in quotes.items():
            # Use mid-price: (bid + ask) / 2
            mid = (float(quote.bid_price) + float(quote.ask_price)) / 2
            prices[sym] = mid
            logger.debug("{} mid-price: ${:.4f}", sym, mid)

        return prices

    # ── Share calculation ──────────────────────────────────────────────────

    def _calc_shares(
        self,
        notional: float,
        price: float,
        hedge_ratio: float = 1.0,
        is_leg2: bool = False,
    ) -> int:
        """
        Calculate integer shares for a leg.

        For leg2 we scale by hedge_ratio to ensure dollar neutrality.
        Returns integer shares (brokers require whole shares for stocks).
        """
        adjusted_notional = notional * hedge_ratio if is_leg2 else notional
        shares = int(adjusted_notional / price)  # floor to whole shares
        return max(shares, 1)  # always at least 1 share

    # ── Order submission ───────────────────────────────────────────────────

    def _submit_market_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        client_order_id: Optional[str] = None,
    ) -> str:
        """
        Submit a single market order. Returns order ID.

        Uses TimeInForce.DAY — expires if unfilled by market close.
        """
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )

        order = self.trading.submit_order(req)
        logger.info(
            "Order submitted | {} {} {} shares | id={}",
            side.value.upper(), qty, symbol, order.id,
        )
        return str(order.id)

    def submit_pair_order(
        self,
        pair_key: str,
        signal: int,
        capital_per_leg: float,
        hedge_ratio: float = 1.0,
    ) -> Optional[PairOrder]:
        """
        Submit a dollar-neutral pair trade.

        Parameters
        ----------
        pair_key : str
            e.g. "gold_silver" — must match config
        signal : int
            +1 = long spread (buy leg1, sell leg2)
            -1 = short spread (sell leg1, buy leg2)
             0 = no new position (use close_pair() to exit)
        capital_per_leg : float
            Dollar amount to deploy in each leg
        hedge_ratio : float
            Leg2 position = leg1_shares * hedge_ratio (from Kalman filter)

        Returns
        -------
        PairOrder or None
            None if signal == 0
        """
        if signal == 0:
            logger.info("Signal=0 for '{}' — no new position submitted", pair_key)
            return None

        pair = self.cfg.data.pairs.get(pair_key)
        if pair is None:
            raise KeyError(f"Unknown pair: {pair_key}")

        sym1 = pair["leg1_alpaca"]
        sym2 = pair["leg2_alpaca"]

        # ── Get live prices ────────────────────────────────────────────────
        prices = self.get_latest_prices([sym1, sym2])
        price1 = prices[sym1]
        price2 = prices[sym2]

        # ── Determine sides ────────────────────────────────────────────────
        if signal == 1:
            side1, side2 = OrderSide.BUY, OrderSide.SELL    # Long spread
        else:
            side1, side2 = OrderSide.SELL, OrderSide.BUY    # Short spread

        # ── Calculate shares (dollar-neutral) ──────────────────────────────
        shares1 = self._calc_shares(capital_per_leg, price1, is_leg2=False)
        shares2 = self._calc_shares(capital_per_leg, price2, hedge_ratio, is_leg2=True)

        # ── Build order record ─────────────────────────────────────────────
        pair_order = PairOrder(
            pair_key=pair_key,
            signal=signal,
            leg1_symbol=sym1,
            leg2_symbol=sym2,
            leg1_side=side1,
            leg2_side=side2,
            leg1_shares=shares1,
            leg2_shares=shares2,
            leg1_price=price1,
            leg2_price=price2,
            notional=capital_per_leg,
        )

        logger.info(
            "Submitting pair | {} | signal={} | {}: {} {} | {}: {} {} | hedge_ratio={:.4f}",
            pair_key, signal,
            sym1, side1.value, shares1,
            sym2, side2.value, shares2,
            hedge_ratio,
        )

        # ── Submit both legs ───────────────────────────────────────────────
        try:
            ts = pair_order.timestamp.strftime("%Y%m%d%H%M%S")
            pair_order.leg1_order_id = self._submit_market_order(
                sym1, side1, shares1,
                client_order_id=f"{pair_key}_leg1_{ts}",
            )
            pair_order.leg2_order_id = self._submit_market_order(
                sym2, side2, shares2,
                client_order_id=f"{pair_key}_leg2_{ts}",
            )
            pair_order.status = "submitted"
            logger.success("Pair order submitted successfully | {}", pair_key)

        except Exception as e:
            pair_order.status = "rejected"
            logger.error("Pair order FAILED for {} | error: {}", pair_key, str(e))
            raise

        self._order_log.append(pair_order)
        return pair_order

    # ── Position management ────────────────────────────────────────────────

    def close_pair(self, pair_key: str) -> None:
        """
        Close all open positions for a pair's ETF symbols.

        Used for: z-score mean reversion exit, stop-loss, regime filter.
        """
        pair = self.cfg.data.pairs[pair_key]
        symbols = [pair["leg1_alpaca"], pair["leg2_alpaca"]]

        for sym in symbols:
            try:
                self.trading.close_position(sym)
                logger.info("Closed position in {}", sym)
            except Exception as e:
                logger.warning("Could not close {} | {}", sym, str(e))

    def get_pair_positions(self, pair_key: str) -> Dict[str, float]:
        """
        Return current position (shares) for both legs of a pair.

        Returns dict: { symbol: qty }  (negative = short)
        """
        pair = self.cfg.data.pairs[pair_key]
        symbols = [pair["leg1_alpaca"], pair["leg2_alpaca"]]
        result = {}

        all_positions = {p.symbol: p for p in self.trading.get_all_positions()}

        for sym in symbols:
            if sym in all_positions:
                pos = all_positions[sym]
                qty = float(pos.qty)
                # Alpaca returns positive qty; side tells us direction
                if pos.side.value == "short":
                    qty = -qty
                result[sym] = qty
            else:
                result[sym] = 0.0

        return result

    # ── Order history ──────────────────────────────────────────────────────

    def get_order_log(self) -> list[PairOrder]:
        """Return all pair orders submitted this session."""
        return self._order_log.copy()

    def print_order_summary(self) -> None:
        """Pretty-print order log to console."""
        if not self._order_log:
            print("No orders submitted this session.")
            return

        print(f"\n{'='*70}")
        print(f"  Order Log — {len(self._order_log)} pair trades")
        print(f"{'='*70}")
        for o in self._order_log:
            print(
                f"  [{o.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{o.pair_key:<15} signal={o.signal:+d} "
                f"| {o.leg1_side.value:4} {o.leg1_shares:>5} {o.leg1_symbol} "
                f"| {o.leg2_side.value:4} {o.leg2_shares:>5} {o.leg2_symbol} "
                f"| status={o.status}"
            )
        print(f"{'='*70}\n")


# ── Quick smoke test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    trader = AlpacaTrader()

    # Show portfolio
    portfolio = trader.get_portfolio()
    print(f"\nPortfolio equity : ${portfolio.equity:,.2f}")
    print(f"Portfolio cash   : ${portfolio.cash:,.2f}")
    print(f"Open positions   : {portfolio.positions}")

    # Check prices
    prices = trader.get_latest_prices(["GLD", "SLV", "USO", "BNO"])
    print("\nETF Proxy Prices:")
    for sym, px in prices.items():
        print(f"  {sym}: ${px:.4f}")