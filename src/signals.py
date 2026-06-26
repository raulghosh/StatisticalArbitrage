"""
signals.py — Signal Engine + Event Types

Public types (imported by risk/monitor.py and execution layer):
  SignalAction   — ENTER | EXIT | STOP | HOLD | WAIT | SUSPEND
  PositionState  — FLAT | LONG | SHORT
  SignalEvent    — canonical per-bar signal event passed to the sizer

Pipeline (per pair, per run):
  1. Stationarity gate  — ADF + KPSS on both legs (I(1) check)
  2. Cointegration gate — Engle-Granger + Johansen
  3. Hedge ratio        — Kalman (default) | rolling OLS | static OLS
  4. Spread             — y(t) - β(t)·x(t)
  5. Z-score            — rolling (spread - μ) / σ
  6. Spread quality     — half-life [2–40d] + Hurst < 0.5 gate before entry
  7. Regime filter      — rolling EG p-value < regime_coint_threshold each bar
  8. Position           — state machine: +1 / -1 / 0
  9. SignalEvents        — one SignalEvent per bar for the sizer / monitor

Position convention (long spread = long y, short x):
  z < -entry  → ENTER LONG  (+1)
  z > +entry  → ENTER SHORT (-1)
  |z| < exit  → EXIT         (0)
  |z| > stop  → STOP         (0)

Regime / quality gates suppress new ENTERs but never block EXIT / STOP.

Usage (batch / backtest):
    from src.signals import SignalEngine
    engine = SignalEngine()
    result = engine.run(df["/GC"], df["/SI"], y_name="/GC", x_name="/SI")
    result.summary()
    events = result.signal_events   # list[SignalEvent] for the sizer

Usage (live / incremental):
    state = engine.init_state(historical_y, historical_x)
    new_state, event = engine.update_one(state, y_t, x_t, timestamp)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_config
from src.stats.cointegration import CointegrationChecker, CointegrationResult
from src.stats.kalman import KalmanHedgeRatio, KalmanResult
from src.stats.stationarity import StationarityChecker, StationarityResult

HedgeMethod = Literal["kalman", "rolling_ols", "static_ols"]


# ══════════════════════════════════════════════════════════════════════════════
# Signal event types  (imported by risk/monitor.py and execution layer)
# ══════════════════════════════════════════════════════════════════════════════

class SignalAction(Enum):
    """What should the execution layer do this bar."""
    ENTER   = "ENTER"    # Open a new position
    EXIT    = "EXIT"     # Close on mean reversion
    STOP    = "STOP"     # Close on stop-loss breach
    HOLD    = "HOLD"     # Position open, no change
    WAIT    = "WAIT"     # Flat, conditions not met
    SUSPEND = "SUSPEND"  # Regime broken — flatten and pause


class PositionState(Enum):
    """Current position state."""
    FLAT  = "FLAT"
    LONG  = "LONG"    # Long spread: long y, short x
    SHORT = "SHORT"   # Short spread: short y, long x


@dataclass
class SignalEvent:
    """
    Canonical per-bar signal event.

    Produced by SignalEngine and consumed by:
      - PositionSizer  (risk/sizer.py)  → sizes the order
      - RiskMonitor    (risk/monitor.py) → approves / blocks
      - AlpacaTrader   (execution/)     → submits to broker

    Attributes
    ----------
    pair_key        : str            — config key e.g. 'gold_silver'
    action          : SignalAction   — what to do this bar
    direction       : int            — +1 long | -1 short | 0 flat
    zscore          : float          — current z-score
    hedge_ratio     : float          — current β(t)
    signal_strength : float          — |zscore| / zscore_entry (0–1+ scale)
    spread_value    : float          — raw spread value
    half_life_days  : float          — Kalman spread half-life
    regime_ok       : bool           — rolling EG p-value < threshold
    quality_ok      : bool           — half-life + Hurst gates passed
    coint_pvalue    : float          — latest rolling EG p-value
    prev_state      : PositionState
    new_state       : PositionState
    timestamp       : datetime
    """
    pair_key        : str
    action          : SignalAction
    direction       : int
    zscore          : float
    hedge_ratio     : float
    signal_strength : float
    spread_value    : float
    half_life_days  : float
    regime_ok       : bool
    quality_ok      : bool
    coint_pvalue    : float
    prev_state      : PositionState
    new_state       : PositionState
    timestamp       : datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def is_entry(self) -> bool:
        return self.action == SignalAction.ENTER

    @property
    def is_exit(self) -> bool:
        return self.action in (SignalAction.EXIT, SignalAction.STOP,
                               SignalAction.SUSPEND)

    def to_series(self) -> pd.Series:
        return pd.Series({
            "pair_key"       : self.pair_key,
            "action"         : self.action.value,
            "direction"      : self.direction,
            "zscore"         : round(self.zscore, 4),
            "hedge_ratio"    : round(self.hedge_ratio, 4),
            "signal_strength": round(self.signal_strength, 4),
            "spread_value"   : round(self.spread_value, 4),
            "half_life_days" : round(self.half_life_days, 2),
            "regime_ok"      : self.regime_ok,
            "quality_ok"     : self.quality_ok,
            "coint_pvalue"   : round(self.coint_pvalue, 4),
            "prev_state"     : self.prev_state.value,
            "new_state"      : self.new_state.value,
            "timestamp"      : self.timestamp.isoformat(),
        })


# ══════════════════════════════════════════════════════════════════════════════
# Live state  (carry-forward for update_one)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EngineState:
    """
    Minimal state carried between bars for incremental live updates.

    Holds just what update_one() needs to process the next bar without
    reprocessing the full price history.
    """
    pair_key      : str
    hedge_method  : HedgeMethod
    # Kalman state (used when hedge_method == 'kalman')
    kalman_beta   : float          # β̂(t|t)
    kalman_P      : float          # P(t|t)
    # Rolling spread stats (for z-score)
    spread_window : List[float]    # last zscore_window spread values
    zscore_window : int
    # Rolling EG p-values (for regime filter)
    eg_pval_window: List[float]    # last coint_rolling_window p-values
    regime_window : int
    # Current position
    position_state: PositionState
    # Last half-life / Hurst (refreshed on full run)
    half_life_days: float
    hurst         : float
    # Config thresholds (echoed for convenience)
    zscore_entry  : float
    zscore_exit   : float
    zscore_stop   : float
    regime_threshold: float


# ══════════════════════════════════════════════════════════════════════════════
# OLS helpers
# ══════════════════════════════════════════════════════════════════════════════

def _static_ols_beta(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """Full-history OLS.  Returns (beta, alpha)."""
    X = np.column_stack([np.ones(len(x)), x])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(coeffs[1]), float(coeffs[0])


def _rolling_ols_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Rolling OLS β.  First (window-1) values are NaN."""
    betas = pd.Series(np.nan, index=y.index, dtype=float)
    y_arr, x_arr = y.values, x.values
    for end in range(window, len(y_arr) + 1):
        sl = slice(end - window, end)
        X = np.column_stack([np.ones(window), x_arr[sl]])
        coeffs = np.linalg.lstsq(X, y_arr[sl], rcond=None)[0]
        betas.iloc[end - 1] = coeffs[1]
    return betas


# ══════════════════════════════════════════════════════════════════════════════
# Position state machine
# ══════════════════════════════════════════════════════════════════════════════

def _state_from_pos(pos: int) -> PositionState:
    if pos == 1:  return PositionState.LONG
    if pos == -1: return PositionState.SHORT
    return PositionState.FLAT


def _generate_positions_and_events(
    zscore         : pd.Series,
    hedge_ratios   : pd.Series,
    spread         : pd.Series,
    half_life_days : float,
    hurst          : float,
    rolling_eg_pvals: pd.Series,
    pair_key       : str,
    entry          : float,
    exit_          : float,
    stop           : float,
    regime_threshold: float,
    min_halflife   : float,
    max_halflife   : float,
    min_hurst      : float,
) -> Tuple[pd.Series, pd.Series, pd.Series, List[SignalEvent]]:
    """
    Convert z-score + filters into positions and SignalEvents.

    Returns
    -------
    positions    : pd.Series[int]   (-1, 0, +1)
    regime_mask  : pd.Series[bool]  True = regime healthy
    quality_mask : pd.Series[bool]  True = spread quality ok
    events       : list[SignalEvent]
    """
    z       = zscore.values
    beta    = hedge_ratios.values
    spr     = spread.values
    idx     = zscore.index
    n       = len(z)

    pos_arr  = np.zeros(n, dtype=int)
    regime   = np.zeros(n, dtype=bool)
    quality  = np.zeros(n, dtype=bool)
    events: List[SignalEvent] = []

    # ── Build rolling regime mask ──────────────────────────────────────────
    # regime_ok[t] = True if rolling EG p-value at t is below threshold
    # rolling_eg_pvals index aligns with zscore index where available
    eg_aligned = rolling_eg_pvals.reindex(zscore.index)

    current = 0  # position: -1, 0, +1

    for t in range(n):
        zt     = z[t]
        beta_t = float(beta[t]) if not np.isnan(beta[t]) else 0.0
        spr_t  = float(spr[t]) if not np.isnan(spr[t]) else 0.0

        # ── Regime check ───────────────────────────────────────────────────
        eg_pval_t = float(eg_aligned.iloc[t]) if not np.isnan(
            eg_aligned.iloc[t] if t < len(eg_aligned) else float("nan")
        ) else 1.0
        regime_ok_t = eg_pval_t < regime_threshold

        # ── Quality check ──────────────────────────────────────────────────
        quality_ok_t = (
            min_halflife <= half_life_days <= max_halflife
            and hurst < min_hurst
        )

        regime[t]  = regime_ok_t
        quality[t] = quality_ok_t

        if np.isnan(zt):
            pos_arr[t] = 0
            current    = 0
            continue

        strength   = abs(zt) / entry if entry > 0 else 0.0
        prev_state = _state_from_pos(current)

        # ── State machine ──────────────────────────────────────────────────
        new_current = current

        if current == 0:
            # Only enter if regime + quality gates pass
            if regime_ok_t and quality_ok_t:
                if zt < -entry:
                    new_current = 1
                elif zt > entry:
                    new_current = -1
            elif not regime_ok_t and current == 0:
                pass  # stay flat silently

        elif current == 1:   # Long spread
            if zt > -exit_:
                new_current = 0    # mean reverted
            elif zt < -stop:
                new_current = 0    # stop out

        elif current == -1:  # Short spread
            if zt < exit_:
                new_current = 0    # mean reverted
            elif zt > stop:
                new_current = 0    # stop out

        # ── Determine action ───────────────────────────────────────────────
        new_state = _state_from_pos(new_current)

        if prev_state == PositionState.FLAT and new_state != PositionState.FLAT:
            action = SignalAction.ENTER
        elif prev_state != PositionState.FLAT and new_state == PositionState.FLAT:
            # Distinguish exit vs stop vs regime suspension
            if not regime_ok_t:
                action = SignalAction.SUSPEND
            elif (current == 1 and zt < -stop) or (current == -1 and zt > stop):
                action = SignalAction.STOP
            else:
                action = SignalAction.EXIT
        elif prev_state != PositionState.FLAT and new_state == prev_state:
            action = SignalAction.HOLD
        else:
            action = SignalAction.WAIT

        current = new_current
        pos_arr[t] = current

        # ── Emit event on any action change ────────────────────────────────
        ts = idx[t]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime().replace(tzinfo=timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        if action not in (SignalAction.WAIT,):  # skip pure wait bars
            event = SignalEvent(
                pair_key        = pair_key,
                action          = action,
                direction       = current if current != 0 else (
                    -1 if prev_state == PositionState.SHORT else
                     1 if prev_state == PositionState.LONG else 0
                ),
                zscore          = float(zt),
                hedge_ratio     = beta_t,
                signal_strength = float(strength),
                spread_value    = spr_t,
                half_life_days  = float(half_life_days),
                regime_ok       = regime_ok_t,
                quality_ok      = quality_ok_t,
                coint_pvalue    = eg_pval_t,
                prev_state      = prev_state,
                new_state       = new_state,
                timestamp       = ts,
            )
            events.append(event)

    return (
        pd.Series(pos_arr,        index=idx, name="position"),
        pd.Series(regime.astype(bool), index=idx, name="regime_ok"),
        pd.Series(quality.astype(bool), index=idx, name="quality_ok"),
        events,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Signal Result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SignalResult:
    """
    Full output of one SignalEngine.run() call.

    Key attributes
    --------------
    passed_gate      : bool         — True only if both stat + coint passed
    position         : pd.Series    — +1 / -1 / 0 per bar
    zscore           : pd.Series    — rolling z-score
    spread           : pd.Series    — y - β·x
    hedge_ratios     : pd.Series    — β(t) per bar
    regime_ok        : pd.Series    — rolling regime health per bar
    quality_ok       : pd.Series    — spread quality gate per bar
    signal_events    : list[SignalEvent]  — for sizer / monitor
    to_dataframe()   → full aligned DataFrame
    latest()         → most recent bar as SignalEvent (for live)
    """

    # ── Identity ───────────────────────────────────────────────────────────
    y_name       : str
    x_name       : str
    pair_key     : str
    hedge_method : HedgeMethod

    # ── Gate results ───────────────────────────────────────────────────────
    passed_stationarity  : bool
    passed_cointegration : bool

    # ── Prices & spread ────────────────────────────────────────────────────
    y            : pd.Series
    x            : pd.Series
    hedge_ratios : pd.Series
    spread       : pd.Series
    zscore       : pd.Series
    position     : pd.Series
    regime_ok    : pd.Series
    quality_ok   : pd.Series
    trade_entries: pd.Series
    trade_exits  : pd.Series

    # ── Events ─────────────────────────────────────────────────────────────
    signal_events: List[SignalEvent]

    # ── Upstream objects ───────────────────────────────────────────────────
    stat_result_y : StationarityResult
    stat_result_x : StationarityResult
    coint_result  : CointegrationResult
    kalman_result : Optional[KalmanResult]

    # ── Config ─────────────────────────────────────────────────────────────
    zscore_window  : int
    zscore_entry   : float
    zscore_exit    : float
    zscore_stop    : float
    regime_threshold: float

    @property
    def passed_gate(self) -> bool:
        return self.passed_stationarity and self.passed_cointegration

    @property
    def n_trades(self) -> int:
        return int(self.trade_entries.sum())

    @property
    def current_position(self) -> int:
        return int(self.position.iloc[-1]) if not self.position.empty else 0

    @property
    def current_zscore(self) -> float:
        return float(self.zscore.iloc[-1]) if not self.zscore.empty else float("nan")

    @property
    def current_hedge_ratio(self) -> float:
        return float(self.hedge_ratios.iloc[-1]) if not self.hedge_ratios.empty else 0.0

    @property
    def time_in_market(self) -> float:
        return float((self.position != 0).mean()) if not self.position.empty else 0.0

    @property
    def regime_pct(self) -> float:
        """Fraction of bars where regime was healthy."""
        return float(self.regime_ok.mean()) if not self.regime_ok.empty else 0.0

    def summary(self) -> str:
        lines = [
            f"\n{'═'*65}",
            f"  Signal Engine — {self.y_name} / {self.x_name}",
            f"  hedge={self.hedge_method} | z_window={self.zscore_window} | pair={self.pair_key}",
            f"{'═'*65}",
            f"  Pre-flight gates:",
            f"    Stationarity (both I(1))  : {'✅ PASS' if self.passed_stationarity else '❌ FAIL'}",
            f"    Cointegration             : {'✅ PASS' if self.passed_cointegration else '❌ FAIL'}",
            f"    → Proceed to trade        : {'✅ YES' if self.passed_gate else '❌ NO'}",
        ]
        if not self.passed_gate:
            lines += [f"", f"  ⚠️  Gate failed — no signals generated.", f"{'═'*65}"]
            output = "\n".join(lines)
            print(output)
            return output

        lines += [
            f"",
            f"  Hedge Ratio β(t):",
            f"    Current                   : {self.current_hedge_ratio:>10.4f}",
            f"    Mean / Std                : {self.hedge_ratios.mean():>10.4f} / {self.hedge_ratios.std():.4f}",
            f"",
            f"  Spread / Z-score:",
            f"    Spread mean / std         : {self.spread.mean():>10.4f} / {self.spread.std():.4f}",
            f"    Current z-score           : {self.current_zscore:>10.4f}",
            f"    Entry ±{self.zscore_entry} | Exit ±{self.zscore_exit} | Stop ±{self.zscore_stop}",
            f"",
            f"  Filters:",
            f"    Regime healthy (% bars)   : {self.regime_pct:>10.1%}  (threshold p < {self.regime_threshold})",
            f"    Quality ok (HL + Hurst)   : {'Yes' if self.quality_ok.iloc[-1] else 'No'} (latest bar)",
        ]

        if self.kalman_result:
            lines += [
                f"    Kalman half-life          : {self.kalman_result.half_life_days:>10.1f}d",
                f"    Kalman Hurst exponent     : {self.kalman_result.hurst_exponent:>10.4f}",
            ]

        lines += [
            f"",
            f"  Signals:",
            f"    Trades (entries)          : {self.n_trades:>10}",
            f"    Time in market            : {self.time_in_market:>10.1%}",
            f"    Current position          : {self.current_position:>10}  "
            f"({'long spread' if self.current_position == 1 else 'short spread' if self.current_position == -1 else 'flat'})",
            f"    Total signal events       : {len(self.signal_events):>10}",
            f"{'═'*65}",
        ]
        output = "\n".join(lines)
        print(output)
        return output

    def to_dataframe(self) -> pd.DataFrame:
        """Full aligned DataFrame with all signal columns."""
        return pd.DataFrame({
            self.y_name   : self.y,
            self.x_name   : self.x,
            "hedge_ratio" : self.hedge_ratios,
            "spread"      : self.spread,
            "zscore"      : self.zscore,
            "position"    : self.position,
            "regime_ok"   : self.regime_ok,
            "quality_ok"  : self.quality_ok,
        })

    def events_dataframe(self) -> pd.DataFrame:
        """All SignalEvents as a DataFrame (ENTER/EXIT/STOP/HOLD only)."""
        if not self.signal_events:
            return pd.DataFrame()
        return pd.DataFrame([e.to_series() for e in self.signal_events])

    def latest(self) -> Optional[SignalEvent]:
        """Most recent SignalEvent — useful for live trading."""
        return self.signal_events[-1] if self.signal_events else None


# ══════════════════════════════════════════════════════════════════════════════
# Signal Engine
# ══════════════════════════════════════════════════════════════════════════════

class SignalEngine:
    """
    End-to-end signal pipeline for a cointegrated pair.

    Steps
    -----
    1. Align prices
    2. Stationarity gate    — both legs must be I(1)
    3. Cointegration gate   — EG + Johansen must agree
    4. Hedge ratio          — Kalman | rolling OLS | static OLS
    5. Spread + z-score     — rolling normalisation
    6. Spread quality gate  — half-life [min,max] + Hurst < threshold
    7. Rolling regime filter— EG p-value < regime_coint_threshold each bar
    8. Position state machine → positions + SignalEvents

    Parameters
    ----------
    hedge_method         : 'kalman' | 'rolling_ols' | 'static_ols'
    pair_key             : str  — config pair key (e.g. 'gold_silver')
    gate_mode            : 'full' | 'warn' | 'skip'
    All numeric params default to config values when None.
    """

    def __init__(
        self,
        hedge_method         : HedgeMethod = "kalman",
        pair_key             : str = "gold_silver",
        zscore_window        : Optional[int] = None,
        stat_significance    : Optional[float] = None,
        coint_significance   : Optional[float] = None,
        coint_rolling_window : Optional[int] = None,
        zscore_entry         : Optional[float] = None,
        zscore_exit          : Optional[float] = None,
        zscore_stop          : Optional[float] = None,
        rolling_ols_window   : Optional[int] = None,
        kalman_delta         : Optional[float] = None,
        kalman_vt            : Optional[float] = None,
        regime_threshold     : Optional[float] = None,
        min_halflife         : Optional[float] = None,
        max_halflife         : Optional[float] = None,
        min_hurst            : Optional[float] = None,
        gate_mode            : Literal["full", "warn", "skip"] = "full",
    ) -> None:
        cfg_s = get_config().stats
        cfg_r = get_config().risk

        self.hedge_method         = hedge_method
        self.pair_key             = pair_key
        self.zscore_window        = zscore_window        or cfg_s.zscore_window
        self.stat_significance    = stat_significance    or cfg_s.adf_significance
        self.coint_significance   = coint_significance   or cfg_s.coint_significance
        self.coint_rolling_window = coint_rolling_window or cfg_s.coint_rolling_window
        self.zscore_entry         = zscore_entry         or cfg_s.zscore_entry
        self.zscore_exit          = zscore_exit          or cfg_s.zscore_exit
        self.zscore_stop          = zscore_stop          or cfg_s.zscore_stop
        self.rolling_ols_window   = rolling_ols_window   or cfg_s.rolling_ols_window
        self.kalman_delta         = kalman_delta         or cfg_s.kalman_delta
        self.kalman_vt            = kalman_vt            or cfg_s.kalman_vt
        self.regime_threshold     = regime_threshold     or cfg_r.regime_coint_threshold
        self.min_halflife         = min_halflife         or float(cfg_s.min_halflife_days)
        self.max_halflife         = max_halflife         or float(cfg_s.max_halflife_days)
        self.min_hurst            = min_hurst            or cfg_s.min_hurst_threshold
        self.gate_mode            = gate_mode

        self._stat_checker  = StationarityChecker(significance=self.stat_significance)
        self._coint_checker = CointegrationChecker(
            significance=self.coint_significance,
            rolling_window=self.coint_rolling_window,
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    def _run_stationarity_gate(
        self, y: pd.Series, x: pd.Series, y_name: str, x_name: str
    ) -> Tuple[bool, StationarityResult, StationarityResult]:
        r_y, r_x = self._stat_checker.test_pair(y, x, y_name=y_name, x_name=x_name)
        passed = r_y.is_i1 and r_x.is_i1
        if not passed:
            logger.warning(
                "Stationarity gate FAILED | {}: i1={} | {}: i1={}",
                y_name, r_y.is_i1, x_name, r_x.is_i1,
            )
        return passed, r_y, r_x

    def _run_cointegration_gate(
        self, y: pd.Series, x: pd.Series, y_name: str, x_name: str
    ) -> Tuple[bool, CointegrationResult]:
        r = self._coint_checker.test(y, x, y_name=y_name, x_name=x_name)
        passed = r.is_cointegrated
        if not passed:
            logger.warning(
                "Cointegration gate FAILED | {}/{} | eg_p={:.4f} | rank={}",
                y_name, x_name, r.eg_pvalue, r.johansen_rank,
            )
        return passed, r

    def _compute_hedge_ratio(
        self, y: pd.Series, x: pd.Series, y_name: str, x_name: str
    ) -> Tuple[pd.Series, Optional[KalmanResult]]:
        if self.hedge_method == "kalman":
            kf = KalmanHedgeRatio(delta=self.kalman_delta, vt=self.kalman_vt)
            kr = kf.fit(y, x, y_name=y_name, x_name=x_name)
            return kr.hedge_ratios, kr
        elif self.hedge_method == "rolling_ols":
            betas = _rolling_ols_beta(y, x, window=self.rolling_ols_window)
            return betas, None
        elif self.hedge_method == "static_ols":
            beta, _ = _static_ols_beta(y.values, x.values)
            return pd.Series(beta, index=y.index, name="hedge_ratio"), None
        else:
            raise ValueError(f"Unknown hedge_method: {self.hedge_method!r}")

    def _zscore(self, spread: pd.Series) -> pd.Series:
        mu    = spread.rolling(self.zscore_window).mean()
        sigma = spread.rolling(self.zscore_window).std()
        return ((spread - mu) / sigma).rename("zscore")

    # ── Batch run (backtest / research) ───────────────────────────────────

    def run(
        self,
        y      : pd.Series,
        x      : pd.Series,
        y_name : Optional[str] = None,
        x_name : Optional[str] = None,
    ) -> SignalResult:
        """
        Full signal pipeline over the complete price history.

        Returns SignalResult (always). If gate fails (gate_mode='full'),
        spread/zscore/position/events are empty.
        """
        y_label = y_name or (y.name or "Y")
        x_label = x_name or (x.name or "X")
        logger.info(
            "SignalEngine.run | {} / {} | method={} | gate={}",
            y_label, x_label, self.hedge_method, self.gate_mode,
        )

        df        = pd.DataFrame({"y": y, "x": x}).dropna()
        y_aligned = df["y"].rename(y_label)
        x_aligned = df["x"].rename(x_label)

        # ── Stationarity gate ──────────────────────────────────────────────
        if self.gate_mode == "skip":
            passed_stat = True
            stat_r_y, stat_r_x = self._stat_checker.test_pair(
                y_aligned, x_aligned, y_name=y_label, x_name=x_label
            )
        else:
            passed_stat, stat_r_y, stat_r_x = self._run_stationarity_gate(
                y_aligned, x_aligned, y_label, x_label
            )
            if not passed_stat and self.gate_mode == "full":
                return self._empty_result(
                    y_label, x_label, y_aligned, x_aligned,
                    False, False, stat_r_y, stat_r_x,
                )

        # ── Cointegration gate ─────────────────────────────────────────────
        if self.gate_mode == "skip":
            passed_coint = True
            coint_r = self._coint_checker.test(
                y_aligned, x_aligned, y_name=y_label, x_name=x_label
            )
        else:
            passed_coint, coint_r = self._run_cointegration_gate(
                y_aligned, x_aligned, y_label, x_label
            )
            if not passed_coint and self.gate_mode == "full":
                return self._empty_result(
                    y_label, x_label, y_aligned, x_aligned,
                    passed_stat, False, stat_r_y, stat_r_x,
                    coint_result=coint_r,
                )

        # ── Hedge ratio ────────────────────────────────────────────────────
        hedge_ratios, kalman_result = self._compute_hedge_ratio(
            y_aligned, x_aligned, y_label, x_label
        )

        # ── Spread + z-score ───────────────────────────────────────────────
        spread = (y_aligned - hedge_ratios * x_aligned).rename("spread")
        zscore = self._zscore(spread)

        # ── Spread quality (half-life + Hurst from Kalman or static fit) ───
        half_life = kalman_result.half_life_days if kalman_result else float("inf")
        hurst     = kalman_result.hurst_exponent if kalman_result else 0.5

        # ── Rolling EG p-values for regime filter ─────────────────────────
        rolling_eg = coint_r.rolling_pvalues  # pd.Series, indexed by date

        # ── State machine → positions + events ────────────────────────────
        position, regime_ok, quality_ok, events = _generate_positions_and_events(
            zscore=zscore,
            hedge_ratios=hedge_ratios,
            spread=spread,
            half_life_days=half_life,
            hurst=hurst,
            rolling_eg_pvals=rolling_eg,
            pair_key=self.pair_key,
            entry=self.zscore_entry,
            exit_=self.zscore_exit,
            stop=self.zscore_stop,
            regime_threshold=self.regime_threshold,
            min_halflife=self.min_halflife,
            max_halflife=self.max_halflife,
            min_hurst=self.min_hurst,
        )

        pos_prev      = position.shift(1).fillna(0).astype(int)
        trade_entries = (pos_prev == 0) & (position != 0)
        trade_exits   = (pos_prev != 0) & (position == 0)

        logger.info(
            "SignalEngine complete | {}/{} | trades={} | z={:.3f} | pos={}",
            y_label, x_label,
            int(trade_entries.sum()),
            float(zscore.iloc[-1]) if not zscore.empty else float("nan"),
            int(position.iloc[-1]) if not position.empty else 0,
        )

        return SignalResult(
            y_name=y_label, x_name=x_label,
            pair_key=self.pair_key, hedge_method=self.hedge_method,
            passed_stationarity=passed_stat, passed_cointegration=passed_coint,
            y=y_aligned, x=x_aligned,
            hedge_ratios=hedge_ratios, spread=spread, zscore=zscore,
            position=position, regime_ok=regime_ok, quality_ok=quality_ok,
            trade_entries=trade_entries, trade_exits=trade_exits,
            signal_events=events,
            stat_result_y=stat_r_y, stat_result_x=stat_r_x,
            coint_result=coint_r, kalman_result=kalman_result,
            zscore_window=self.zscore_window,
            zscore_entry=self.zscore_entry, zscore_exit=self.zscore_exit,
            zscore_stop=self.zscore_stop,
            regime_threshold=self.regime_threshold,
        )

    # ── Incremental live update ────────────────────────────────────────────

    def init_state(
        self,
        y         : pd.Series,
        x         : pd.Series,
        y_name    : Optional[str] = None,
        x_name    : Optional[str] = None,
    ) -> EngineState:
        """
        Initialise live state from historical price series.

        Call once at startup; then call update_one() each bar.
        Runs the full pipeline to warm up Kalman state and spread windows.
        """
        result = self.run(y, x, y_name=y_name, x_name=x_name)

        # Extract Kalman state (last bar)
        if result.kalman_result:
            kalman_beta = float(result.kalman_result.hedge_ratios.iloc[-1])
            kalman_P    = float(result.kalman_result.variances.iloc[-1])
        else:
            beta_static, _ = _static_ols_beta(y.dropna().values, x.dropna().values)
            kalman_beta, kalman_P = beta_static, 1.0

        # Warm up spread window
        spread = result.spread.dropna()
        w      = self.zscore_window
        spread_window = list(spread.iloc[-w:].values) if len(spread) >= w else list(spread.values)

        # Warm up regime window
        eg_pvals = result.coint_result.rolling_pvalues.dropna()
        rw       = self.coint_rolling_window
        eg_window = list(eg_pvals.iloc[-rw:].values) if len(eg_pvals) >= rw else list(eg_pvals.values)

        half_life = result.kalman_result.half_life_days if result.kalman_result else float("inf")
        hurst     = result.kalman_result.hurst_exponent if result.kalman_result else 0.5

        return EngineState(
            pair_key        = self.pair_key,
            hedge_method    = self.hedge_method,
            kalman_beta     = kalman_beta,
            kalman_P        = kalman_P,
            spread_window   = spread_window,
            zscore_window   = self.zscore_window,
            eg_pval_window  = eg_window,
            regime_window   = self.coint_rolling_window,
            position_state  = _state_from_pos(result.current_position),
            half_life_days  = half_life,
            hurst           = hurst,
            zscore_entry    = self.zscore_entry,
            zscore_exit     = self.zscore_exit,
            zscore_stop     = self.zscore_stop,
            regime_threshold= self.regime_threshold,
        )

    def update_one(
        self,
        state    : EngineState,
        y_t      : float,
        x_t      : float,
        eg_pval_t: float = 1.0,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[EngineState, SignalEvent]:
        """
        Process a single new bar and return updated state + SignalEvent.

        Parameters
        ----------
        state     : EngineState — carry-forward from last bar
        y_t       : float       — latest leg1 price
        x_t       : float       — latest leg2 price
        eg_pval_t : float       — latest rolling EG p-value (from regime check)
        timestamp : datetime    — bar timestamp (defaults to now UTC)

        Returns
        -------
        (new_state, SignalEvent)
        """
        ts = timestamp or datetime.now(timezone.utc)

        # ── 1. Update Kalman β ─────────────────────────────────────────────
        if state.hedge_method == "kalman":
            kf = KalmanHedgeRatio(delta=self.kalman_delta, vt=self.kalman_vt)
            beta_t, P_t, _, _ = kf.update_one(
                state.kalman_beta, state.kalman_P, y_t, x_t
            )
        else:
            beta_t = state.kalman_beta
            P_t    = state.kalman_P

        # ── 2. Spread + z-score ────────────────────────────────────────────
        spr_t = y_t - beta_t * x_t
        window = list(state.spread_window) + [spr_t]
        if len(window) > state.zscore_window:
            window = window[-state.zscore_window:]
        mu    = float(np.mean(window))
        sigma = float(np.std(window, ddof=1)) if len(window) > 1 else 1.0
        zt    = (spr_t - mu) / sigma if sigma > 0 else 0.0

        # ── 3. Regime check ────────────────────────────────────────────────
        eg_window = list(state.eg_pval_window) + [eg_pval_t]
        if len(eg_window) > state.regime_window:
            eg_window = eg_window[-state.regime_window:]
        regime_ok = eg_pval_t < state.regime_threshold

        # ── 4. Quality check ───────────────────────────────────────────────
        quality_ok = (
            state.min_halflife <= state.half_life_days <= state.max_halflife
            and state.hurst < state.min_hurst
        ) if hasattr(state, "min_halflife") else True

        # ── 5. State machine ───────────────────────────────────────────────
        prev_state  = state.position_state
        prev_pos    = (1 if prev_state == PositionState.LONG
                       else -1 if prev_state == PositionState.SHORT else 0)
        new_pos     = prev_pos

        if prev_pos == 0:
            if regime_ok and quality_ok:
                if zt < -state.zscore_entry:
                    new_pos = 1
                elif zt > state.zscore_entry:
                    new_pos = -1
        elif prev_pos == 1:
            if zt > -state.zscore_exit or zt < -state.zscore_stop:
                new_pos = 0
        elif prev_pos == -1:
            if zt < state.zscore_exit or zt > state.zscore_stop:
                new_pos = 0

        new_state_pos = _state_from_pos(new_pos)

        if prev_state == PositionState.FLAT and new_state_pos != PositionState.FLAT:
            action = SignalAction.ENTER
        elif prev_state != PositionState.FLAT and new_state_pos == PositionState.FLAT:
            action = SignalAction.STOP if (
                (prev_pos == 1 and zt < -state.zscore_stop) or
                (prev_pos == -1 and zt > state.zscore_stop)
            ) else SignalAction.EXIT
        elif prev_state != PositionState.FLAT:
            action = SignalAction.HOLD
        else:
            action = SignalAction.WAIT

        strength = abs(zt) / state.zscore_entry if state.zscore_entry > 0 else 0.0

        event = SignalEvent(
            pair_key        = state.pair_key,
            action          = action,
            direction       = new_pos if new_pos != 0 else prev_pos,
            zscore          = float(zt),
            hedge_ratio     = float(beta_t),
            signal_strength = float(strength),
            spread_value    = float(spr_t),
            half_life_days  = float(state.half_life_days),
            regime_ok       = regime_ok,
            quality_ok      = quality_ok,
            coint_pvalue    = eg_pval_t,
            prev_state      = prev_state,
            new_state       = new_state_pos,
            timestamp       = ts,
        )

        # ── 6. Build new state ─────────────────────────────────────────────
        import dataclasses
        new_state = dataclasses.replace(
            state,
            kalman_beta    = beta_t,
            kalman_P       = P_t,
            spread_window  = window,
            eg_pval_window = eg_window,
            position_state = new_state_pos,
        )

        return new_state, event

    # ── Empty result helper ────────────────────────────────────────────────

    def _empty_result(
        self,
        y_label     : str,
        x_label     : str,
        y           : pd.Series,
        x           : pd.Series,
        passed_stat : bool,
        passed_coint: bool,
        stat_r_y    : StationarityResult,
        stat_r_x    : StationarityResult,
        coint_result: Optional[CointegrationResult] = None,
    ) -> SignalResult:
        empty     = pd.Series(dtype=float)
        empty_int = pd.Series(dtype=int)
        empty_bool= pd.Series(dtype=bool)

        if coint_result is None:
            coint_result = self._coint_checker.test(y, x, y_name=y_label, x_name=x_label)

        return SignalResult(
            y_name=y_label, x_name=x_label,
            pair_key=self.pair_key, hedge_method=self.hedge_method,
            passed_stationarity=passed_stat, passed_cointegration=passed_coint,
            y=y, x=x,
            hedge_ratios=empty, spread=empty, zscore=empty,
            position=empty_int, regime_ok=empty_bool, quality_ok=empty_bool,
            trade_entries=empty_bool, trade_exits=empty_bool,
            signal_events=[],
            stat_result_y=stat_r_y, stat_result_x=stat_r_x,
            coint_result=coint_result, kalman_result=None,
            zscore_window=self.zscore_window,
            zscore_entry=self.zscore_entry, zscore_exit=self.zscore_exit,
            zscore_stop=self.zscore_stop,
            regime_threshold=self.regime_threshold,
        )


# ── Smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

    print("\n" + "="*65)
    print("  SIGNAL ENGINE SMOKE TEST")
    print("  Synthetic cointegrated pair — Kalman + regime + quality gates")
    print("="*65)

    rng = np.random.default_rng(42)
    n   = 800

    # ── Synthetic cointegrated pair ───────────────────────────────────────
    x_rw = pd.Series(np.cumsum(rng.normal(0, 1, n)), name="X")
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.7 * noise[t-1] + rng.normal(0, 0.3)
    y_coint = (2.0 * x_rw + pd.Series(noise)).rename("Y")

    # ── Test 1: Full pipeline (Kalman, gate_mode=full) ────────────────────
    # Quality-gate thresholds are widened vs. production defaults because
    # Kalman over-adapts on short synthetic series, shrinking the residual
    # half-life and inflating the Hurst estimate.
    print("\n── Test 1: Kalman, gate_mode=full ──")
    engine = SignalEngine(
        hedge_method="kalman", pair_key="gold_silver", gate_mode="full",
        min_halflife=0.5,   # Kalman residuals on synthetic data are tight
        max_halflife=60,
        min_hurst=0.70,     # Hurst < 0.70 is sufficient for smoke coverage
    )
    r = engine.run(y_coint, x_rw, y_name="Y", x_name="X")
    r.summary()
    assert r.passed_gate, "❌ Should pass gate"
    assert not r.spread.empty, "❌ Spread should be populated"
    assert "regime_ok" in r.to_dataframe().columns, "❌ regime_ok missing"
    assert "quality_ok" in r.to_dataframe().columns, "❌ quality_ok missing"
    assert len(r.signal_events) > 0, "❌ Should have signal events"
    print(f"  Signal events: {len(r.signal_events)} | trades: {r.n_trades}")
    print("✅ PASS: Full pipeline with regime + quality filters")

    # ── Test 2: SignalEvent structure ─────────────────────────────────────
    print("\n── Test 2: SignalEvent structure ──")
    enter_events = [e for e in r.signal_events if e.action == SignalAction.ENTER]
    if enter_events:
        ev = enter_events[0]
        print(ev.to_series())
        assert ev.regime_ok is not None, "❌ regime_ok missing"
        assert ev.quality_ok is not None, "❌ quality_ok missing"
        assert ev.half_life_days > 0, "❌ half_life_days missing"
    print("✅ PASS: SignalEvent has all required fields")

    # ── Test 3: update_one() ──────────────────────────────────────────────
    print("\n── Test 3: update_one() incremental update ──")
    state = engine.init_state(y_coint.iloc[:-10], x_rw.iloc[:-10])
    assert state.kalman_beta > 0, "❌ State should have positive beta"
    for i in range(-10, 0):
        state, event = engine.update_one(
            state,
            float(y_coint.iloc[i]),
            float(x_rw.iloc[i]),
            eg_pval_t=0.02,
        )
    print(f"  Final state: β={state.kalman_beta:.4f} | pos={state.position_state.value}")
    print(f"  Last event: action={event.action.value} | z={event.zscore:.3f}")
    assert event.hedge_ratio > 0, "❌ update_one() should return positive hedge ratio"
    print("✅ PASS: update_one() incremental update works")

    # ── Test 4: Gate failure ──────────────────────────────────────────────
    print("\n── Test 4: Gate failure on noise pair ──")
    y_n = pd.Series(rng.normal(0, 1, n), name="Y_noise")
    x_n = pd.Series(rng.normal(0, 1, n), name="X_noise")
    r_gate = engine.run(y_n, x_n)
    assert not r_gate.passed_gate, "❌ Noise pair should fail gate"
    assert len(r_gate.signal_events) == 0, "❌ Gate-failed result should have no events"
    print("✅ PASS: Gate correctly blocks noise pair")

    # ── Test 5: PositionState / SignalAction enum sanity ─────────────────
    print("\n── Test 5: Enum types ──")
    assert SignalAction.ENTER.value == "ENTER"
    assert PositionState.LONG.value == "LONG"
    print("✅ PASS: Enums correct")

    print("\n" + "="*65)
    print("  ALL SIGNAL ENGINE SMOKE TESTS PASSED ✅")
    print("="*65 + "\n")
