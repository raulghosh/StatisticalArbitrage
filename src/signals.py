"""
signals.py — Spread Signal Engine

Pipeline (per pair, per run):
  1. Stationarity gate  — ADF + KPSS on both legs (I(1) check)
  2. Cointegration gate — Engle-Granger + Johansen
  3. Hedge ratio        — Kalman (default) | rolling OLS | static OLS
  4. Spread             — y(t) - β(t)·x(t)
  5. Z-score            — rolling (spread - μ) / σ
  6. Position           — discrete: +1 / -1 / 0
                           entry  when |z| crosses zscore_entry
                           exit   when |z| crosses zscore_exit  (mean reversion)
                           stop   when |z| crosses zscore_stop  (hard stop)

Position convention (long spread = long y, short x):
  z < -entry → position = +1  (spread too low  → buy y, sell x)
  z > +entry → position = -1  (spread too high → sell y, buy x)
  |z| < exit → position =  0  (mean reverted → flat)
  |z| > stop → position =  0  (blown through → stop out)

Usage:
    from src.signals import SignalEngine
    engine = SignalEngine()
    result = engine.run(pair_df["/GC"], pair_df["/SI"],
                        y_name="/GC", x_name="/SI")
    result.summary()
    df = result.to_dataframe()   # prices, spread, zscore, position
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_config
from src.stats.cointegration import CointegrationChecker, CointegrationResult
from src.stats.kalman import KalmanHedgeRatio, KalmanResult
from src.stats.stationarity import StationarityChecker, StationarityResult

HedgeMethod = Literal["kalman", "rolling_ols", "static_ols"]


# ══════════════════════════════════════════════════════════════════════════════
# OLS helpers
# ══════════════════════════════════════════════════════════════════════════════

def _static_ols_beta(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """Full-history OLS.  Returns (beta, alpha)."""
    X = np.column_stack([np.ones(len(x)), x])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(coeffs[1]), float(coeffs[0])


def _rolling_ols_beta(
    y: pd.Series, x: pd.Series, window: int
) -> pd.Series:
    """
    Rolling OLS β over a sliding window.

    Returns a pd.Series of hedge ratios aligned with y/x index.
    First `window-1` values are NaN (not enough history).
    """
    betas = pd.Series(index=y.index, dtype=float)
    y_arr = y.values
    x_arr = x.values
    n = len(y_arr)

    for end in range(window, n + 1):
        start = end - window
        y_sub = y_arr[start:end]
        x_sub = x_arr[start:end]
        X = np.column_stack([np.ones(window), x_sub])
        coeffs = np.linalg.lstsq(X, y_sub, rcond=None)[0]
        betas.iloc[end - 1] = coeffs[1]

    return betas


# ══════════════════════════════════════════════════════════════════════════════
# Position state machine
# ══════════════════════════════════════════════════════════════════════════════

def _generate_positions(
    zscore: pd.Series,
    entry: float,
    exit_: float,
    stop: float,
) -> pd.Series:
    """
    Convert a z-score series into a discrete position series.

    State machine with hysteresis:
      - Flat (0) → enter long  (+1) when z < -entry
      - Flat (0) → enter short (-1) when z > +entry
      - Long (+1) → exit (0) when z > -exit_  (mean reverted)
      - Long (+1) → stop (0) when z < -stop   (blown through lower)
      - Short (-1) → exit (0) when z < +exit_ (mean reverted)
      - Short (-1) → stop (0) when z > +stop  (blown through upper)

    Parameters
    ----------
    zscore : pd.Series
    entry  : float  — |z| threshold to enter (e.g. 2.0)
    exit_  : float  — |z| threshold to exit  (e.g. 0.5)
    stop   : float  — |z| threshold to stop  (e.g. 3.5)

    Returns
    -------
    pd.Series of int  (-1, 0, +1), same index as zscore.
    """
    z = zscore.values
    n = len(z)
    pos = np.zeros(n, dtype=int)
    current = 0  # current position: -1, 0, +1

    for t in range(n):
        zt = z[t]
        if np.isnan(zt):
            pos[t] = 0
            current = 0
            continue

        if current == 0:
            # Look for entry
            if zt < -entry:
                current = 1    # spread too low → long
            elif zt > entry:
                current = -1   # spread too high → short

        elif current == 1:
            # Long spread: exit when z reverts above -exit, stop if z < -stop
            if zt > -exit_:
                current = 0
            elif zt < -stop:
                current = 0    # blown through stop

        elif current == -1:
            # Short spread: exit when z reverts below +exit, stop if z > +stop
            if zt < exit_:
                current = 0
            elif zt > stop:
                current = 0    # blown through stop

        pos[t] = current

    return pd.Series(pos, index=zscore.index, name="position")


# ══════════════════════════════════════════════════════════════════════════════
# Signal Result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SignalResult:
    """
    Full output of one SignalEngine.run() call.

    Attributes
    ----------
    y_name / x_name : str
    hedge_method : str
    passed_stationarity : bool
    passed_cointegration : bool
    passed_gate : bool
        True only if both stationarity and cointegration gates passed.

    y / x : pd.Series
        Aligned price series (common index, NaN dropped).
    hedge_ratios : pd.Series
        β(t) — may be constant (OLS) or time-varying (Kalman/rolling OLS).
    spread : pd.Series
        z(t) = y(t) - β(t)·x(t)
    zscore : pd.Series
        Rolling z-score of the spread.
    position : pd.Series
        Discrete position: +1 (long spread), -1 (short spread), 0 (flat).
    trade_entries : pd.Series
        Boolean mask of bars where position changed from 0 to ±1.
    trade_exits : pd.Series
        Boolean mask of bars where position changed from ±1 to 0.

    stat_result_y / stat_result_x : StationarityResult
    coint_result : CointegrationResult
    kalman_result : KalmanResult or None
        Populated when hedge_method == 'kalman'.

    config_* : float
        Entry/exit/stop thresholds used.
    """

    # ── Identity ───────────────────────────────────────────────────────────
    y_name: str
    x_name: str
    hedge_method: HedgeMethod

    # ── Gate results ───────────────────────────────────────────────────────
    passed_stationarity: bool
    passed_cointegration: bool

    # ── Price & spread ─────────────────────────────────────────────────────
    y: pd.Series
    x: pd.Series
    hedge_ratios: pd.Series
    spread: pd.Series
    zscore: pd.Series
    position: pd.Series
    trade_entries: pd.Series
    trade_exits: pd.Series

    # ── Upstream results ───────────────────────────────────────────────────
    stat_result_y: StationarityResult
    stat_result_x: StationarityResult
    coint_result: CointegrationResult
    kalman_result: Optional[KalmanResult]

    # ── Config echoed back ──────────────────────────────────────────────────
    zscore_window: int
    zscore_entry: float
    zscore_exit: float
    zscore_stop: float

    @property
    def passed_gate(self) -> bool:
        return self.passed_stationarity and self.passed_cointegration

    @property
    def n_trades(self) -> int:
        return int(self.trade_entries.sum())

    @property
    def current_position(self) -> int:
        return int(self.position.iloc[-1])

    @property
    def current_zscore(self) -> float:
        return float(self.zscore.iloc[-1])

    @property
    def current_hedge_ratio(self) -> float:
        return float(self.hedge_ratios.iloc[-1])

    @property
    def time_in_market(self) -> float:
        """Fraction of bars where |position| = 1."""
        return float((self.position != 0).mean())

    def summary(self) -> str:
        lines = [
            f"\n{'═'*65}",
            f"  Signal Engine — {self.y_name} / {self.x_name}",
            f"  hedge={self.hedge_method} | z_window={self.zscore_window}",
            f"{'═'*65}",
            f"",
            f"  Pre-flight gates:",
            f"    Stationarity (both I(1)) : "
            f"{'✅ PASS' if self.passed_stationarity else '❌ FAIL'}",
            f"    Cointegration            : "
            f"{'✅ PASS' if self.passed_cointegration else '❌ FAIL'}",
            f"    → Proceed to trade       : "
            f"{'✅ YES' if self.passed_gate else '❌ NO'}",
        ]

        if not self.passed_gate:
            lines += [
                f"",
                f"  ⚠️  Gate failed — no signals generated.",
                f"{'═'*65}",
            ]
            output = "\n".join(lines)
            print(output)
            return output

        lines += [
            f"",
            f"  Hedge Ratio β(t):",
            f"    Current                  : {self.current_hedge_ratio:>10.4f}",
            f"    Mean                     : {self.hedge_ratios.mean():>10.4f}",
            f"    Std                      : {self.hedge_ratios.std():>10.4f}",
            f"",
            f"  Spread:",
            f"    Mean                     : {self.spread.mean():>10.4f}",
            f"    Std                      : {self.spread.std():>10.4f}",
            f"    Min / Max                : {self.spread.min():>10.4f} / {self.spread.max():.4f}",
            f"",
            f"  Z-score (window={self.zscore_window}):",
            f"    Current                  : {self.current_zscore:>10.4f}",
            f"    Entry ±{self.zscore_entry} | Exit ±{self.zscore_exit} | Stop ±{self.zscore_stop}",
            f"",
            f"  Signals:",
            f"    Trades (entries)         : {self.n_trades:>10}",
            f"    Time in market           : {self.time_in_market:>10.1%}",
            f"    Current position         : {self.current_position:>10}  "
            f"({'long spread' if self.current_position==1 else 'short spread' if self.current_position==-1 else 'flat'})",
            f"",
            f"  Cointegration (EG p-val)  : {self.coint_result.eg_pvalue:.4f}",
            f"  Rolling coint fraction    : {self.coint_result.rolling_coint_fraction:.1%}",
        ]

        if self.kalman_result:
            lines += [
                f"  Kalman half-life          : {self.kalman_result.half_life_days:.1f} days",
                f"  Kalman Hurst exponent     : {self.kalman_result.hurst_exponent:.4f}",
            ]

        lines += [f"{'═'*65}"]
        output = "\n".join(lines)
        print(output)
        return output

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a tidy DataFrame with all signal columns aligned on the same index.

        Columns: y, x, hedge_ratio, spread, zscore, position
        """
        df = pd.DataFrame({
            self.y_name: self.y,
            self.x_name: self.x,
            "hedge_ratio": self.hedge_ratios,
            "spread": self.spread,
            "zscore": self.zscore,
            "position": self.position,
        })
        return df

    def latest(self) -> pd.Series:
        """Most recent bar as a Series — handy for live trading."""
        return pd.Series({
            "y_name": self.y_name,
            "x_name": self.x_name,
            "hedge_ratio": round(self.current_hedge_ratio, 4),
            "spread": round(float(self.spread.iloc[-1]), 4),
            "zscore": round(self.current_zscore, 4),
            "position": self.current_position,
            "passed_gate": self.passed_gate,
        })


# ══════════════════════════════════════════════════════════════════════════════
# Signal Engine
# ══════════════════════════════════════════════════════════════════════════════

class SignalEngine:
    """
    End-to-end signal pipeline for a cointegrated pair.

    Steps
    -----
    1. Align prices (drop NaN, common index)
    2. Stationarity gate  — both legs must be I(1)
    3. Cointegration gate — EG + Johansen must both pass
    4. Hedge ratio        — Kalman | rolling OLS | static OLS
    5. Spread             — y(t) - β(t)·x(t)
    6. Z-score            — rolling (spread - μ) / σ
    7. Position           — state-machine on z-score

    Parameters
    ----------
    hedge_method : str
        'kalman' (default) | 'rolling_ols' | 'static_ols'
    zscore_window : int or None
        Rolling window for z-score. None → use config value (60).
    stat_significance : float or None
        Alpha for ADF/KPSS. None → use config (0.05).
    coint_significance : float or None
        Alpha for EG/Johansen. None → use config (0.05).
    coint_rolling_window : int or None
        Rolling window for stability check. None → use config (252).
    zscore_entry / zscore_exit / zscore_stop : float or None
        Signal thresholds. None → use config values.
    rolling_ols_window : int or None
        Window for rolling OLS. None → use config (60).
    kalman_delta / kalman_vt : float or None
        Kalman parameters. None → use config.
    gate_mode : str
        'full'  — skip signal generation if gate fails (default)
        'warn'  — log warning but still generate signals
        'skip'  — bypass gate entirely
    """

    def __init__(
        self,
        hedge_method: HedgeMethod = "kalman",
        zscore_window: Optional[int] = None,
        stat_significance: Optional[float] = None,
        coint_significance: Optional[float] = None,
        coint_rolling_window: Optional[int] = None,
        zscore_entry: Optional[float] = None,
        zscore_exit: Optional[float] = None,
        zscore_stop: Optional[float] = None,
        rolling_ols_window: Optional[int] = None,
        kalman_delta: Optional[float] = None,
        kalman_vt: Optional[float] = None,
        gate_mode: Literal["full", "warn", "skip"] = "full",
    ) -> None:
        cfg = get_config().stats

        self.hedge_method         = hedge_method
        self.zscore_window        = zscore_window        or cfg.zscore_window
        self.stat_significance    = stat_significance    or cfg.adf_significance
        self.coint_significance   = coint_significance   or cfg.coint_significance
        self.coint_rolling_window = coint_rolling_window or cfg.coint_rolling_window
        self.zscore_entry         = zscore_entry         or cfg.zscore_entry
        self.zscore_exit          = zscore_exit          or cfg.zscore_exit
        self.zscore_stop          = zscore_stop          or cfg.zscore_stop
        self.rolling_ols_window   = rolling_ols_window   or cfg.rolling_ols_window
        self.kalman_delta         = kalman_delta         or cfg.kalman_delta
        self.kalman_vt            = kalman_vt            or cfg.kalman_vt
        self.gate_mode            = gate_mode

        # Sub-modules
        self._stat_checker  = StationarityChecker(significance=self.stat_significance)
        self._coint_checker = CointegrationChecker(
            significance=self.coint_significance,
            rolling_window=self.coint_rolling_window,
        )

    # ── Gate ───────────────────────────────────────────────────────────────

    def _run_stationarity_gate(
        self, y: pd.Series, x: pd.Series, y_name: str, x_name: str
    ) -> Tuple[bool, StationarityResult, StationarityResult]:
        r_y, r_x = self._stat_checker.test_pair(y, x, y_name=y_name, x_name=x_name)
        passed = r_y.is_i1 and r_x.is_i1
        if not passed:
            logger.warning(
                "Stationarity gate FAILED | {}: is_i1={} | {}: is_i1={}",
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
                "Cointegration gate FAILED | {}/{} | eg_p={:.4f} | johansen_rank={}",
                y_name, x_name, r.eg_pvalue, r.johansen_rank,
            )
        return passed, r

    # ── Hedge ratio ────────────────────────────────────────────────────────

    def _compute_hedge_ratio(
        self, y: pd.Series, x: pd.Series, y_name: str, x_name: str
    ) -> Tuple[pd.Series, Optional[KalmanResult]]:
        """
        Returns (hedge_ratios, kalman_result_or_None).
        hedge_ratios is a pd.Series aligned with y/x index.
        """
        if self.hedge_method == "kalman":
            kf = KalmanHedgeRatio(delta=self.kalman_delta, vt=self.kalman_vt)
            kr = kf.fit(y, x, y_name=y_name, x_name=x_name)
            return kr.hedge_ratios, kr

        elif self.hedge_method == "rolling_ols":
            betas = _rolling_ols_beta(y, x, window=self.rolling_ols_window)
            logger.info(
                "Rolling OLS hedge | {} / {} | window={} | current_beta={:.4f}",
                y_name, x_name, self.rolling_ols_window,
                float(betas.dropna().iloc[-1]) if betas.dropna().shape[0] > 0 else float("nan"),
            )
            return betas, None

        elif self.hedge_method == "static_ols":
            beta, alpha = _static_ols_beta(y.values, x.values)
            betas = pd.Series(beta, index=y.index, name="hedge_ratio")
            logger.info(
                "Static OLS hedge | {} / {} | beta={:.4f} | alpha={:.4f}",
                y_name, x_name, beta, alpha,
            )
            return betas, None

        else:
            raise ValueError(f"Unknown hedge_method: {self.hedge_method!r}")

    # ── Spread & z-score ───────────────────────────────────────────────────

    @staticmethod
    def _compute_spread(
        y: pd.Series, x: pd.Series, hedge_ratios: pd.Series
    ) -> pd.Series:
        spread = y - hedge_ratios * x
        spread.name = "spread"
        return spread

    def _compute_zscore(self, spread: pd.Series) -> pd.Series:
        mu    = spread.rolling(self.zscore_window).mean()
        sigma = spread.rolling(self.zscore_window).std()
        zscore = (spread - mu) / sigma
        zscore.name = "zscore"
        return zscore

    # ── Main entry point ───────────────────────────────────────────────────

    def run(
        self,
        y: pd.Series,
        x: pd.Series,
        y_name: Optional[str] = None,
        x_name: Optional[str] = None,
    ) -> SignalResult:
        """
        Run the full signal pipeline for a pair.

        Parameters
        ----------
        y : pd.Series  — leg1 prices
        x : pd.Series  — leg2 prices
        y_name / x_name : str, optional

        Returns
        -------
        SignalResult
            Always returns a result. If gate fails (gate_mode='full'),
            spread/zscore/position will be empty Series.
        """
        y_label = y_name or (y.name or "Y")
        x_label = x_name or (x.name or "X")

        logger.info(
            "SignalEngine.run | {} / {} | method={} | gate_mode={}",
            y_label, x_label, self.hedge_method, self.gate_mode,
        )

        # ── 1. Align ───────────────────────────────────────────────────────
        df = pd.DataFrame({"y": y, "x": x}).dropna()
        y_aligned = df["y"].rename(y_label)
        x_aligned = df["x"].rename(x_label)

        # ── 2. Stationarity gate ───────────────────────────────────────────
        if self.gate_mode == "skip":
            passed_stat = True
            # Still need result objects — run tests but ignore outcome
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
                    passed_stat, False, stat_r_y, stat_r_x,
                )

        # ── 3. Cointegration gate ──────────────────────────────────────────
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

        # ── 4. Hedge ratio ─────────────────────────────────────────────────
        hedge_ratios, kalman_result = self._compute_hedge_ratio(
            y_aligned, x_aligned, y_label, x_label
        )

        # ── 5. Spread ──────────────────────────────────────────────────────
        spread = self._compute_spread(y_aligned, x_aligned, hedge_ratios)

        # ── 6. Z-score ─────────────────────────────────────────────────────
        zscore = self._compute_zscore(spread)

        # ── 7. Positions ───────────────────────────────────────────────────
        position = _generate_positions(
            zscore,
            entry=self.zscore_entry,
            exit_=self.zscore_exit,
            stop=self.zscore_stop,
        )

        # ── 8. Trade markers ───────────────────────────────────────────────
        pos_prev      = position.shift(1).fillna(0).astype(int)
        trade_entries = (pos_prev == 0) & (position != 0)
        trade_exits   = (pos_prev != 0) & (position == 0)

        logger.info(
            "SignalEngine complete | {}/{} | n_trades={} | "
            "current_z={:.4f} | current_pos={}",
            y_label, x_label,
            int(trade_entries.sum()),
            float(zscore.iloc[-1]) if not zscore.empty else float("nan"),
            int(position.iloc[-1]) if not position.empty else 0,
        )

        return SignalResult(
            y_name=y_label,
            x_name=x_label,
            hedge_method=self.hedge_method,
            passed_stationarity=passed_stat,
            passed_cointegration=passed_coint,
            y=y_aligned,
            x=x_aligned,
            hedge_ratios=hedge_ratios,
            spread=spread,
            zscore=zscore,
            position=position,
            trade_entries=trade_entries,
            trade_exits=trade_exits,
            stat_result_y=stat_r_y,
            stat_result_x=stat_r_x,
            coint_result=coint_r,
            kalman_result=kalman_result,
            zscore_window=self.zscore_window,
            zscore_entry=self.zscore_entry,
            zscore_exit=self.zscore_exit,
            zscore_stop=self.zscore_stop,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _empty_result(
        self,
        y_label: str,
        x_label: str,
        y: pd.Series,
        x: pd.Series,
        passed_stat: bool,
        passed_coint: bool,
        stat_r_y: StationarityResult,
        stat_r_x: StationarityResult,
        coint_result: Optional[CointegrationResult] = None,
    ) -> SignalResult:
        """Return a gate-failed SignalResult with empty signal arrays."""
        empty = pd.Series(dtype=float)
        empty_int = pd.Series(dtype=int)
        empty_bool = pd.Series(dtype=bool)

        if coint_result is None:
            # Build a minimal dummy CointegrationResult so the dataclass is always populated
            coint_result = self._coint_checker.test(y, x, y_name=y_label, x_name=x_label)

        return SignalResult(
            y_name=y_label,
            x_name=x_label,
            hedge_method=self.hedge_method,
            passed_stationarity=passed_stat,
            passed_cointegration=passed_coint,
            y=y,
            x=x,
            hedge_ratios=empty,
            spread=empty,
            zscore=empty,
            position=empty_int,
            trade_entries=empty_bool,
            trade_exits=empty_bool,
            stat_result_y=stat_r_y,
            stat_result_x=stat_r_x,
            coint_result=coint_result,
            kalman_result=None,
            zscore_window=self.zscore_window,
            zscore_entry=self.zscore_entry,
            zscore_exit=self.zscore_exit,
            zscore_stop=self.zscore_stop,
        )


# ── Smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

    print("\n" + "="*65)
    print("  SIGNAL ENGINE SMOKE TEST")
    print("  Synthetic cointegrated pair — all three hedge methods")
    print("="*65)

    rng = np.random.default_rng(42)
    n   = 800

    # ── Build a synthetic cointegrated pair ───────────────────────────────
    # x is a random walk; y = 2·x + slow AR(1) noise
    x_rw = pd.Series(np.cumsum(rng.normal(0, 1, n)), name="X")
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.7 * noise[t - 1] + rng.normal(0, 0.3)
    y_coint = (2.0 * x_rw + pd.Series(noise, name="Y")).rename("Y")

    # ── Engine — Kalman ───────────────────────────────────────────────────
    print("\n── Method: Kalman (gate_mode=full) ──")
    engine_k = SignalEngine(
        hedge_method="kalman",
        coint_rolling_window=252,
        gate_mode="full",
    )
    r_k = engine_k.run(y_coint, x_rw, y_name="Y", x_name="X")
    r_k.summary()

    assert r_k.passed_gate,      "❌ Synthetic coint pair should pass gate"
    assert not r_k.spread.empty, "❌ Spread should be populated"
    assert not r_k.zscore.empty, "❌ Z-score should be populated"
    assert r_k.n_trades >= 0,    "❌ n_trades should be non-negative"
    assert r_k.current_hedge_ratio > 0, "❌ Hedge ratio should be positive"
    print("✅ PASS: Kalman engine — gate, spread, zscore, position all valid")

    # ── Engine — Rolling OLS ──────────────────────────────────────────────
    print("\n── Method: Rolling OLS (gate_mode=warn) ──")
    engine_r = SignalEngine(
        hedge_method="rolling_ols",
        rolling_ols_window=60,
        gate_mode="warn",
    )
    r_r = engine_r.run(y_coint, x_rw, y_name="Y", x_name="X")
    assert not r_r.spread.dropna().empty, "❌ Rolling OLS spread should have valid values"
    print(f"  Rolling OLS current β = {r_r.current_hedge_ratio:.4f}")
    print("✅ PASS: Rolling OLS engine")

    # ── Engine — Static OLS ───────────────────────────────────────────────
    print("\n── Method: Static OLS (gate_mode=skip) ──")
    engine_s = SignalEngine(hedge_method="static_ols", gate_mode="skip")
    r_s = engine_s.run(y_coint, x_rw, y_name="Y", x_name="X")
    assert r_s.hedge_ratios.nunique() == 1, "❌ Static OLS should have constant hedge ratio"
    print(f"  Static OLS β = {r_s.current_hedge_ratio:.4f}  (true = 2.0)")
    assert abs(r_s.current_hedge_ratio - 2.0) < 0.3, "❌ Static OLS beta far from true value"
    print("✅ PASS: Static OLS engine")

    # ── Gate failure ──────────────────────────────────────────────────────
    print("\n── Gate failure: non-cointegrated pair ──")
    y_noise = pd.Series(rng.normal(0, 1, n), name="Y_noise")
    x_noise = pd.Series(rng.normal(0, 1, n), name="X_noise")
    engine_gate = SignalEngine(hedge_method="kalman", gate_mode="full")
    r_gate = engine_gate.run(y_noise, x_noise, y_name="Y_noise", x_name="X_noise")
    r_gate.summary()
    assert not r_gate.passed_gate, "❌ Noise pair should fail gate"
    assert r_gate.spread.empty,    "❌ Gate-failed result should have empty spread"
    print("✅ PASS: Gate correctly blocked noise pair")

    # ── to_dataframe() ────────────────────────────────────────────────────
    print("\n── to_dataframe() ──")
    df = r_k.to_dataframe()
    assert set(["Y", "X", "hedge_ratio", "spread", "zscore", "position"]).issubset(df.columns), \
        "❌ to_dataframe() missing expected columns"
    print(df.tail(3))
    print("✅ PASS: to_dataframe() structure correct")

    # ── latest() ─────────────────────────────────────────────────────────
    print("\n── latest() ──")
    latest = r_k.latest()
    print(latest)
    assert "position" in latest.index, "❌ latest() missing 'position'"
    print("✅ PASS: latest() works correctly")

    print("\n" + "="*65)
    print("  ALL SIGNAL ENGINE SMOKE TESTS PASSED ✅")
    print("="*65 + "\n")
