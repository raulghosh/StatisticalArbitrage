"""
stationarity.py — ADF & KPSS Stationarity Tests

Theory:
  Before testing cointegration we need to verify each price series is
  integrated of order 1, i.e. I(1):
    - The LEVEL is non-stationary (has a unit root)
    - The FIRST DIFFERENCE is stationary

  Two complementary tests with opposite null hypotheses:

  ADF (Augmented Dickey-Fuller):
    H0: Series has a unit root (non-stationary)
    H1: Series is stationary
    → Reject H0 (low p-value) → stationary

  KPSS (Kwiatkowski-Phillips-Schmidt-Shin):
    H0: Series is stationary (trend-stationary)
    H1: Series has a unit root
    → Reject H0 (low p-value) → non-stationary

  Combined interpretation:
    I(1) candidate:
      Level   → ADF fails to reject (p > α) AND KPSS rejects (p < α)
      Diff    → ADF rejects (p < α) AND KPSS fails to reject (p > α)

Usage:
    from src.stats.stationarity import StationarityChecker
    checker = StationarityChecker()
    result = checker.test(price_series, name="Gold (/GC)")
    result.summary()
    print(result.is_i1)   # True if suitable for cointegration
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.stattools import adfuller, kpss


# ══════════════════════════════════════════════════════════════════════════════
# Result Dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StationarityResult:
    """
    Full stationarity test output for a single price series.

    Attributes
    ----------
    name : str
        Series label.
    n_obs : int
        Number of observations used.

    adf_stat : float
        ADF test statistic (more negative = more evidence against unit root).
    adf_pvalue : float
        ADF p-value. Low (< α) → reject H0 → series is stationary.
    adf_critical : dict
        Critical values at 1%, 5%, 10%.
    adf_stationary : bool
        True if ADF p-value < significance level.

    kpss_stat : float
        KPSS test statistic.
    kpss_pvalue : float
        KPSS p-value. Low (< α) → reject H0 → series is non-stationary.
    kpss_critical : dict
        Critical values at 1%, 2.5%, 5%, 10%.
    kpss_nonstationary : bool
        True if KPSS p-value < significance level.

    diff_adf_stat / diff_adf_pvalue : float
        Same tests on the first-differenced series.
    diff_kpss_stat / diff_kpss_pvalue : float
        Same tests on the first-differenced series.

    significance : float
        Alpha level used for decisions.
    """

    name: str
    n_obs: int
    significance: float

    # ── Level tests ────────────────────────────────────────────────────────
    adf_stat: float
    adf_pvalue: float
    adf_critical: dict
    adf_stationary: bool

    kpss_stat: float
    kpss_pvalue: float
    kpss_critical: dict
    kpss_nonstationary: bool

    # ── First-difference tests ─────────────────────────────────────────────
    diff_adf_stat: float
    diff_adf_pvalue: float
    diff_adf_stationary: bool

    diff_kpss_stat: float
    diff_kpss_pvalue: float
    diff_kpss_nonstationary: bool

    @property
    def level_has_unit_root(self) -> bool:
        """
        Level is non-stationary.

        ADF fails to reject (p > α) AND KPSS rejects (p < α).
        Both tests agree the level is non-stationary.
        """
        return (not self.adf_stationary) and self.kpss_nonstationary

    @property
    def diff_is_stationary(self) -> bool:
        """
        First difference is stationary.

        ADF rejects (p < α) AND KPSS fails to reject (p > α).
        Both tests agree the diff is stationary.
        """
        return self.diff_adf_stationary and (not self.diff_kpss_nonstationary)

    @property
    def is_i1(self) -> bool:
        """
        True if the series is integrated of order 1 — suitable for cointegration.

        Requires:
          - Level has a unit root (non-stationary)
          - First difference is stationary
        """
        return self.level_has_unit_root and self.diff_is_stationary

    def summary(self) -> str:
        def pval_str(p: float) -> str:
            return f"{p:.4f}" if p < 0.999 else ">0.999"

        lines = [
            f"\n{'─'*65}",
            f"  Stationarity Tests — {self.name}",
            f"  n={self.n_obs} | α={self.significance}",
            f"{'─'*65}",
            f"  Level Series:",
            f"    ADF  stat={self.adf_stat:>9.4f}  p={pval_str(self.adf_pvalue)}"
            f"  → {'STATIONARY' if self.adf_stationary else 'unit root (non-stationary)'}",
            f"    KPSS stat={self.kpss_stat:>9.4f}  p={pval_str(self.kpss_pvalue)}"
            f"  → {'non-stationary' if self.kpss_nonstationary else 'stationary'}",
            f"    Level unit root : {'✅ YES (good for coint)' if self.level_has_unit_root else '❌ UNCLEAR'}",
            f"",
            f"  First-Difference:",
            f"    ADF  stat={self.diff_adf_stat:>9.4f}  p={pval_str(self.diff_adf_pvalue)}"
            f"  → {'STATIONARY' if self.diff_adf_stationary else 'non-stationary'}",
            f"    KPSS stat={self.diff_kpss_stat:>9.4f}  p={pval_str(self.diff_kpss_pvalue)}"
            f"  → {'stationary' if not self.diff_kpss_nonstationary else 'NON-STATIONARY'}",
            f"    Diff stationary : {'✅ YES (good for coint)' if self.diff_is_stationary else '❌ UNCLEAR'}",
            f"",
            f"  Verdict: {'✅ I(1) — SUITABLE FOR COINTEGRATION' if self.is_i1 else '⚠️  NOT CONFIRMED I(1)'}",
            f"{'─'*65}",
        ]
        output = "\n".join(lines)
        print(output)
        return output

    def to_series(self) -> pd.Series:
        return pd.Series({
            "name": self.name,
            "n_obs": self.n_obs,
            "adf_stat": round(self.adf_stat, 4),
            "adf_pvalue": round(self.adf_pvalue, 4),
            "adf_stationary": self.adf_stationary,
            "kpss_stat": round(self.kpss_stat, 4),
            "kpss_pvalue": round(self.kpss_pvalue, 4),
            "kpss_nonstationary": self.kpss_nonstationary,
            "level_has_unit_root": self.level_has_unit_root,
            "diff_is_stationary": self.diff_is_stationary,
            "is_i1": self.is_i1,
        })


# ══════════════════════════════════════════════════════════════════════════════
# Stationarity Checker
# ══════════════════════════════════════════════════════════════════════════════

class StationarityChecker:
    """
    Run ADF and KPSS tests on a price series to determine I(1) status.

    Parameters
    ----------
    significance : float
        Alpha level for hypothesis tests. Default: 0.05.
    adf_regression : str
        ADF regression type: 'c' (constant), 'ct' (constant+trend),
        'ctt', or 'n' (no constant). Price series typically use 'c'.
    kpss_regression : str
        KPSS regression type: 'c' (level stationary) or 'ct' (trend stationary).
    adf_maxlags : int or None
        Max lags for ADF. None = automatic via Akaike IC.
    """

    def __init__(
        self,
        significance: float = 0.05,
        adf_regression: str = "c",
        kpss_regression: str = "c",
        adf_maxlags: Optional[int] = None,
    ) -> None:
        self.significance    = significance
        self.adf_regression  = adf_regression
        self.kpss_regression = kpss_regression
        self.adf_maxlags     = adf_maxlags

    def _run_adf(self, series: pd.Series) -> Tuple[float, float, dict]:
        """Run ADF and return (stat, pvalue, critical_values)."""
        result = adfuller(
            series.dropna().values,
            regression=self.adf_regression,
            maxlag=self.adf_maxlags,
            autolag="AIC",
        )
        stat, pvalue, critical = float(result[0]), float(result[1]), result[4]
        return stat, pvalue, critical

    def _run_kpss(self, series: pd.Series) -> Tuple[float, float, dict]:
        """Run KPSS and return (stat, pvalue, critical_values)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pvalue, _, critical = kpss(
                series.dropna().values,
                regression=self.kpss_regression,
                nlags="auto",
            )
        return float(stat), float(pvalue), critical

    def test(
        self,
        series: pd.Series,
        name: Optional[str] = None,
    ) -> StationarityResult:
        """
        Run ADF + KPSS on both the level and first difference.

        Parameters
        ----------
        series : pd.Series
            Price series (must have ≥ 20 observations).
        name : str, optional
            Label for logging and summary output.

        Returns
        -------
        StationarityResult
        """
        series_clean = series.dropna()
        label = name or (series.name or "Series")
        n = len(series_clean)

        if n < 20:
            raise ValueError(f"Need ≥ 20 observations, got {n} for '{label}'.")

        logger.info("Stationarity test | {} | n={} | α={}", label, n, self.significance)

        # ── Level ──────────────────────────────────────────────────────────
        adf_stat, adf_p, adf_crit    = self._run_adf(series_clean)
        kpss_stat, kpss_p, kpss_crit = self._run_kpss(series_clean)

        adf_stationary     = adf_p < self.significance
        kpss_nonstationary = kpss_p < self.significance

        # ── First difference ───────────────────────────────────────────────
        diff_series = series_clean.diff().dropna()

        d_adf_stat, d_adf_p, _   = self._run_adf(diff_series)
        d_kpss_stat, d_kpss_p, _ = self._run_kpss(diff_series)

        diff_adf_stationary     = d_adf_p < self.significance
        diff_kpss_nonstationary = d_kpss_p < self.significance

        result = StationarityResult(
            name=label,
            n_obs=n,
            significance=self.significance,
            adf_stat=adf_stat,
            adf_pvalue=adf_p,
            adf_critical=adf_crit,
            adf_stationary=adf_stationary,
            kpss_stat=kpss_stat,
            kpss_pvalue=kpss_p,
            kpss_critical=kpss_crit,
            kpss_nonstationary=kpss_nonstationary,
            diff_adf_stat=d_adf_stat,
            diff_adf_pvalue=d_adf_p,
            diff_adf_stationary=diff_adf_stationary,
            diff_kpss_stat=d_kpss_stat,
            diff_kpss_pvalue=d_kpss_p,
            diff_kpss_nonstationary=diff_kpss_nonstationary,
        )

        logger.info(
            "Stationarity | {} | is_i1={} | adf_p={:.4f} | kpss_p={:.4f}",
            label, result.is_i1, adf_p, kpss_p,
        )

        return result

    def test_pair(
        self,
        y: pd.Series,
        x: pd.Series,
        y_name: Optional[str] = None,
        x_name: Optional[str] = None,
    ) -> Tuple[StationarityResult, StationarityResult]:
        """
        Test both legs of a pair for I(1) status.

        Returns
        -------
        (result_y, result_x)
        """
        y_label = y_name or (y.name or "Y")
        x_label = x_name or (x.name or "X")

        r_y = self.test(y, name=y_label)
        r_x = self.test(x, name=x_label)

        both_i1 = r_y.is_i1 and r_x.is_i1
        logger.info(
            "Pair stationarity | {}/{} | both_i1={}",
            y_label, x_label, both_i1,
        )
        return r_y, r_x


# ── Smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

    print("\n" + "="*65)
    print("  STATIONARITY SMOKE TEST")
    print("  Synthetic I(1) vs Stationary series")
    print("="*65)

    rng = np.random.default_rng(42)
    n   = 500

    # ── Synthetic I(1) series (random walk) ───────────────────────────────
    rw = pd.Series(np.cumsum(rng.normal(0, 1, n)), name="RandomWalk_I1")

    # ── Synthetic stationary series (AR(1) with φ=0.7) ────────────────────
    ar1 = np.zeros(n)
    for t in range(1, n):
        ar1[t] = 0.7 * ar1[t - 1] + rng.normal(0, 1)
    ar1_series = pd.Series(ar1, name="AR1_Stationary")

    checker = StationarityChecker(significance=0.05)

    print("\n── Test 1: Random Walk (expect is_i1 = True) ──")
    r1 = checker.test(rw)
    r1.summary()
    assert r1.is_i1, "❌ Random walk should be I(1)"
    print("✅ PASS: Random walk correctly identified as I(1)")

    print("\n── Test 2: Stationary AR(1) (expect is_i1 = False) ──")
    r2 = checker.test(ar1_series)
    r2.summary()
    assert not r2.is_i1, "❌ Stationary AR(1) should NOT be I(1)"
    print("✅ PASS: Stationary AR(1) correctly identified as NOT I(1)")

    print("\n── Test 3: Pair test on two correlated random walks ──")
    rw2 = pd.Series(np.cumsum(rng.normal(0, 1, n)), name="RandomWalk2_I1")
    r_y, r_x = checker.test_pair(rw, rw2, y_name="RW1", x_name="RW2")
    both_i1 = r_y.is_i1 and r_x.is_i1
    print(f"  Both I(1): {both_i1}")
    assert both_i1, "❌ Both random walks should be I(1)"
    print("✅ PASS: Both legs confirmed I(1)")

    print("\n── Test 4: to_series() output ──")
    s = r1.to_series()
    print(s)
    assert "is_i1" in s.index, "❌ to_series() missing 'is_i1'"
    print("✅ PASS: to_series() works correctly")

    print("\n" + "="*65)
    print("  ALL STATIONARITY SMOKE TESTS PASSED ✅")
    print("="*65 + "\n")
