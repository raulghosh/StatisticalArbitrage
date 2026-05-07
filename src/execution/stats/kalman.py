"""
kalman.py — Kalman Filter for Dynamic Hedge Ratio Estimation

Built from first principles. No black-box libraries.

Theory:
  We model the spread as a state-space system where the hedge ratio β(t)
  is a hidden state that evolves over time (random walk):

  Observation equation:  y(t) = β(t)·x(t) + ε(t),   ε(t) ~ N(0, Vt)
  State equation:        β(t) = β(t-1) + δ(t),       δ(t) ~ N(0, Wt)

  Where:
    y(t)  = leg1 price (e.g. /GC Gold)
    x(t)  = leg2 price (e.g. /SI Silver)
    β(t)  = dynamic hedge ratio (what we estimate)
    ε(t)  = observation noise (spread noise)
    δ(t)  = state noise (how much β can drift per step)

  The Kalman filter gives us the OPTIMAL linear estimate of β(t)
  given all observations up to time t. "Optimal" = minimum variance.

  Key parameters:
    delta (Wt/Vt ratio): controls how fast β can change
      - Large delta → β tracks prices aggressively (responsive but noisy)
      - Small delta → β changes slowly (smooth but lags regime changes)
      - Typical range: 1e-5 to 1e-3. We use 1e-4.

Kalman Filter Equations (5-step predict-update cycle):
  Predict:
    β̂(t|t-1) = β̂(t-1|t-1)              (state prediction = random walk)
    P(t|t-1)  = P(t-1|t-1) + Wt          (variance prediction)

  Update (when new observation arrives):
    K(t)      = P(t|t-1)·x(t) / [x(t)²·P(t|t-1) + Vt]   (Kalman gain)
    β̂(t|t)   = β̂(t|t-1) + K(t)·[y(t) - β̂(t|t-1)·x(t)] (state update)
    P(t|t)    = [1 - K(t)·x(t)]·P(t|t-1)                 (variance update)

  The innovation: ν(t) = y(t) - β̂(t|t-1)·x(t)  (prediction error)
  A large innovation → K increases → β updates more aggressively.

Usage:
    from src.stats.kalman import KalmanHedgeRatio
    kf = KalmanHedgeRatio(delta=1e-4, vt=1e-3)
    result = kf.fit(df["/GC"], df["/SI"])
    result.summary()
    spread = result.spread
    hedge_ratios = result.hedge_ratios
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ══════════════════════════════════════════════════════════════════════════════
# Result Dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class KalmanResult:
    """
    Output of the Kalman Filter hedge ratio estimation.

    Attributes
    ----------
    hedge_ratios : pd.Series
        Dynamic β(t) — the time-varying hedge ratio. Shape: (T,)
    variances : pd.Series
        P(t|t) — posterior variance of β(t). Shape: (T,)
        High variance = high uncertainty in current hedge ratio.
    kalman_gains : pd.Series
        K(t) — how aggressively the filter updated at each step.
        K → 0 means β is very stable. K → 1 means highly uncertain.
    innovations : pd.Series
        ν(t) = y(t) - β̂(t|t-1)·x(t) — prediction errors.
        Large innovations signal regime changes or spread dislocations.
    spread : pd.Series
        z(t) = y(t) - β(t)·x(t) — the dynamic spread.
        This is what we trade. Should be mean-reverting.
    half_life_days : float
        Estimated half-life of mean reversion in the spread (days).
        From fitting AR(1): spread(t) = φ·spread(t-1) + ε
        Half-life = -log(2)/log(φ)
    hurst_exponent : float
        H < 0.5 → mean-reverting (we want this)
        H = 0.5 → random walk
        H > 0.5 → trending
    ou_theta : float
        Ornstein-Uhlenbeck mean reversion speed θ.
        Annualized: θ = -log(φ)·252
    ou_mu : float
        O-U long-run mean μ.
    ou_sigma : float
        O-U diffusion parameter σ.
    """
    y_name: str
    x_name: str
    delta: float
    vt: float

    hedge_ratios: pd.Series
    variances: pd.Series
    kalman_gains: pd.Series
    innovations: pd.Series
    spread: pd.Series

    # ── Spread analytics ───────────────────────────────────────────────────
    half_life_days: float
    hurst_exponent: float
    ou_theta: float
    ou_mu: float
    ou_sigma: float

    @property
    def current_hedge_ratio(self) -> float:
        return float(self.hedge_ratios.iloc[-1])

    @property
    def current_variance(self) -> float:
        return float(self.variances.iloc[-1])

    @property
    def is_mean_reverting(self) -> bool:
        return self.hurst_exponent < 0.50

    @property
    def has_tradeable_halflife(self) -> bool:
        """Half-life between 2 and 40 days — practical for daily trading."""
        return 2.0 <= self.half_life_days <= 40.0

    @property
    def is_tradeable(self) -> bool:
        return self.is_mean_reverting and self.has_tradeable_halflife

    def summary(self) -> str:
        lines = [
            f"\n{'─'*65}",
            f"  Kalman Filter — Dynamic Hedge Ratio",
            f"  {self.y_name}  ~  β(t)·{self.x_name}",
            f"{'─'*65}",
            f"  Parameters:",
            f"    delta (state noise)  : {self.delta:.2e}",
            f"    Vt (obs noise)       : {self.vt:.2e}",
            f"",
            f"  Hedge Ratio β(t):",
            f"    Current (latest)     : {self.current_hedge_ratio:>10.4f}",
            f"    Mean                 : {self.hedge_ratios.mean():>10.4f}",
            f"    Std Dev              : {self.hedge_ratios.std():>10.4f}",
            f"    Min                  : {self.hedge_ratios.min():>10.4f}",
            f"    Max                  : {self.hedge_ratios.max():>10.4f}",
            f"",
            f"  Filter Diagnostics:",
            f"    Current variance P   : {self.current_variance:>10.6f}",
            f"    Mean Kalman gain K   : {self.kalman_gains.mean():>10.6f}",
            f"    Innovation std       : {self.innovations.std():>10.4f}",
            f"",
            f"  Spread Analytics:",
            f"    Half-life (days)     : {self.half_life_days:>10.2f}",
            f"    Hurst Exponent       : {self.hurst_exponent:>10.4f}",
            f"    O-U θ (annualized)   : {self.ou_theta:>10.4f}",
            f"    O-U μ (long-run mean): {self.ou_mu:>10.4f}",
            f"    O-U σ (diffusion)    : {self.ou_sigma:>10.4f}",
            f"",
            f"  Tradability checks:",
            f"    Mean-reverting (H<0.5)   : {'✅ YES' if self.is_mean_reverting else '❌ NO'}  (H={self.hurst_exponent:.4f})",
            f"    Half-life 2-40 days      : {'✅ YES' if self.has_tradeable_halflife else '❌ NO'}  ({self.half_life_days:.1f}d)",
            f"    Overall tradeable        : {'✅ PROCEED' if self.is_tradeable else '❌ DO NOT TRADE'}",
            f"{'─'*65}",
        ]
        output = "\n".join(lines)
        print(output)
        return output

    def to_series(self) -> pd.Series:
        return pd.Series({
            "y": self.y_name,
            "x": self.x_name,
            "delta": self.delta,
            "current_hedge_ratio": round(self.current_hedge_ratio, 4),
            "hedge_ratio_mean": round(self.hedge_ratios.mean(), 4),
            "hedge_ratio_std": round(self.hedge_ratios.std(), 4),
            "half_life_days": round(self.half_life_days, 2),
            "hurst_exponent": round(self.hurst_exponent, 4),
            "ou_theta": round(self.ou_theta, 4),
            "ou_mu": round(self.ou_mu, 4),
            "ou_sigma": round(self.ou_sigma, 4),
            "is_mean_reverting": self.is_mean_reverting,
            "has_tradeable_halflife": self.has_tradeable_halflife,
            "is_tradeable": self.is_tradeable,
        })

    def get_zscore(self, window: int = 60) -> pd.Series:
        """
        Compute rolling z-score of the Kalman spread.

        z(t) = (spread(t) - μ_rolling(t)) / σ_rolling(t)

        This is the primary trading signal. Delegated here so the
        signal module can call result.get_zscore(window=60).
        """
        mu    = self.spread.rolling(window).mean()
        sigma = self.spread.rolling(window).std()
        zscore = (self.spread - mu) / sigma
        zscore.name = "zscore"
        return zscore


# ══════════════════════════════════════════════════════════════════════════════
# Spread analytics helpers
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_halflife(spread: pd.Series) -> float:
    """
    Estimate half-life of mean reversion from AR(1) fit.

    Regress: Δspread(t) = θ·spread(t-1) + ε
    φ = 1 + θ  (AR coefficient)
    Half-life = -log(2) / log(φ)

    If φ ≥ 1 (non-mean-reverting), returns inf.
    """
    spread_clean = spread.dropna()
    delta_spread = spread_clean.diff().dropna()
    spread_lag   = spread_clean.shift(1).dropna()

    # Align
    common_idx = delta_spread.index.intersection(spread_lag.index)
    dy = delta_spread.loc[common_idx].values
    y_lag = spread_lag.loc[common_idx].values

    if len(dy) < 10:
        return float("inf")

    # OLS: Δy = θ·y_lag
    X = y_lag.reshape(-1, 1)
    theta = float(np.linalg.lstsq(X, dy, rcond=None)[0][0])
    phi = 1 + theta  # AR coefficient

    if phi >= 1.0:
        return float("inf")   # Not mean-reverting

    half_life = -np.log(2) / np.log(abs(phi))
    return max(0.0, float(half_life))


def _estimate_hurst(spread: pd.Series, max_lag: int = 100) -> float:
    """
    Estimate Hurst exponent using rescaled range (R/S) analysis.

    For a range of window sizes τ:
        E[R(τ)/S(τ)] ∝ τ^H

    H < 0.5 → mean-reverting (anti-persistent)
    H = 0.5 → random walk (Brownian motion)
    H > 0.5 → trending (persistent)

    We fit log(R/S) = H·log(τ) + c via OLS.
    """
    series = spread.dropna().values
    n = len(series)

    if n < max_lag * 2:
        max_lag = n // 4

    lags = range(2, max_lag + 1)
    rs_values = []

    for lag in lags:
        # Subseries of length lag starting at each position
        sub_series = [series[i: i + lag] for i in range(0, n - lag, lag)]

        rs_list = []
        for sub in sub_series:
            if len(sub) < 2:
                continue
            mean_sub = np.mean(sub)
            deviations = np.cumsum(sub - mean_sub)
            r = np.max(deviations) - np.min(deviations)  # Range
            s = np.std(sub, ddof=1)                        # Std deviation
            if s > 0:
                rs_list.append(r / s)

        if rs_list:
            rs_values.append(np.mean(rs_list))
        else:
            rs_values.append(np.nan)

    # OLS: log(RS) = H · log(lag) + c
    log_lags = np.log(list(lags))
    log_rs   = np.log(rs_values)

    valid = np.isfinite(log_rs)
    if valid.sum() < 5:
        return 0.5  # Can't estimate — return neutral

    X = np.column_stack([np.ones(valid.sum()), log_lags[valid]])
    y = log_rs[valid]
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    hurst = float(coeffs[1])

    return max(0.0, min(1.0, hurst))


def _fit_ou_params(spread: pd.Series) -> Tuple[float, float, float]:
    """
    Fit Ornstein-Uhlenbeck parameters via maximum likelihood (discrete).

    The discrete O-U process:
        spread(t) = μ + (spread(t-1) - μ)·exp(-θ·dt) + σ_ε·ε(t)

    Equivalent AR(1):
        spread(t) = c + φ·spread(t-1) + ε(t)

    Where:
        φ     = exp(-θ·dt)
        μ     = c / (1 - φ)
        σ_ε²  = σ²·(1 - exp(-2θ·dt)) / (2θ)

    We use dt = 1/252 (daily).

    Returns (theta_annualized, mu, sigma_annualized)
    """
    dt = 1.0 / 252.0
    s = spread.dropna()
    n = len(s)

    if n < 20:
        return 0.0, float(s.mean()), float(s.std())

    s_lag = s.shift(1).dropna()
    s_curr = s.iloc[1:]

    # OLS: s(t) = c + φ·s(t-1) + ε
    X = np.column_stack([np.ones(len(s_lag)), s_lag.values])
    y = s_curr.values
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    c, phi = float(coeffs[0]), float(coeffs[1])

    if phi <= 0 or phi >= 1:
        # Not mean-reverting — return neutral params
        return 0.0, float(s.mean()), float(s.std())

    theta  = -np.log(phi) / dt                         # annualized
    mu     = c / (1 - phi)
    resid  = y - c - phi * s_lag.values
    s2_eps = float(np.var(resid))
    sigma2 = s2_eps * 2 * theta / (1 - np.exp(-2 * theta * dt))
    sigma  = np.sqrt(max(sigma2, 1e-12)) * np.sqrt(252)  # annualized

    return float(theta), float(mu), float(sigma)


# ══════════════════════════════════════════════════════════════════════════════
# Kalman Filter
# ══════════════════════════════════════════════════════════════════════════════

class KalmanHedgeRatio:
    """
    Kalman Filter for dynamic hedge ratio estimation.

    State-space model:
        Observation: y(t) = β(t)·x(t) + ε(t),   ε ~ N(0, Vt)
        State:       β(t) = β(t-1) + δ(t),       δ ~ N(0, Wt)

    Parameters
    ----------
    delta : float
        State transition variance (Wt = delta * I).
        Controls how fast β can drift.
        Default: 1e-4 (β can drift ~1% per 100 steps)

    vt : float
        Observation noise variance.
        Roughly: (typical daily spread move)²
        Default: 1e-3
    """

    def __init__(self, delta: float = 1e-4, vt: float = 1e-3) -> None:
        self.delta = delta
        self.vt    = vt
        logger.info("KalmanHedgeRatio initialised | delta={} | vt={}", delta, vt)

    def fit(
        self,
        y: pd.Series,
        x: pd.Series,
        y_name: Optional[str] = None,
        x_name: Optional[str] = None,
    ) -> KalmanResult:
        """
        Run the Kalman filter on the full price history.

        Parameters
        ----------
        y : pd.Series  — leg1 prices (e.g. /GC Gold)
        x : pd.Series  — leg2 prices (e.g. /SI Silver)

        Returns
        -------
        KalmanResult
            Contains hedge_ratios, spread, half_life, Hurst, O-U params.
        """
        y_name = y_name or (y.name or "Y")
        x_name = x_name or (x.name or "X")

        # ── Align ──────────────────────────────────────────────────────────
        df = pd.DataFrame({"y": y, "x": x}).dropna()
        y_arr = df["y"].values.astype(float)
        x_arr = df["x"].values.astype(float)
        n     = len(df)

        logger.info(
            "Kalman fit | {} ~ β(t)·{} | n={} | delta={} | vt={}",
            y_name, x_name, n, self.delta, self.vt,
        )

        # ── Initialise state ───────────────────────────────────────────────
        # β̂(0|0) initialized to OLS estimate on first 20 observations
        init_n   = min(20, n // 5)
        X_init   = np.column_stack([np.ones(init_n), x_arr[:init_n]])
        ols_init = np.linalg.lstsq(X_init, y_arr[:init_n], rcond=None)[0]
        beta_hat = ols_init[1]              # Initial β estimate
        P        = 1.0                      # Initial variance (uncertain)

        Wt = self.delta                     # State noise variance (scalar)
        Vt = self.vt                        # Observation noise variance

        # ── Storage arrays ─────────────────────────────────────────────────
        beta_arr  = np.zeros(n)
        P_arr     = np.zeros(n)
        K_arr     = np.zeros(n)
        innov_arr = np.zeros(n)

        # ── Filter loop ────────────────────────────────────────────────────
        for t in range(n):
            x_t = x_arr[t]
            y_t = y_arr[t]

            # ── PREDICT ───────────────────────────────────────────────────
            # β̂(t|t-1) = β̂(t-1|t-1)   [random walk — no drift]
            beta_pred = beta_hat
            P_pred    = P + Wt            # P(t|t-1) = P(t-1|t-1) + Wt

            # ── INNOVATION ────────────────────────────────────────────────
            # ν(t) = y(t) - β̂(t|t-1)·x(t)
            innovation = y_t - beta_pred * x_t

            # ── KALMAN GAIN ────────────────────────────────────────────────
            # K(t) = P(t|t-1)·x(t) / [x(t)²·P(t|t-1) + Vt]
            S_t = x_t ** 2 * P_pred + Vt   # Innovation variance
            K   = P_pred * x_t / S_t       # Kalman gain

            # ── UPDATE ─────────────────────────────────────────────────────
            # β̂(t|t) = β̂(t|t-1) + K(t)·ν(t)
            beta_hat = beta_pred + K * innovation
            # P(t|t)  = (1 - K·x(t))·P(t|t-1)   [Joseph form for stability]
            P = (1 - K * x_t) * P_pred

            # ── Store ──────────────────────────────────────────────────────
            beta_arr[t]  = beta_hat
            P_arr[t]     = P
            K_arr[t]     = K
            innov_arr[t] = innovation

        # ── Build output series ────────────────────────────────────────────
        idx = df.index
        hedge_ratios = pd.Series(beta_arr,  index=idx, name="hedge_ratio")
        variances    = pd.Series(P_arr,     index=idx, name="variance")
        kalman_gains = pd.Series(K_arr,     index=idx, name="kalman_gain")
        innovations  = pd.Series(innov_arr, index=idx, name="innovation")

        # Dynamic spread: y(t) - β(t)·x(t)
        spread = pd.Series(
            y_arr - beta_arr * x_arr,
            index=idx,
            name=f"spread_{y_name}_{x_name}",
        )

        # ── Spread analytics ───────────────────────────────────────────────
        half_life = _estimate_halflife(spread)
        hurst     = _estimate_hurst(spread)
        ou_theta, ou_mu, ou_sigma = _fit_ou_params(spread)

        logger.info(
            "Kalman complete | current_beta={:.4f} | half_life={:.1f}d | hurst={:.4f}",
            float(beta_arr[-1]), half_life, hurst,
        )

        return KalmanResult(
            y_name=y_name,
            x_name=x_name,
            delta=self.delta,
            vt=self.vt,
            hedge_ratios=hedge_ratios,
            variances=variances,
            kalman_gains=kalman_gains,
            innovations=innovations,
            spread=spread,
            half_life_days=half_life,
            hurst_exponent=hurst,
            ou_theta=ou_theta,
            ou_mu=ou_mu,
            ou_sigma=ou_sigma,
        )

    def update_one(
        self,
        beta_prev: float,
        P_prev: float,
        y_t: float,
        x_t: float,
    ) -> Tuple[float, float, float, float]:
        """
        Single-step Kalman update — for live trading.

        Called each day with the latest prices to update the hedge ratio.

        Parameters
        ----------
        beta_prev : float  — β̂(t-1|t-1) previous posterior estimate
        P_prev    : float  — P(t-1|t-1) previous posterior variance
        y_t       : float  — current leg1 price
        x_t       : float  — current leg2 price

        Returns
        -------
        (beta_new, P_new, K, innovation)
        """
        # Predict
        P_pred     = P_prev + self.delta
        innovation = y_t - beta_prev * x_t

        # Update
        S_t      = x_t ** 2 * P_pred + self.vt
        K        = P_pred * x_t / S_t
        beta_new = beta_prev + K * innovation
        P_new    = (1 - K * x_t) * P_pred

        return float(beta_new), float(P_new), float(K), float(innovation)


# ══════════════════════════════════════════════════════════════════════════════
# Parameter sensitivity analysis
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_delta(
    y: pd.Series,
    x: pd.Series,
    deltas: Optional[list] = None,
    vt: float = 1e-3,
    y_name: Optional[str] = None,
    x_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Sweep delta values and compare half-life and Hurst exponent.

    Use this to find the optimal delta for your pair.

    Returns
    -------
    pd.DataFrame
        One row per delta value with half_life, hurst, current_hedge_ratio.
    """
    if deltas is None:
        deltas = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]

    y_name = y_name or (y.name or "Y")
    x_name = x_name or (x.name or "X")

    rows = []
    for d in deltas:
        kf = KalmanHedgeRatio(delta=d, vt=vt)
        r  = kf.fit(y, x, y_name=y_name, x_name=x_name)
        rows.append({
            "delta": d,
            "half_life_days": round(r.half_life_days, 2),
            "hurst_exponent": round(r.hurst_exponent, 4),
            "current_hedge_ratio": round(r.current_hedge_ratio, 4),
            "hedge_ratio_std": round(r.hedge_ratios.std(), 4),
            "is_mean_reverting": r.is_mean_reverting,
            "has_tradeable_halflife": r.has_tradeable_halflife,
            "is_tradeable": r.is_tradeable,
        })

    df = pd.DataFrame(rows)
    print(f"\n{'─'*75}")
    print(f"  Delta Calibration — {y_name} / {x_name}")
    print(f"{'─'*75}")
    print(df.to_string(index=False))
    print(f"{'─'*75}\n")

    return df


# ── Smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

    from src.data.schwab_loader import SchwabLoader

    print("\n" + "="*65)
    print("  KALMAN FILTER SMOKE TEST")
    print("  Dynamic Hedge Ratio: Gold (/GC) vs Silver (/SI)")
    print("="*65)

    loader = SchwabLoader()
    pair_df = loader.get_pair_prices("gold_silver")
    gc = pair_df["/GC"].rename("Gold (/GC)")
    si = pair_df["/SI"].rename("Silver (/SI)")

    # ── Fit Kalman filter ──────────────────────────────────────────────────
    kf = KalmanHedgeRatio(delta=1e-4, vt=1e-3)
    result = kf.fit(gc, si)
    result.summary()

    # ── Compute z-score ────────────────────────────────────────────────────
    zscore = result.get_zscore(window=60)
    print(f"\nZ-score (last 5 values):")
    print(zscore.tail())

    # ── Delta calibration ──────────────────────────────────────────────────
    print("\n\nDelta sensitivity analysis:")
    calib = calibrate_delta(gc, si)

    # ── Machine-readable ───────────────────────────────────────────────────
    print("\nResult as Series:")
    print(result.to_series())