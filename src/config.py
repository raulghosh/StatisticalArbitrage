"""
config.py — Central configuration for the StatArb project.

All parameters live here. Never hardcode values in strategy files.
Loaded once at startup; accessed everywhere via get_config().
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

# ── Load .env ──────────────────────────────────────────────────────────────
load_dotenv()

# ── Project root (one level up from src/) ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ══════════════════════════════════════════════════════════════════════════════
# Sub-configs (plain Pydantic models — no env reading)
# ══════════════════════════════════════════════════════════════════════════════

class DataConfig(BaseModel):
    """Instrument universe and data settings."""

    # Schwab futures symbols → Alpaca ETF proxy symbols
    pairs: Dict[str, Dict[str, str]] = {
        "gold_silver": {
            "leg1_schwab": "/GC",   # Gold futures
            "leg2_schwab": "/SI",   # Silver futures
            "leg1_alpaca": "GLD",   # Gold ETF proxy
            "leg2_alpaca": "SLV",   # Silver ETF proxy
            "name": "Gold/Silver",
        },
        "wti_brent": {
            "leg1_schwab": "/CL",   # WTI futures
            "leg2_schwab": "/BZ",   # Brent futures
            "leg1_alpaca": "USO",   # WTI ETF proxy
            "leg2_alpaca": "BNO",   # Brent ETF proxy
            "name": "WTI/Brent",
        },
    }

    # Price history
    history_years: int = 10          # Years of daily data to pull
    frequency: str = "daily"         # daily | weekly
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"

    # Cache TTL: re-fetch if file older than this many hours
    cache_ttl_hours: int = 12


class StatConfig(BaseModel):
    """Parameters for statistical tests and signal generation."""

    # ── Stationarity ──────────────────────────────────────────────────────
    adf_significance: float = 0.05          # ADF p-value threshold
    kpss_significance: float = 0.05         # KPSS p-value threshold

    # ── Cointegration ─────────────────────────────────────────────────────
    coint_significance: float = 0.05        # Johansen trace test threshold
    coint_rolling_window: int = 252         # Rolling window (1 year daily)
    min_coint_fraction: float = 0.70        # % of rolling windows that must be cointegrated

    # ── Hedge Ratio ────────────────────────────────────────────────────────
    hedge_method: str = "kalman"            # "ols" | "rolling_ols" | "kalman"
    rolling_ols_window: int = 60            # days for rolling OLS
    kalman_delta: float = 1e-4             # Kalman: state transition variance (smoothness)
    kalman_vt: float = 1e-3               # Kalman: observation variance

    # ── Spread Z-score ─────────────────────────────────────────────────────
    zscore_window: int = 60                 # Rolling mean/std window
    zscore_entry: float = 2.0              # Enter when |z| > this
    zscore_exit: float = 0.5              # Exit when |z| < this
    zscore_stop: float = 3.5              # Hard stop when |z| > this

    # ── Ornstein-Uhlenbeck ─────────────────────────────────────────────────
    min_halflife_days: int = 2             # Reject if too fast (noise)
    max_halflife_days: int = 40            # Reject if too slow (not mean-reverting)
    min_hurst_threshold: float = 0.50     # Must be < 0.5 (mean-reverting)


class RiskConfig(BaseModel):
    """Position sizing and risk management parameters."""

    target_annual_vol: float = 0.10        # 10% annualized portfolio vol
    max_position_pct: float = 0.20        # Max 20% NAV per pair
    max_drawdown_pct: float = 0.15        # Stop trading if DD > 15%

    # Transaction costs (realistic for ETFs)
    slippage_bps: float = 5.0             # 5 bps slippage per side
    commission_per_share: float = 0.0     # Alpaca = $0 commission

    # Regime filter: pause if rolling coint p-value > this
    regime_coint_threshold: float = 0.10


class BacktestConfig(BaseModel):
    """Backtesting engine parameters."""

    initial_capital: float = 100_000.0
    in_sample_end: str = "2022-12-31"     # Walk-forward split
    out_sample_start: str = "2023-01-01"
    risk_free_rate: float = 0.05          # For Sharpe calculation


# ══════════════════════════════════════════════════════════════════════════════
# Main Settings (reads from environment)
# ══════════════════════════════════════════════════════════════════════════════

class Settings(BaseSettings):
    """Top-level settings. Reads API credentials from environment."""

    model_config = {"env_file": ".env", "extra": "ignore"}

    # ── Schwab credentials ─────────────────────────────────────────────────
    schwab_app_key: str = Field(default="", alias="SCHWAB_APP_KEY")
    schwab_app_secret: str = Field(default="", alias="SCHWAB_APP_SECRET")
    schwab_callback_url: str = Field(
        default="https://127.0.0.1:3000/api/schwab/callback",
        alias="SCHWAB_CALLBACK_URL",
    )
    schwab_token_path: Path = PROJECT_ROOT / ".schwab_token.json"

    # ── Alpaca credentials ─────────────────────────────────────────────────
    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY_PAPER")
    alpaca_secret: str = Field(default="", alias="ALPACA_SECRET")
    alpaca_paper: bool = True  # Always paper for now

    # ── Sub-configs (static — not from env) ───────────────────────────────
    data: DataConfig = DataConfig()
    stats: StatConfig = StatConfig()
    risk: RiskConfig = RiskConfig()
    backtest: BacktestConfig = BacktestConfig()

    @field_validator("schwab_app_key", "schwab_app_secret", mode="before")
    @classmethod
    def warn_if_empty(cls, v: str, info) -> str:
        if not v:
            print(f"⚠️  WARNING: {info.field_name} is not set in .env")
        return v

    def validate_dirs(self) -> None:
        """Create cache/raw dirs if they don't exist."""
        for d in [self.data.cache_dir, self.data.raw_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ── Singleton accessor ──────────────────────────────────────────────────────
_config: Settings | None = None


def get_config() -> Settings:
    """Return the singleton Settings instance."""
    global _config
    if _config is None:
        _config = Settings()
        _config.validate_dirs()
    return _config


# ── Quick sanity check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = get_config()
    print("✅ Config loaded")
    print(f"   Schwab key   : {'set' if cfg.schwab_app_key else 'MISSING'}")
    print(f"   Alpaca key   : {'set' if cfg.alpaca_api_key else 'MISSING'}")
    print(f"   Cache dir    : {cfg.data.cache_dir}")
    print(f"   Pairs        : {list(cfg.data.pairs.keys())}")
    print(f"   Z-score entry: ±{cfg.stats.zscore_entry}")
    print(f"   Hedge method : {cfg.stats.hedge_method}")