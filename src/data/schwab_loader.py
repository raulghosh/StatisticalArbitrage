"""
schwab_loader.py — Schwab API data layer.

Responsibilities:
  1. OAuth2 authentication (token managed by schwab-py)
  2. Pull OHLCV price history for futures symbols
  3. Cache to Parquet (avoid hammering the API during research)
  4. Return clean, timezone-aware DataFrames

Usage:
    from src.data.schwab_loader import SchwabLoader
    loader = SchwabLoader()
    df = loader.get_price_history("/GC", years=10)
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import schwab
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_config


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# Column rename map: Schwab JSON → our standard names
COL_MAP = {
    "open":   "open",
    "high":   "high",
    "low":    "low",
    "close":  "close",
    "volume": "volume",
}


# ══════════════════════════════════════════════════════════════════════════════
# Loader class
# ══════════════════════════════════════════════════════════════════════════════

class SchwabLoader:
    """
    Thin wrapper around schwab-py for market data.

    The client uses OAuth2 with a local token file. On first run it opens
    a browser for you to log in. Subsequent runs refresh silently.

    Parameters
    ----------
    token_path : Path, optional
        Where to store the OAuth token. Defaults to project root.
    """

    def __init__(self, token_path: Optional[Path] = None) -> None:
        self.cfg = get_config()
        self.token_path = token_path or self.cfg.schwab_token_path
        self._client: Optional[schwab.client.Client] = None
        logger.info("SchwabLoader initialised | token_path={}", self.token_path)

    # ── Authentication ─────────────────────────────────────────────────────

    def _get_client(self) -> schwab.client.Client:
        """
        Return an authenticated Schwab client.

        First call: opens browser for OAuth login, saves token.
        Subsequent calls: loads + auto-refreshes token from disk.
        """
        if self._client is not None:
            return self._client

        app_key    = self.cfg.schwab_app_key
        app_secret = self.cfg.schwab_app_secret
        callback   = self.cfg.schwab_callback_url

        if not app_key or not app_secret:
            raise EnvironmentError(
                "SCHWAB_APP_KEY and SCHWAB_APP_SECRET must be set in .env"
            )

        if self.token_path.exists():
            logger.info("Loading existing Schwab token from {}", self.token_path)
            self._client = schwab.auth.client_from_token_file(
                token_path=str(self.token_path),
                api_key=app_key,
                app_secret=app_secret,
            )
        else:
            logger.info("No token found — opening browser for OAuth login...")
            self._client = schwab.auth.client_from_login_flow(
                api_key=app_key,
                app_secret=app_secret,
                callback_url=callback,
                token_path=str(self.token_path),
            )
            logger.success("Schwab OAuth complete. Token saved to {}", self.token_path)

        return self._client

    # ── Price History ──────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _fetch_from_api(
        self,
        symbol: str,
        years: int,
    ) -> pd.DataFrame:
        """
        Raw API call — returns a clean DataFrame.

        FIX: get_price_history_every_day() in newer schwab-py does NOT accept
        period_type/period kwargs. Use start_datetime/end_datetime instead.
        The method already sets FrequencyType=DAILY internally.

        Parameters
        ----------
        symbol : str
            Schwab futures symbol, e.g. "/GC", "/SI", "/CL", "/BZ"
        years : int
            Number of years of daily history to pull

        Returns
        -------
        pd.DataFrame
            DatetimeIndex (UTC), columns: open, high, low, close, volume
        """
        client = self._get_client()

        # ── Date range (timezone-aware) ────────────────────────────────────
        end_dt   = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=int(years * 365.25))

        logger.info(
            "Fetching {} | {} → {} | via start/end datetime",
            symbol,
            start_dt.date(),
            end_dt.date(),
        )

        # ── API call (no period_type — method sets DAILY internally) ───────
        resp = client.get_price_history_every_day(
            symbol=symbol,
            start_datetime=start_dt,
            end_datetime=end_dt,
            need_extended_hours_data=False,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Schwab API error {resp.status_code} for {symbol}: {resp.text}"
            )

        data = resp.json()

        if "candles" not in data or not data["candles"]:
            raise ValueError(
                f"No candles returned for {symbol}. "
                f"Check symbol format — futures require leading slash e.g. '/GC'"
            )

        df = pd.DataFrame(data["candles"])

        # ── Timestamp: Schwab returns epoch milliseconds ───────────────────
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
        df = df.set_index("datetime").sort_index()

        # ── Rename & select columns ────────────────────────────────────────
        df = df.rename(columns=COL_MAP)[list(COL_MAP.values())]
        df = df.astype({
            "open":   float,
            "high":   float,
            "low":    float,
            "close":  float,
            "volume": float,
        })

        # ── Sanity checks ──────────────────────────────────────────────────
        n_nulls = df.isnull().sum().sum()
        if n_nulls > 0:
            logger.warning("{} has {} null values — forward-filling", symbol, n_nulls)
            df = df.ffill()

        logger.success(
            "Fetched {} | rows={} | {} → {}",
            symbol,
            len(df),
            df.index[0].date(),
            df.index[-1].date(),
        )
        return df

    # ── Cache logic ────────────────────────────────────────────────────────

    def _cache_path(self, symbol: str) -> Path:
        """Parquet file path for a symbol."""
        safe_name = symbol.replace("/", "FUTURES_")
        return self.cfg.data.cache_dir / f"{safe_name}_daily.parquet"

    def _is_cache_fresh(self, path: Path) -> bool:
        """Return True if cache file exists and is within TTL."""
        if not path.exists():
            return False
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        fresh = age_hours < self.cfg.data.cache_ttl_hours
        if not fresh:
            logger.info("Cache stale ({:.1f}h old) — will re-fetch", age_hours)
        return fresh

    def _save_cache(self, df: pd.DataFrame, path: Path) -> None:
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        logger.debug("Cached {} rows → {}", len(df), path.name)

    def _load_cache(self, path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info("Loaded from cache: {} ({} rows)", path.name, len(df))
        return df

    # ── Public interface ───────────────────────────────────────────────────

    def get_price_history(
        self,
        symbol: str,
        years: Optional[int] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return daily OHLCV for a futures symbol.

        Uses Parquet cache — only hits Schwab API when cache is stale
        or force_refresh=True.

        Parameters
        ----------
        symbol : str
            e.g. "/GC", "/SI", "/CL", "/BZ"
        years : int, optional
            Years of history. Defaults to config value (10).
        force_refresh : bool
            Bypass cache and fetch fresh from API.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex (UTC), columns: open, high, low, close, volume
        """
        years = years or self.cfg.data.history_years
        cache_path = self._cache_path(symbol)

        if not force_refresh and self._is_cache_fresh(cache_path):
            return self._load_cache(cache_path)

        df = self._fetch_from_api(symbol, years)
        self._save_cache(df, cache_path)
        return df

    def get_pair_prices(
        self,
        pair_key: str,
        column: str = "close",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch both legs of a configured pair, aligned on common dates.

        Parameters
        ----------
        pair_key : str
            Key in config.data.pairs, e.g. "gold_silver"
        column : str
            Which OHLCV column to return ("close" for strategy use)

        Returns
        -------
        pd.DataFrame
            DatetimeIndex, columns: [leg1_symbol, leg2_symbol]
            Only dates where BOTH instruments have data (inner join).
        """
        pair = self.cfg.data.pairs.get(pair_key)
        if pair is None:
            raise KeyError(
                f"Unknown pair '{pair_key}'. Available: {list(self.cfg.data.pairs)}"
            )

        sym1 = pair["leg1_schwab"]
        sym2 = pair["leg2_schwab"]

        df1 = self.get_price_history(sym1, force_refresh=force_refresh)[column].rename(sym1)
        df2 = self.get_price_history(sym2, force_refresh=force_refresh)[column].rename(sym2)

        # Inner join — only keep dates where BOTH have prices
        combined = pd.concat([df1, df2], axis=1).dropna()

        # Normalize to date (drop time for daily work)
        combined.index = combined.index.normalize()

        logger.info(
            "Pair '{}' aligned | rows={} | {} → {}",
            pair_key,
            len(combined),
            combined.index[0].date(),
            combined.index[-1].date(),
        )
        return combined

    def list_cached_symbols(self) -> list[str]:
        """Show what's already in cache."""
        files = list(self.cfg.data.cache_dir.glob("*_daily.parquet"))
        symbols = [
            f.stem.replace("FUTURES_", "/").replace("_daily", "")
            for f in files
        ]
        return sorted(symbols)


# ── Quick smoke test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = SchwabLoader()
    print("Cached symbols:", loader.list_cached_symbols())

    df_gc = loader.get_price_history("/GC", years=5)
    print(f"\n/GC shape : {df_gc.shape}")
    print(df_gc.tail(3))

    pair_df = loader.get_pair_prices("gold_silver")
    print(f"\nGold/Silver pair shape: {pair_df.shape}")
    print(pair_df.tail(3))