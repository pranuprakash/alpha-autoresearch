"""Options chain fetching, Greeks, and IV rank utilities."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("PlayOptions")

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> Dict[str, float]:
    """Black-Scholes price and Greeks. T in years."""
    if not HAS_SCIPY or T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.40 if option_type == "call" else -0.40,
                "theta": -0.02, "vega": 0.10, "gamma": 0.005, "bs_price": 0.0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        delta = float(norm.cdf(d1))
        bs_price = float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        delta = float(norm.cdf(d1) - 1)
        bs_price = float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

    gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
    vega = float(S * norm.pdf(d1) * np.sqrt(T) / 100)   # per 1% IV move
    theta_sign = 1 if option_type == "call" else 1
    theta = float(
        -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
          - r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type == "call" else norm.cdf(-d2)))
        / 365
    )

    return {
        "delta": delta,
        "theta": theta,
        "vega": vega,
        "gamma": gamma,
        "bs_price": max(bs_price, 0.01),
    }


def fetch_options_chain(
    ticker: str,
    target_dte_range: Tuple[int, int] = (20, 60),
) -> Optional[Dict[str, Any]]:
    """
    Fetch options chain for the expiry closest to 35 DTE within target_dte_range.
    Falls back to nearest expiry > 7 DTE if none in range.
    """
    if not HAS_YF:
        return None

    try:
        t = yf.Ticker(ticker)
        info = t.fast_info if hasattr(t, "fast_info") else t.info
        try:
            price = float(info.get("last_price") or info.get("regularMarketPrice") or 0)
        except Exception:
            price = 0.0

        if not price:
            hist = t.history(period="2d")
            price = float(hist["Close"].iloc[-1]) if not hist.empty else None
        if not price:
            return None

        expirations = t.options
        if not expirations:
            return None

        now = datetime.now()
        best_expiry: Optional[str] = None
        best_dte: Optional[int] = None

        for exp in expirations:
            try:
                dte = (datetime.strptime(exp, "%Y-%m-%d") - now).days
                if target_dte_range[0] <= dte <= target_dte_range[1]:
                    if best_dte is None or abs(dte - 35) < abs(best_dte - 35):
                        best_expiry, best_dte = exp, dte
            except ValueError:
                continue

        if not best_expiry:
            for exp in expirations:
                try:
                    dte = (datetime.strptime(exp, "%Y-%m-%d") - now).days
                    if dte > 7:
                        best_expiry, best_dte = exp, dte
                        break
                except ValueError:
                    continue

        if not best_expiry:
            return None

        chain = t.option_chain(best_expiry)
        return {
            "ticker": ticker,
            "price": price,
            "expiry": best_expiry,
            "dte": best_dte or 30,
            "calls": chain.calls,
            "puts": chain.puts,
        }

    except Exception as e:
        logger.warning(f"Options chain fetch failed for {ticker}: {e}")
        return None


def select_strike_by_delta(
    chain_df: Any,
    target_delta: float,
    option_type: str,
    price: float,
    expiry: str,
    risk_free_rate: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """Select the contract whose delta is closest to target_delta."""
    if chain_df is None or len(chain_df) == 0:
        return None

    try:
        now = datetime.now()
        T = max((datetime.strptime(expiry, "%Y-%m-%d") - now).days / 365.0, 0.001)
    except Exception:
        T = 30 / 365.0

    liquid = chain_df[chain_df["bid"] > 0].copy() if "bid" in chain_df.columns else chain_df.copy()
    if liquid.empty:
        liquid = chain_df.copy()

    best: Optional[Dict[str, Any]] = None
    best_diff = float("inf")

    for _, row in liquid.iterrows():
        K = float(row["strike"])
        bid = float(row.get("bid", 0))
        ask = float(row.get("ask", 0))
        mid = (bid + ask) / 2 if (bid + ask) > 0 else float(row.get("lastPrice", 0))
        if mid <= 0:
            continue

        iv = float(row.get("impliedVolatility", 0.30))
        if iv <= 0:
            iv = 0.30

        greeks = black_scholes_greeks(price, K, T, risk_free_rate, iv, option_type)
        diff = abs(abs(greeks["delta"]) - abs(target_delta))

        if diff < best_diff:
            best_diff = diff
            best = {
                "strike": K,
                "mid_price": round(mid, 2),
                "bid": round(bid, 2),
                "ask": round(ask, 2),
                "iv": iv,
                "open_interest": int(row.get("openInterest", 0)),
                "volume": int(row.get("volume", 0)),
                **greeks,
            }

    return best


def compute_iv_rank(ticker: str, current_iv: float) -> float:
    """
    Approximate IV rank (0–100) using 52-week rolling realized vol as proxy.
    Returns 50 on any failure.
    """
    if not HAS_YF:
        return 50.0
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if len(hist) < 30:
            return 50.0
        log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        rv = log_ret.rolling(21).std() * np.sqrt(252)
        rv = rv.dropna()
        if rv.empty:
            return 50.0
        lo, hi = float(rv.min()), float(rv.max())
        if hi <= lo:
            return 50.0
        return float(np.clip((current_iv - lo) / (hi - lo) * 100, 0, 100))
    except Exception:
        return 50.0
