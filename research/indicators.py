"""
Quantitative indicator library for market research.

All indicators take a DataFrame with standard OHLCV columns and return
a dict of computed values. No LLM calls here — pure math.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────
# Trend & Momentum
# ─────────────────────────────────────────

def compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Average Directional Index — trend strength (0-100). >25 = strong trend."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    dm_plus = (high.diff()).clip(lower=0)
    dm_minus = (-low.diff()).clip(lower=0)
    dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

    atr = tr.ewm(span=period, adjust=False).mean()
    di_plus = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan))
    adx = dx.ewm(span=period, adjust=False).mean()
    return float(adx.iloc[-1]) if not adx.empty else 0.0


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Relative Strength Index (0-100). >70 overbought, <30 oversold."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # loss=0: RSI=100 if trending up, 50 if flat
    zero_loss = loss == 0
    rsi = rsi.where(~zero_loss, other=(gain > 0).astype(float) * 50 + 50)
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Dict[str, float]:
    """MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": round(float(macd_line.iloc[-1]), 4),
        "signal": round(float(signal_line.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
        "bullish_cross": bool(macd_line.iloc[-1] > signal_line.iloc[-1] and
                              macd_line.iloc[-2] <= signal_line.iloc[-2]),
        "bearish_cross": bool(macd_line.iloc[-1] < signal_line.iloc[-1] and
                              macd_line.iloc[-2] >= signal_line.iloc[-2]),
    }


def compute_momentum(series: pd.Series, periods: int = 20) -> float:
    """Rate of change over N periods (%)."""
    if len(series) < periods + 1:
        return 0.0
    return float((series.iloc[-1] / series.iloc[-periods] - 1) * 100)


# ─────────────────────────────────────────
# Volatility
# ─────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range — absolute volatility."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return float(atr.iloc[-1]) if not atr.empty else 0.0


def compute_bollinger(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> Dict[str, float]:
    """Bollinger Bands — squeeze detection + band position."""
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    last = series.iloc[-1]
    band_width = float((upper.iloc[-1] - lower.iloc[-1]) / ma.iloc[-1]) if ma.iloc[-1] != 0 else 0.0
    pct_b = float((last - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])) if (upper.iloc[-1] - lower.iloc[-1]) != 0 else 0.5
    return {
        "upper": round(float(upper.iloc[-1]), 4),
        "middle": round(float(ma.iloc[-1]), 4),
        "lower": round(float(lower.iloc[-1]), 4),
        "band_width": round(band_width, 4),
        "pct_b": round(pct_b, 4),
        "squeeze": band_width < 0.10,    # very tight bands = squeeze
        "extended_high": pct_b > 0.95,   # near upper band
        "extended_low": pct_b < 0.05,    # near lower band
    }


def compute_historical_volatility(series: pd.Series, period: int = 20) -> float:
    """Annualized historical volatility (%)."""
    returns = np.log(series / series.shift(1)).dropna()
    if len(returns) < period:
        return 0.0
    return float(returns.tail(period).std() * np.sqrt(252) * 100)


# ─────────────────────────────────────────
# Volume / Microstructure
# ─────────────────────────────────────────

def compute_vwap_deviation(df: pd.DataFrame) -> float:
    """% deviation from 20-day VWAP."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].replace(0, np.nan)
    vwap = (typical_price * vol).rolling(20).sum() / vol.rolling(20).sum()
    last_vwap = vwap.iloc[-1]
    last_close = df["Close"].iloc[-1]
    if pd.isna(last_vwap) or last_vwap == 0:
        return 0.0
    return float((last_close - last_vwap) / last_vwap * 100)


def compute_volume_ratio(df: pd.DataFrame, period: int = 20) -> float:
    """Current volume vs N-period average. >2.0 = volume spike."""
    if "Volume" not in df.columns:
        return 1.0
    avg_vol = df["Volume"].rolling(period).mean()
    cur_vol = df["Volume"].iloc[-1]
    avg = avg_vol.iloc[-1]
    return float(cur_vol / avg) if avg > 0 else 1.0


def compute_obv_trend(df: pd.DataFrame, period: int = 20) -> str:
    """On-Balance Volume trend: 'up' | 'down' | 'flat'."""
    obv = (np.sign(df["Close"].diff()) * df.get("Volume", pd.Series(1, index=df.index))).cumsum()
    obv_ma = obv.rolling(period).mean()
    recent_slope = float(obv_ma.iloc[-1] - obv_ma.iloc[-period // 2]) if len(obv_ma) >= period else 0.0
    if recent_slope > 0:
        return "up"
    elif recent_slope < 0:
        return "down"
    return "flat"


# ─────────────────────────────────────────
# Support / Resistance
# ─────────────────────────────────────────

def compute_support_resistance(
    series: pd.Series,
    lookback: int = 50,
) -> Dict[str, float]:
    """Simple pivot-based support and resistance levels."""
    recent = series.tail(lookback)
    resistance = float(recent.max())
    support = float(recent.min())
    last = float(series.iloc[-1])
    midpoint = (resistance + support) / 2
    pct_to_resistance = (resistance - last) / last * 100 if last > 0 else 0.0
    pct_to_support = (last - support) / last * 100 if last > 0 else 0.0
    return {
        "resistance": round(resistance, 2),
        "support": round(support, 2),
        "midpoint": round(midpoint, 2),
        "last_price": round(last, 2),
        "pct_to_resistance": round(pct_to_resistance, 2),
        "pct_to_support": round(pct_to_support, 2),
        "position": "near_resistance" if pct_to_resistance < 2 else
                    "near_support" if pct_to_support < 2 else "mid_range",
    }


# ─────────────────────────────────────────
# Regime Detection
# ─────────────────────────────────────────

def detect_regime(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Classify current market regime from price data.
    Returns regime label + confidence + supporting metrics.
    """
    close = df["Close"]
    n = len(close)

    # Trend metrics
    sma_20 = close.rolling(20).mean().iloc[-1]
    sma_50 = close.rolling(50).mean().iloc[-1] if n >= 50 else sma_20
    sma_200 = close.rolling(200).mean().iloc[-1] if n >= 200 else sma_50
    last = close.iloc[-1]

    adx = compute_adx(df) if n >= 30 else 20.0
    rsi = compute_rsi(close)
    hv = compute_historical_volatility(close)
    mom_20 = compute_momentum(close, 20)

    # Regime scoring
    above_200ma = last > sma_200
    above_50ma = last > sma_50
    above_20ma = last > sma_20
    trending = adx > 25
    high_vol = hv > 25
    extreme_rsi = rsi > 70 or rsi < 30

    # Label logic
    if trending and above_200ma and above_50ma and mom_20 > 2:
        regime = "bull_trend"
        confidence = min(0.9, adx / 100)
    elif trending and not above_200ma and not above_50ma and mom_20 < -2:
        regime = "bear_trend"
        confidence = min(0.9, adx / 100)
    elif high_vol and not trending:
        regime = "high_volatility_choppy"
        confidence = 0.6
    elif not trending and not high_vol and not extreme_rsi:
        regime = "low_volatility_range"
        confidence = 0.65
    elif extreme_rsi or (not trending and abs(mom_20) < 1):
        regime = "mean_reverting"
        confidence = 0.55
    else:
        regime = "transitional"
        confidence = 0.4

    return {
        "regime": regime,
        "confidence": round(confidence, 3),
        "adx": round(adx, 2),
        "rsi": round(rsi, 2),
        "historical_volatility_pct": round(hv, 2),
        "momentum_20d_pct": round(mom_20, 2),
        "above_200ma": above_200ma,
        "above_50ma": above_50ma,
        "above_20ma": above_20ma,
        "last_price": round(float(last), 2),
        "sma_20": round(float(sma_20), 2),
        "sma_50": round(float(sma_50), 2),
        "sma_200": round(float(sma_200), 2),
    }


# ─────────────────────────────────────────
# Full scan for one ticker
# ─────────────────────────────────────────

def full_indicator_scan(df: pd.DataFrame, ticker: str = "") -> Dict[str, Any]:
    """
    Run all indicators on a ticker DataFrame.
    Returns a flat dict suitable for agent consumption.
    """
    close = df["Close"]
    n = len(df)

    result: Dict[str, Any] = {"ticker": ticker, "rows": n}

    if n < 10:
        result["error"] = "insufficient data"
        return result

    # Core indicators
    result["regime"] = detect_regime(df) if n >= 30 else {"regime": "unknown"}
    result["rsi_14"] = compute_rsi(close)
    result["macd"] = compute_macd(close) if n >= 30 else {}
    result["bollinger_20"] = compute_bollinger(close) if n >= 20 else {}
    result["atr_14"] = compute_atr(df) if n >= 20 else 0.0
    result["adx_14"] = compute_adx(df) if n >= 30 else 0.0
    result["historical_volatility_20d"] = compute_historical_volatility(close)
    result["momentum_20d_pct"] = compute_momentum(close, 20) if n >= 21 else 0.0
    result["momentum_5d_pct"] = compute_momentum(close, 5) if n >= 6 else 0.0
    result["volume_ratio_20d"] = compute_volume_ratio(df) if "Volume" in df.columns else 1.0
    result["vwap_deviation_pct"] = compute_vwap_deviation(df) if all(c in df.columns for c in ["High","Low","Volume"]) else 0.0
    result["support_resistance"] = compute_support_resistance(close) if n >= 10 else {}
    result["obv_trend"] = compute_obv_trend(df) if "Volume" in df.columns else "unknown"

    # Pattern flags
    result["patterns"] = _detect_patterns(df, result)

    return result


def _detect_patterns(df: pd.DataFrame, indicators: Dict[str, Any]) -> List[str]:
    """Identify notable pattern flags from computed indicators."""
    patterns = []
    rsi = indicators.get("rsi_14", 50)
    bb = indicators.get("bollinger_20", {})
    macd = indicators.get("macd", {})
    mom20 = indicators.get("momentum_20d_pct", 0)
    vol_ratio = indicators.get("volume_ratio_20d", 1.0)
    regime = indicators.get("regime", {}).get("regime", "")

    if rsi < 30:
        patterns.append("RSI_OVERSOLD")
    if rsi > 70:
        patterns.append("RSI_OVERBOUGHT")
    if bb.get("squeeze"):
        patterns.append("BOLLINGER_SQUEEZE")
    if bb.get("extended_high"):
        patterns.append("BOLLINGER_UPPER_TOUCH")
    if bb.get("extended_low"):
        patterns.append("BOLLINGER_LOWER_TOUCH")
    if macd.get("bullish_cross"):
        patterns.append("MACD_BULLISH_CROSS")
    if macd.get("bearish_cross"):
        patterns.append("MACD_BEARISH_CROSS")
    if abs(mom20) > 10:
        patterns.append("STRONG_MOMENTUM_20D")
    if vol_ratio > 2.0:
        patterns.append("VOLUME_SPIKE")
    if "bull_trend" in regime:
        patterns.append("UPTREND")
    if "bear_trend" in regime:
        patterns.append("DOWNTREND")
    if "mean_reverting" in regime:
        patterns.append("MEAN_REVERSION_SETUP")

    return patterns
