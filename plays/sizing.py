"""Position sizing utilities."""

from __future__ import annotations


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    max_fraction: float = 0.05,
) -> float:
    """
    Kelly criterion: f* = (p*b - q) / b
    Capped at max_fraction to stay conservative.
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.02
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    kelly = (win_rate * b - q) / b
    return min(max(kelly, 0.005), max_fraction)


def fixed_fraction(portfolio_value: float, risk_pct: float = 0.02) -> float:
    """Risk exactly risk_pct of portfolio."""
    return portfolio_value * risk_pct


def volatility_adjusted_size(
    portfolio_value: float,
    target_volatility: float,
    instrument_volatility: float,
    max_pct: float = 0.05,
) -> float:
    """Size so position vol contribution equals target_volatility."""
    if instrument_volatility <= 0:
        return portfolio_value * 0.02
    size = (target_volatility / instrument_volatility) * portfolio_value
    return min(size, portfolio_value * max_pct)


def compute_shares(capital: float, price: float, round_lots: bool = False) -> int:
    if price <= 0:
        return 0
    shares = capital / price
    if round_lots:
        rounded = int(shares / 100) * 100
        if rounded == 0:
            rounded = int(shares)
        shares = rounded
    return max(int(shares), 0)


def compute_contracts(capital: float, option_price: float, min_contracts: int = 1) -> int:
    """1 contract = 100 shares premium."""
    if option_price <= 0:
        return min_contracts
    contracts = int(capital / (option_price * 100))
    return max(contracts, min_contracts)
