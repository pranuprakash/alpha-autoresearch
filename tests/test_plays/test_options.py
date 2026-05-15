"""Tests for plays/options.py — Black-Scholes and strike selection."""

import pytest
from plays.options import black_scholes_greeks, select_strike_by_delta


class TestBlackScholesGreeks:
    def test_call_delta_bounded(self):
        g = black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call")
        assert 0 < g["delta"] < 1

    def test_put_delta_negative(self):
        g = black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="put")
        assert -1 < g["delta"] < 0

    def test_atm_call_delta_near_half(self):
        g = black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call")
        assert 0.45 < g["delta"] < 0.65

    def test_deep_itm_call_delta_near_one(self):
        g = black_scholes_greeks(S=200, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call")
        assert g["delta"] > 0.90

    def test_theta_negative_for_long_call(self):
        g = black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call")
        assert g["theta"] < 0

    def test_vega_positive(self):
        g = black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call")
        assert g["vega"] > 0

    def test_gamma_positive(self):
        g = black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call")
        assert g["gamma"] > 0

    def test_zero_time_returns_fallback(self):
        g = black_scholes_greeks(S=100, K=100, T=0, r=0.05, sigma=0.20, option_type="call")
        assert "delta" in g

    def test_bs_price_non_negative(self):
        g = black_scholes_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call")
        assert g["bs_price"] > 0

    def test_put_call_parity_approx(self):
        S, K, T, r, sig = 100, 100, 0.25, 0.05, 0.20
        call = black_scholes_greeks(S, K, T, r, sig, "call")["bs_price"]
        put = black_scholes_greeks(S, K, T, r, sig, "put")["bs_price"]
        # C - P ≈ S - K*exp(-r*T)
        import math
        parity = S - K * math.exp(-r * T)
        assert abs((call - put) - parity) < 0.10


class TestSelectStrikeByDelta:
    def _make_chain(self):
        import pandas as pd
        return pd.DataFrame({
            "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
            "bid":    [11.0, 7.5,  4.5,   2.2,   0.9],
            "ask":    [11.5, 8.0,  5.0,   2.7,   1.1],
            "lastPrice": [11.2, 7.8, 4.7, 2.5, 1.0],
            "impliedVolatility": [0.25, 0.25, 0.25, 0.25, 0.25],
            "openInterest": [100, 200, 500, 300, 150],
            "volume": [50, 100, 250, 120, 60],
        })

    def test_returns_dict(self):
        chain = self._make_chain()
        result = select_strike_by_delta(chain, 0.40, "call", 100.0, "2026-06-20")
        assert result is not None
        assert "strike" in result
        assert "mid_price" in result
        assert "delta" in result

    def test_mid_price_positive(self):
        chain = self._make_chain()
        result = select_strike_by_delta(chain, 0.40, "call", 100.0, "2026-06-20")
        assert result["mid_price"] > 0

    def test_none_on_empty_chain(self):
        import pandas as pd
        result = select_strike_by_delta(pd.DataFrame(), 0.40, "call", 100.0, "2026-06-20")
        assert result is None

    def test_put_delta_negative(self):
        chain = self._make_chain()
        result = select_strike_by_delta(chain, 0.40, "put", 100.0, "2026-06-20")
        assert result is not None
        assert result["delta"] < 0
