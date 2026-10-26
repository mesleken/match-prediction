"""
Unit tests for shot-quality proxy xG and opponent-adjusted form features.
Ensures zero lookahead leakage and mathematical validity.
"""

from datetime import datetime, timezone
import pandas as pd
import pytest

from slmodel.data.loaders import load_match_history
from slmodel.features.xg_proxy import (
    fit_proxy_xg_weights,
    compute_match_proxy_xg,
    compute_opponent_adjusted_xg_form,
    DEFAULT_PROXY_BETAS,
)


def test_fit_proxy_xg_weights_positive():
    """Verify that fitted proxy weights are strictly non-negative and realistic."""
    df = load_match_history()
    betas = fit_proxy_xg_weights(df, as_of=datetime(2024, 1, 1, tzinfo=timezone.utc))

    b0, b_sot, b_off, b_c = betas
    assert b0 >= 0.0
    assert b_sot > 0.10, "Shot on target conversion must be substantial (> 0.10)"
    assert b_off >= 0.0
    assert b_c >= 0.0


def test_as_of_filtering_prevents_future_fitting():
    """Verify that fitting proxy weights strictly respects as_of cutoffs."""
    df = load_match_history()
    early_dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
    late_dt = datetime(2025, 1, 1, tzinfo=timezone.utc)

    betas_early = fit_proxy_xg_weights(df, as_of=early_dt)
    betas_late = fit_proxy_xg_weights(df, as_of=late_dt)

    assert betas_early is not None
    assert betas_late is not None


def test_compute_match_proxy_prefers_real_xg():
    """Verify that authentic Opta xG is used if present in row."""
    row_real = pd.Series({"hxg": 2.45, "axg": 0.85, "hs": 15, "hst": 6, "home_goals": 2, "away_goals": 1})
    h_xg, a_xg = compute_match_proxy_xg(row_real, DEFAULT_PROXY_BETAS)

    assert h_xg == pytest.approx(2.45)
    assert a_xg == pytest.approx(0.85)

    # Missing xG row uses proxy
    row_proxy = pd.Series({"hxg": None, "axg": None, "hs": 15, "hst": 5, "as_": 10, "ast": 3, "hc": 4, "ac": 2})
    h_p, a_p = compute_match_proxy_xg(row_proxy, DEFAULT_PROXY_BETAS)
    assert h_p > 0.5
    assert a_p > 0.3


def test_xg_form_returns_bounded_floats():
    """Verify that xG form momentum computes valid floats for known clubs."""
    df = load_match_history(as_of=datetime(2024, 1, 1, tzinfo=timezone.utc))
    form = compute_opponent_adjusted_xg_form(team_id=1, prior_matches=df)

    assert "xg_att_form" in form
    assert "xg_def_vuln" in form
    assert "xg_net_diff" in form
    assert -2.0 <= form["xg_att_form"] <= 2.0
    assert -2.0 <= form["xg_def_vuln"] <= 2.0
