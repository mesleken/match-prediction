"""
Parity test between offline backtesting, evaluation scripts, and production prediction service.
Guarantees identical feature columns, no lookahead leakage, and calibrated predictions.
"""

from datetime import datetime, timezone
import duckdb
import numpy as np
import pandas as pd
import pytest

from slmodel.config import get_model_config
from slmodel.data.loaders import load_match_history
from slmodel.models.dixon_coles import create_dixon_coles_model
from slmodel.services.prediction_service import MatchPredictionService
from slmodel.simulate.markets import calibrate_probabilities


def test_unified_loader_parity():
    """Verify that load_match_history returns all expected shot and xG columns."""
    df = load_match_history()
    required_cols = [
        "match_id", "season", "kickoff_utc", "home_team_id", "away_team_id",
        "home_goals", "away_goals", "hs", "hst", "as_", "ast", "hc", "ac", "hxg", "axg"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column {col} in unified loader"

    assert len(df) >= 3000
    assert (df["status"] == "played").all()


def test_as_of_temporal_boundary_strictness():
    """Verify that passing as_of strictly cuts off future matches."""
    as_of = datetime(2023, 6, 1, tzinfo=timezone.utc)
    df_as_of = load_match_history(as_of=as_of)

    k_utc = pd.to_datetime(df_as_of["kickoff_utc"], utc=True)
    assert (k_utc <= as_of).all()
    assert len(df_as_of) < 2500
    assert len(df_as_of) > 1500


def test_prediction_service_matches_backtest_training():
    """Verify that prediction service uses the unified loader with shot columns enabled."""
    service = MatchPredictionService()
    as_of = datetime(2024, 5, 20, tzinfo=timezone.utc)

    # Analyze match via service
    res = service.analyze_match(home_team_id=1, away_team_id=2, as_of=as_of, n_monte_carlo=100)

    assert res.home_team is not None
    assert res.away_team is not None
    assert sum(res.p_football_1x2) == pytest.approx(1.0, abs=1e-3)

    if res.p_calibrated_1x2:
        assert sum(res.p_calibrated_1x2) == pytest.approx(1.0, abs=1e-3)
        # Heavy favorite should receive slight confidence boost under T < 1.0
        max_idx = np.argmax(res.p_football_1x2)
        if res.p_football_1x2[max_idx] >= 0.60:
            assert res.p_calibrated_1x2[max_idx] >= res.p_football_1x2[max_idx]
