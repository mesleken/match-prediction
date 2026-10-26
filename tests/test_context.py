"""Unit tests for context features: saturating rest, congestion, and geography."""

from datetime import datetime, timezone
import pandas as pd
import pytest
from slmodel.features.context import haversine_km, saturating_rest, ContextFeatureExtractor


def test_saturating_rest_properties():
    """Verify non-linear fatigue recovery curve."""
    # <= 2 days rest gives zero excess recovery
    assert saturating_rest(1.0) == 0.0
    assert saturating_rest(2.0) == 0.0

    # 3 days vs 5 days vs 10 days
    r3 = saturating_rest(3.0)
    r5 = saturating_rest(5.0)
    r10 = saturating_rest(10.0)

    assert 0.0 < r3 < r5 < r10 < 1.0
    # Incremental gain from 2 to 4 days is larger than from 7 to 9 days
    gain_early = saturating_rest(4.0) - saturating_rest(2.0)
    gain_late = saturating_rest(9.0) - saturating_rest(7.0)
    assert gain_early > gain_late


def test_haversine_distance():
    """Verify distance calculation between Istanbul and Trabzon (~900 km)."""
    # Istanbul approx 41.0, 29.0, Trabzon approx 41.0, 39.7
    dist = haversine_km(41.0, 29.0, 41.0, 39.7)
    assert 800 < dist < 1000


def test_context_extractor_derby_and_shock():
    """Verify derby identification and altitude shock."""
    extractor = ContextFeatureExtractor()

    # Galatasaray (1) vs Fenerbahce (2) is Kıtalararası Derbi
    assert extractor.is_derby(1, 2) is True
    assert extractor.is_derby(2, 1) is True

    # Galatasaray (1) vs Sivasspor (9) is not a derby
    assert extractor.is_derby(1, 9) is False

    # Altitude shock: Sivas (1285m) hosting Galatasaray (75m)
    # altitude_diff should be > 1000m
    hist = pd.DataFrame([
        {
            "match_id": 1, "kickoff_utc": pd.Timestamp("2024-01-01 16:00:00", tz="UTC"),
            "home_team_id": 9, "away_team_id": 2, "status": "played"
        },
        {
            "match_id": 2, "kickoff_utc": pd.Timestamp("2024-01-02 16:00:00", tz="UTC"),
            "home_team_id": 1, "away_team_id": 3, "status": "played"
        },
    ])

    feats = extractor.extract_features_for_match(
        home_team_id=9,
        away_team_id=1,
        kickoff_utc=pd.Timestamp("2024-01-10 16:00:00", tz="UTC"),
        matches_history=hist,
    )

    assert feats["altitude_diff_m"] > 1000
    assert feats["is_derby"] == 0.0
    assert feats["home_rest_days"] == 9.0
    assert feats["away_rest_days"] == 8.0


def test_low_tempo_pace_damping_increases_draw_and_preserves_superiority():
    """
    Verify that low-tempo / sterile matchups (negative pace_dev_h + pace_dev_a)
    symmetrically damp expected goals, strictly preserve the superiority ratio (lam / mu),
    and organically increase P(Draw) and P(Under 2.5).
    """
    import numpy as np
    from slmodel.models.dixon_coles import DixonColesModel
    from slmodel.simulate.score_matrix import score_matrix
    from slmodel.simulate.markets import one_x_two, over_under

    model = DixonColesModel(low_tempo_damping_weight=0.09)
    model.is_fitted = True
    model.team_to_idx = {6: 0, 10: 1}
    model.team_ids = [6, 10]
    model.att = np.array([-0.10, -0.08], dtype=np.float64)
    model.def_ = np.array([0.05, 0.04], dtype=np.float64)
    model.mu0 = 0.20
    model.home_adv = 0.22
    model.rho = -0.08

    # Neutral pace vs sterile low-tempo pace (e.g. Konyaspor vs Alanyaspor)
    lam_0, mu_0, rho_0 = model.predict_lambdas(6, 10, context_features={"pace_dev_h": 0.0, "pace_dev_a": 0.0})
    lam_low, mu_low, rho_low = model.predict_lambdas(6, 10, context_features={"pace_dev_h": -0.75, "pace_dev_a": -0.55})

    # 1. Total expected goals must decrease
    assert (lam_low + mu_low) < (lam_0 + mu_0)
    # 2. Exact superiority ratio (lam / mu) must be preserved
    assert np.isclose(lam_low / mu_low, lam_0 / mu_0, atol=1e-12)

    # 3. Organic draw probability and Under 2.5 probability must increase
    M_0 = score_matrix(lam_0, mu_0, rho=rho_0)
    M_low = score_matrix(lam_low, mu_low, rho=rho_low)

    p_1x2_0 = one_x_two(M_0)
    p_1x2_low = one_x_two(M_low)
    ou_0 = over_under(M_0, line=2.5)
    ou_low = over_under(M_low, line=2.5)

    assert p_1x2_low[1] > p_1x2_0[1]  # P(Draw) increases organically
    assert ou_low["under"] > ou_0["under"]  # P(Under 2.5) increases organically

