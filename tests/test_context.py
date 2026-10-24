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
