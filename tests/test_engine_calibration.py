"""
Unit and regression tests for calibrated MatchEngine moments against Süper Lig warehouse reality.
Ensures match engine does not drift away from empirical statistics.
"""

import numpy as np
import pytest
from slmodel.simulate.match_engine import MatchEngine
from slmodel.simulate.params import DEFAULT_ENGINE_PARAMS


def test_engine_moments_within_warehouse_tolerances():
    """Verify that 500 simulated matches stay within empirical bounds from 3,129 real matches."""
    rng = np.random.RandomState(42)
    engine = MatchEngine(rng=rng, params=DEFAULT_ENGINE_PARAMS)

    goals = []
    shots = []
    corners = []
    yellows = []
    xgs = []
    over25 = []
    btts = []

    for _ in range(500):
        tl = engine.simulate_match(home_team_id=1, away_team_id=2)
        gh = tl.score_h
        ga = tl.score_a
        goals.append(gh + ga)
        shots.append(tl.stats.shots_h + tl.stats.shots_a)
        corners.append(tl.stats.corners_h + tl.stats.corners_a)
        yellows.append(tl.stats.yellow_cards_h + tl.stats.yellow_cards_a)
        xgs.append(tl.stats.xg_h + tl.stats.xg_a)
        over25.append(1.0 if (gh + ga) > 2.5 else 0.0)
        btts.append(1.0 if (gh > 0 and ga > 0) else 0.0)

    # Warehouse reality (3,129 matches): Goals: 2.825, Shots: 24.35, Corners: 9.61, Yellows: 4.54, Goal/Shot Conversion: 0.116
    mean_goals = float(np.mean(goals))
    mean_shots = float(np.mean(shots))
    mean_corners = float(np.mean(corners))
    mean_yellows = float(np.mean(yellows))
    shot_conversion = mean_goals / mean_shots
    p_over25 = float(np.mean(over25))
    p_btts = float(np.mean(btts))

    assert 2.30 <= mean_goals <= 3.20, f"Simulated goals {mean_goals} out of tolerance [2.30, 3.20]"
    assert 18.0 <= mean_shots <= 28.0, f"Simulated shots {mean_shots} out of tolerance [18.0, 28.0]"
    assert 7.5 <= mean_corners <= 11.5, f"Simulated corners {mean_corners} out of tolerance [7.5, 11.5]"
    assert 3.0 <= mean_yellows <= 6.0, f"Simulated yellows {mean_yellows} out of tolerance [3.0, 6.0]"
    assert 0.095 <= shot_conversion <= 0.145, f"Simulated shot conversion {shot_conversion:.3f} out of tolerance [0.095, 0.145]"
    assert 0.42 <= p_over25 <= 0.62, f"Simulated Over 2.5 {p_over25} out of tolerance [0.42, 0.62]"
    assert 0.45 <= p_btts <= 0.68, f"Simulated BTTS {p_btts} out of tolerance [0.45, 0.68]"


def test_dixon_coles_lambda_coupling():
    """Verify that specifying Dixon-Coles lambda and mu directly steers match expected goals and 1X2."""
    rng = np.random.RandomState(123)
    engine = MatchEngine(rng=rng, params=DEFAULT_ENGINE_PARAMS)

    # Scenario 1: Heavy Home Favorite (lambda_h=2.60, mu_a=0.75)
    home_fav_h_xg = []
    home_fav_a_xg = []
    home_fav_wins = 0

    for _ in range(100):
        tl = engine.simulate_match(home_team_id=1, away_team_id=2, lam_h=2.60, lam_a=0.75)
        home_fav_h_xg.append(tl.stats.xg_h)
        home_fav_a_xg.append(tl.stats.xg_a)
        if tl.score_h > tl.score_a:
            home_fav_wins += 1

    assert np.mean(home_fav_h_xg) > 1.6 * np.mean(home_fav_a_xg)
    assert home_fav_wins >= 50, f"Home favorite should win majority of matches (got {home_fav_wins}/100)"

    # Scenario 2: Heavy Away Favorite (lambda_h=0.80, mu_a=2.30)
    away_fav_h_xg = []
    away_fav_a_xg = []
    away_fav_wins = 0

    for _ in range(100):
        tl = engine.simulate_match(home_team_id=1, away_team_id=2, lam_h=0.80, lam_a=2.30)
        away_fav_h_xg.append(tl.stats.xg_h)
        away_fav_a_xg.append(tl.stats.xg_a)
        if tl.score_a > tl.score_h:
            away_fav_wins += 1

    assert np.mean(away_fav_a_xg) > 1.6 * np.mean(away_fav_h_xg)
    assert away_fav_wins >= 50, f"Away favorite should win majority of matches (got {away_fav_wins}/100)"
