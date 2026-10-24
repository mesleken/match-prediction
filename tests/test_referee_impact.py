"""Comprehensive tests for Layer G: Referee Officiating Impact, Whistle Tempo, and Booking Points."""

from datetime import datetime, timezone
import numpy as np
import pytest
from slmodel.models.referee_engine import RefereeEngine, LEAGUE_AVG_FOULS, LEAGUE_AVG_YELLOW, LEAGUE_AVG_RED
from slmodel.models.sidemodels import SideMarketsModel, predict_corners, predict_cards
from slmodel.features.context import ContextFeatureExtractor
from slmodel.evaluate.audit import SeasonSimulationAuditor, ErrorDiagnosticsEngine


def test_referee_engine_catalog_and_fallback():
    """Verify that RefereeEngine initializes with catalogue and handles fallbacks gracefully."""
    engine = RefereeEngine(db_path="data/warehouse.duckdb")
    all_refs = engine.get_all_active_referees()
    assert len(all_refs) >= 15

    # Known referee
    meler = engine.get_referee_profile("Halil Umut Meler")
    assert meler["name"] == "Halil Umut Meler"
    assert meler["matches_officiated"] >= 100
    assert meler["whistle_tempo_index"] < 0.0  # Fluid referee
    assert meler["card_strictness_mult"] < 1.0  # More lenient with cards

    # Case insensitive
    karaoglan = engine.get_referee_profile("atilla karaoğlan")
    assert "Atilla" in karaoglan["name"]
    assert karaoglan["whistle_tempo_index"] > 0.0  # Stops play more frequently

    # Fallback / Unknown referee
    unknown = engine.get_referee_profile("Bilinmeyen Hakem XYZ")
    assert unknown["whistle_tempo_index"] == 0.0
    assert unknown["card_strictness_mult"] == 1.0
    assert abs(unknown["avg_fouls"] - LEAGUE_AVG_FOULS) < 1e-4

    # None / Empty
    none_ref = engine.get_referee_profile(None)
    assert none_ref["whistle_tempo_index"] == 0.0


def test_referee_tempo_flow_adjustments():
    """Verify that whistle-heavy referees dampen xG & corners and increase draw probability."""
    engine = RefereeEngine(db_path="data/warehouse.duckdb")

    whistle_heavy = {
        "name": "Strict Whistle",
        "whistle_tempo_index": +0.65,
        "card_strictness_mult": 1.15,
        "home_card_bias": 0.0,
    }
    fluid_ref = {
        "name": "Fluid Game",
        "whistle_tempo_index": -0.65,
        "card_strictness_mult": 0.88,
        "home_card_bias": 0.0,
    }

    adj_heavy = engine.compute_flow_and_tempo_adjustments(whistle_heavy, att_diff=0.0)
    adj_fluid = engine.compute_flow_and_tempo_adjustments(fluid_ref, att_diff=0.0)

    # 1. Pace shift direction
    assert adj_heavy["delta_pace"] < 0.0  # Slows play down, dampens open-play xG
    assert adj_fluid["delta_pace"] > 0.0  # Fluid transitions boost open-play xG
    assert adj_heavy["delta_pace"] < adj_fluid["delta_pace"]

    # 2. Draw inflation
    assert adj_heavy["draw_inflation"] > 0.01  # Choppy matches have elevated draw variance
    assert adj_fluid["draw_inflation"] == 0.0   # Fluid matches do not inflate draws

    # 3. Corner multiplier
    assert adj_heavy["corner_multiplier"] < 1.0  # Fewer corners in low-tempo games
    assert adj_fluid["corner_multiplier"] > 1.0  # More corners in fast transitions


def test_booking_points_distribution_properties():
    """Verify mathematical properties of the booking points (Yellow=10, Red=25) distribution."""
    engine = RefereeEngine(db_path="data/warehouse.duckdb")

    strict_ref = {
        "name": "Strict Cards",
        "whistle_tempo_index": +0.40,
        "card_strictness_mult": 1.20,
        "red_card_mult": 1.35,
        "home_card_bias": 0.0,
        "avg_yellow": 5.2,
        "avg_red": 0.35,
    }

    lenient_ref = {
        "name": "Lenient Cards",
        "whistle_tempo_index": -0.30,
        "card_strictness_mult": 0.85,
        "red_card_mult": 0.70,
        "home_card_bias": 0.0,
        "avg_yellow": 3.4,
        "avg_red": 0.12,
    }

    dist_strict = engine.compute_booking_points_distribution(strict_ref, is_derby=True)
    dist_lenient = engine.compute_booking_points_distribution(lenient_ref, is_derby=False)

    # Expected booking points strictly higher for strict ref in derby
    assert dist_strict["expected_booking_points"] > dist_lenient["expected_booking_points"]
    assert dist_strict["expected_booking_points"] >= 50.0

    # Over/Under lines monotonicity: P(Over 35.5) >= P(Over 45.5) >= P(Over 55.5) >= P(Over 65.5)
    lines_s = dist_strict["booking_points_lines"]
    assert lines_s["over_35.5"] >= lines_s["over_45.5"] >= lines_s["over_55.5"] >= lines_s["over_65.5"]

    # Card 1X2 sum to 1.0
    c1x2 = dist_strict["card_1x2"]
    prob_sum = c1x2["1_home_more"] + c1x2["X_equal"] + c1x2["2_away_more"]
    assert abs(prob_sum - 1.0) < 1e-5


def test_sidemodels_referee_integration():
    """Verify that predict_corners and predict_cards correctly incorporate referee profiles."""
    side_model = SideMarketsModel()

    whistle_heavy_ref = {
        "name": "Direnç Tonusluoğlu",
        "whistle_tempo_index": +0.57,
        "card_strictness_mult": 1.18,
        "red_card_mult": 1.25,
        "home_card_bias": 0.0,
    }
    fluid_ref = {
        "name": "Halil Umut Meler",
        "whistle_tempo_index": -0.22,
        "card_strictness_mult": 0.82,
        "red_card_mult": 0.75,
        "home_card_bias": 0.0,
    }

    # Cards under whistle heavy strict referee vs fluid referee
    cards_heavy = predict_cards(
        lam_h=1.8, lam_a=1.2, is_derby=False, home_team_id=1, away_team_id=2,
        side_model=side_model, referee_profile=whistle_heavy_ref,
    )
    cards_fluid = predict_cards(
        lam_h=1.8, lam_a=1.2, is_derby=False, home_team_id=1, away_team_id=2,
        side_model=side_model, referee_profile=fluid_ref,
    )

    assert cards_heavy["expected_total"] > cards_fluid["expected_total"]
    assert cards_heavy["expected_booking_points"] > cards_fluid["expected_booking_points"]
    assert cards_heavy["lines"]["over_4.5"] > cards_fluid["lines"]["over_4.5"]

    # Corners under whistle heavy referee vs fluid referee
    corners_heavy = predict_corners(
        lam_h=1.8, lam_a=1.2, home_team_id=1, away_team_id=2,
        side_model=side_model, referee_profile=whistle_heavy_ref,
    )
    corners_fluid = predict_corners(
        lam_h=1.8, lam_a=1.2, home_team_id=1, away_team_id=2,
        side_model=side_model, referee_profile=fluid_ref,
    )

    assert corners_heavy["expected_total"] < corners_fluid["expected_total"]


def test_context_extractor_referee_features():
    """Verify that ContextFeatureExtractor correctly extracts referee features."""
    import duckdb
    con = duckdb.connect("data/warehouse.duckdb")
    matches_df = con.execute("SELECT * FROM matches WHERE status = 'played'").fetchdf()
    con.close()

    ctx_extractor = ContextFeatureExtractor(db_path="data/warehouse.duckdb")
    now_utc = datetime.now(timezone.utc)

    # With high whistle referee
    ctx_heavy = ctx_extractor.extract_features_for_match(
        home_team_id=1, away_team_id=2, kickoff_utc=now_utc,
        matches_history=matches_df,
        referee_name="Atilla Karaoğlan",
    )
    assert "ref_delta_h" in ctx_heavy
    assert "ref_whistle_index" in ctx_heavy
    assert ctx_heavy["ref_whistle_index"] > 0.0
    assert ctx_heavy["ref_delta_h"] < 0.0

    # Without referee (league average default)
    ctx_none = ctx_extractor.extract_features_for_match(
        home_team_id=1, away_team_id=2, kickoff_utc=now_utc,
        matches_history=matches_df,
        referee_name=None,
    )
    assert ctx_none["ref_delta_h"] == 0.0
    assert ctx_none["ref_draw_inf"] == 0.0
