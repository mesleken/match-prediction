"""
Unit tests for the event-based match simulation engine and Monte Carlo scenario generator.
"""

import numpy as np
import pytest

from slmodel.simulate.state import MatchState, PitchZone, EventType
from slmodel.simulate.tactics import TacticalProfile, adjust_tactics_for_game_state
from slmodel.simulate.shot_model import resolve_shot_attempt
from slmodel.simulate.event_generator import EventGenerator
from slmodel.simulate.match_engine import MatchEngine
from slmodel.simulate.monte_carlo import run_monte_carlo_simulation


def test_tactics_game_state_adaptation():
    """Verify that tactical profiles dynamically adapt to trailing/leading scorelines and red cards."""
    base = TacticalProfile(tempo=1.0, line_height=0.5, pressing_intensity=0.5, risk_appetite=0.5)

    # Trailing late in match (minute 80, -1 deficit) -> higher risk, higher line
    trailing = adjust_tactics_for_game_state(base, score_diff=-1, minute=80)
    assert trailing.risk_appetite > base.risk_appetite
    assert trailing.line_height > base.line_height

    # Leading late in match (minute 80, +1 lead) -> lower line, lower risk
    leading = adjust_tactics_for_game_state(base, score_diff=1, minute=80)
    assert leading.risk_appetite < base.risk_appetite
    assert leading.line_height < base.line_height

    # Red card -> conservative block
    red_carded = adjust_tactics_for_game_state(base, score_diff=0, minute=30, own_red_cards=1)
    assert red_carded.line_height < base.line_height
    assert red_carded.pressing_intensity < base.pressing_intensity


def test_shot_model_spatial_xg_distribution():
    """Verify that shots inside penalty box yield higher xG than long-range shots."""
    rng = np.random.RandomState(42)
    box_xgs = [resolve_shot_attempt(PitchZone.PENALTY_BOX, rng=rng).xg for _ in range(50)]
    dist_xgs = [resolve_shot_attempt(PitchZone.ATTACKING_THIRD, rng=rng).xg for _ in range(50)]

    assert np.mean(box_xgs) > np.mean(dist_xgs)
    assert min(box_xgs) >= 0.01
    assert max(box_xgs) <= 0.95


def test_single_match_simulation_timeline():
    """Verify full 90-minute match simulation generates realistic stats and chronologically ordered events."""
    engine = MatchEngine(rng=np.random.RandomState(123))
    timeline = engine.simulate_match(
        home_team_id=1,
        away_team_id=2,
        att_h=0.25,
        def_h=0.25,
        att_a=0.15,
        def_a=0.15,
        seed=123,
    )

    assert timeline.score_h >= 0
    assert timeline.score_a >= 0
    assert timeline.ht_score_h >= 0
    assert timeline.ht_score_a >= 0
    assert len(timeline.events) >= 20

    # Events must be chronologically ordered
    minutes = [ev.minute for ev in timeline.events]
    assert minutes == sorted(minutes)

    # Full time event must be present
    assert timeline.events[-1].event_type == EventType.FULL_TIME
    assert timeline.stats.shots_h >= timeline.stats.shots_on_target_h
    assert timeline.stats.shots_a >= timeline.stats.shots_on_target_a


def test_monte_carlo_joint_probability_matrix():
    """Verify Monte Carlo simulation aggregates normalized joint score matrix and probabilities."""
    res = run_monte_carlo_simulation(
        home_team_id=1,
        away_team_id=2,
        att_h=0.25,
        def_h=0.20,
        att_a=0.15,
        def_a=0.20,
        n_simulations=100,
        seed=999,
    )

    # 1X2 probabilities must sum to 1.0
    assert np.isclose(res.p_home + res.p_draw + res.p_away, 1.0, atol=1e-3)
    # Score matrix must sum to 1.0
    assert np.isclose(float(np.sum(res.score_matrix)), 1.0, atol=1e-3)

    assert 0.0 <= res.p_over_25 <= 1.0
    assert 0.0 <= res.p_btts_yes <= 1.0
    assert len(res.top_scores) == 5
    assert res.mean_corners_tot > 0
    assert res.mean_cards_tot >= 0
