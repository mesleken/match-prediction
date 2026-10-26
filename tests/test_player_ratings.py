"""Unit tests for Layer B (Player Ratings) & Layer C (Tactical Unit Cohesion / Synergy) simulation."""

import numpy as np
import pytest
from slmodel.models.player_ratings import PlayerRatingEngine


@pytest.fixture
def player_eng():
    return PlayerRatingEngine()


def test_default_lineup_benchmark_orthogonality(player_eng):
    """Verify that default benchmark starting 11 produces zero shift (strict orthogonality to Dixon-Coles)."""
    # Galatasaray vs Fenerbahce
    sim = player_eng.simulate_match_with_lineups(
        home_team_id=1,
        away_team_id=2,
        base_lam_h=1.90,
        base_lam_a=1.45,
        rho=-0.05,
    )

    shifts = sim["shifts"]
    assert np.isclose(shifts["delta_lam_h"], 0.0, atol=1e-6)
    assert np.isclose(shifts["delta_lam_a"], 0.0, atol=1e-6)
    assert np.isclose(shifts["delta_p_home"], 0.0, atol=1e-6)
    assert np.isclose(shifts["delta_p_draw"], 0.0, atol=1e-6)
    assert np.isclose(shifts["delta_p_away"], 0.0, atol=1e-6)

    assert np.isclose(sim["home_adjustment"]["delta_att"], 0.0, atol=1e-6)
    assert np.isclose(sim["home_adjustment"]["delta_def"], 0.0, atol=1e-6)
    assert np.isclose(sim["away_adjustment"]["delta_att"], 0.0, atol=1e-6)
    assert np.isclose(sim["away_adjustment"]["delta_def"], 0.0, atol=1e-6)


def test_star_striker_absence_decreases_expected_goals(player_eng):
    """Verify that missing an elite striker (e.g. Victor Osimhen) realistically reduces team xG and win%."""
    gs_starters = [p["player_id"] for p in player_eng.get_default_lineup(1)]
    gs_roster = player_eng.get_team_roster(1)

    osimhen = next(p for p in gs_roster if "Osimhen" in p["name"])
    bench_fwds = [p for p in gs_roster if p["position"] in ("CF", "RW", "LW") and p["player_id"] not in gs_starters]
    sub_fwd = bench_fwds[0]

    # Lineup with bench forward starting instead of Osimhen
    modified_lineup = [sub_fwd["player_id"] if pid == osimhen["player_id"] else pid for pid in gs_starters]

    sim = player_eng.simulate_match_with_lineups(
        home_team_id=1,
        away_team_id=2,
        home_player_ids=modified_lineup,
        base_lam_h=1.90,
        base_lam_a=1.45,
    )

    # Galatasaray expected goals must drop
    assert sim["adjusted"]["lam_h"] < sim["base"]["lam_h"]
    assert sim["shifts"]["delta_lam_h"] < -0.08
    # Galatasaray win probability must drop
    assert sim["adjusted"]["p_home"] < sim["base"]["p_home"]
    assert sim["shifts"]["delta_p_home"] < -0.02
    # Missing starters reported
    assert any(p["name"] == osimhen["name"] for p in sim["home_adjustment"]["missing_starters"])


def test_key_defender_and_gk_absence_increases_opponent_goals(player_eng):
    """Verify that missing top defender or goalkeeper increases the opponent's expected goals."""
    gs_starters = [p["player_id"] for p in player_eng.get_default_lineup(1)]
    gs_roster = player_eng.get_team_roster(1)

    sanchez = next(p for p in gs_roster if "Sanchez" in p["name"])
    bench_defs = [p for p in gs_roster if p["position"] in ("CB", "LB", "RB") and p["player_id"] not in gs_starters]
    sub_def = next(p for p in bench_defs if p["def_rating"] < sanchez["def_rating"])

    modified_lineup = [sub_def["player_id"] if pid == sanchez["player_id"] else pid for pid in gs_starters]

    sim = player_eng.simulate_match_with_lineups(
        home_team_id=1,
        away_team_id=2,
        home_player_ids=modified_lineup,
        base_lam_h=1.90,
        base_lam_a=1.45,
    )

    # Opponent (Fenerbahce) expected goals must increase because Galatasaray defense is weaker
    assert sim["adjusted"]["lam_a"] > sim["base"]["lam_a"]
    assert sim["shifts"]["delta_lam_a"] > 0.03
    # Fenerbahce win probability increases
    assert sim["adjusted"]["p_away"] > sim["base"]["p_away"]


def test_tactical_synergy_penalty_when_pair_broken(player_eng):
    """Verify that breaking an established central defensive tandem penalizes unit cohesion."""
    gs_starters = [p["player_id"] for p in player_eng.get_default_lineup(1)]
    gs_roster = player_eng.get_team_roster(1)

    sanchez = next(p for p in gs_roster if "Sanchez" in p["name"])
    bench_defs = [p for p in gs_roster if p["position"] in ("CB", "LB", "RB") and p["player_id"] not in gs_starters]
    sub_def = bench_defs[0]
    bench_lineup = [sub_def["player_id"] if pid == sanchez["player_id"] else pid for pid in gs_starters]

    adj = player_eng.compute_lineup_adjustment(1, bench_lineup)
    # Unit cohesion delta must be negative due to broken tandem
    assert adj["delta_c_def"] < 0.0


def test_marginal_player_importance_ranking(player_eng):
    """Verify that marginal importance ranking produces positive losses when key starters are absent."""
    importance = player_eng.evaluate_player_importance(team_id=1, base_lam_for=1.85, base_lam_against=1.15)
    assert len(importance) == 11
    # Top players should have significant marginal goal difference impact
    top_player = importance[0]
    assert top_player["marginal_drop_win_pct"] > 3.0
    assert top_player["marginal_goal_diff_loss"] > 0.10


def test_all_18_teams_valid_tactical_formation_and_positions(player_eng):
    """
    Verify that every team has a valid 11-player tactical formation (1 GK, 1 RB, 2 CB, 1 LB,
    1 DM, 1 CM, 1 AM, 1 RW, 1 LW, 1 CF), authentic player positions, and starters selected
    by who started the most matches in each position.
    """
    import json
    from collections import Counter
    from pathlib import Path

    expected_formation = Counter({
        "GK": 1, "RB": 1, "CB": 2, "LB": 1,
        "DM": 1, "CM": 1, "AM": 1,
        "RW": 1, "LW": 1, "CF": 1,
    })

    squads_json = json.loads(Path("config/squads_2627.json").read_text(encoding="utf-8"))
    assert len(player_eng._cache_rosters) == 18

    for tid, roster in player_eng._cache_rosters.items():
        starters = player_eng.get_default_lineup(tid)
        assert len(starters) == 11, f"Team {tid} does not have 11 starters"
        pos_counts = Counter(p["position"] for p in starters)
        assert pos_counts == expected_formation, f"Team {tid} has invalid formation: {pos_counts}"

        # Verify starter merit by starts within each position in squads_2627.json
        team_json_players = squads_json[str(tid)]["players"]
        for pos, count in expected_formation.items():
            pos_players = [p for p in team_json_players if p["pos"] == pos]
            pos_starters = [p for p in pos_players if p["starter"]]
            pos_bench = [p for p in pos_players if not p["starter"]]
            assert len(pos_starters) == count
            if pos_starters and pos_bench:
                min_starter_merit = min((p["starts"], p["minutes"], p["matches"], p["val"]) for p in pos_starters)
                max_bench_merit = max((p["starts"], p["minutes"], p["matches"], p["val"]) for p in pos_bench)
                assert min_starter_merit >= max_bench_merit, (
                    f"Team {tid} pos {pos}: bench player has more starts/merit than starter!"
                )

    # Specific player position and starter checks (e.g. Beşiktaş Vlahovic vs Hyeon-Gyu Oh)
    bjk_roster = {p["name"]: p for p in player_eng.get_team_roster(3)}
    assert bjk_roster["Dusan Vlahovic"]["position"] == "CF"
    assert bjk_roster["Hyeon-Gyu Oh"]["position"] == "CF"
    assert bjk_roster["Vaclav Cerny"]["position"] == "RW"
    assert bjk_roster["Vaclav Cerny"]["is_starter"] is True
    assert bjk_roster["Leandro Trossard"]["position"] == "LW"
    assert bjk_roster["Leandro Trossard"]["is_starter"] is True
    assert bjk_roster["Junior Olaitan"]["position"] == "AM"

    # Vlahovic and Hyeon-Gyu Oh must never both be starters simultaneously
    bjk_starter_names = {p["name"] for p in player_eng.get_default_lineup(3)}
    assert not ("Dusan Vlahovic" in bjk_starter_names and "Hyeon-Gyu Oh" in bjk_starter_names)
    assert "Hyeon-Gyu Oh" in bjk_starter_names  # 9 starts vs Vlahovic 7 starts

