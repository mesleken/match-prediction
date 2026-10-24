"""Unit tests for player match performance generation, statistics aggregation, and Bayesian rating calculations."""

import duckdb
import numpy as np
import pytest
from slmodel.data.player_performance import (
    generate_and_sync_player_match_stats,
    compute_player_season_stats_and_ratings,
    get_player_season_leaderboards,
)


def test_player_match_stats_generation_and_consistency():
    """Verify player match stats are generated and match team goals and cards."""
    res = generate_and_sync_player_match_stats(season="2627")
    assert res["matches_processed"] > 0
    assert res["records_generated"] > 0

    con = duckdb.connect("data/warehouse.duckdb")
    try:
        # Check that player goals in match 3089 (Galatasaray vs Corum FK: 2-2) sum to team goals
        gs_goals = con.execute("SELECT SUM(goals) FROM player_match_stats WHERE match_id = 3089 AND team_id = 1").fetchone()[0]
        corum_goals = con.execute("SELECT SUM(goals) FROM player_match_stats WHERE match_id = 3089 AND team_id = 38").fetchone()[0]
        assert gs_goals == 2
        assert corum_goals == 2

        # Check clean sheet logic: when a team concedes 0 goals, the playing GK and DEF get clean_sheet = 1
        # In match 3098 (Konyaspor 0 - 4 Galatasaray), Galatasaray kept a clean sheet
        gs_cs = con.execute("""
            SELECT SUM(clean_sheet) FROM player_match_stats
            WHERE match_id = 3098 AND team_id = 1
        """).fetchone()[0]
        assert gs_cs >= 4  # Muslera and defenders
    finally:
        con.close()


def test_player_season_stats_and_bayesian_ratings():
    """Verify aggregated season statistics and Bayesian regularized ratings."""
    df = compute_player_season_stats_and_ratings(season="2627", update_players_table=True)
    assert not df.empty
    assert len(df) >= 240  # 18 teams * players

    # Ratings sanity checks
    assert (df["computed_overall_rating"] >= 68.0).all()
    assert (df["computed_overall_rating"] <= 88.0).all()
    assert (df["computed_att_rating"] >= 0.0).all()
    assert (df["computed_def_rating"] >= 0.0).all()

    # Forwards should have higher att_rating than GKs
    top_cfs = df[df["position"] == "CF"]
    gks = df[df["position"] == "GK"]
    assert top_cfs["computed_att_rating"].mean() > gks["computed_att_rating"].mean()

    # GKs and CBs should have higher def_rating than CFs
    cbs = df[df["position"] == "CB"]
    assert cbs["computed_def_rating"].mean() > top_cfs["computed_def_rating"].mean()


def test_player_leaderboards():
    """Verify leaderboard extraction for goals, assists, goalkeepers, and top rated."""
    lbs = get_player_season_leaderboards(season="2627")
    assert not lbs["top_scorers"].empty
    assert not lbs["top_assisters"].empty
    assert not lbs["best_goalkeepers"].empty
    assert not lbs["top_rated"].empty

    # Top scorer has goals >= 5
    top_scorer = lbs["top_scorers"].iloc[0]
    assert top_scorer["Gol"] >= 5

    # Best GK has saves >= 20
    best_gk = lbs["best_goalkeepers"].iloc[0]
    assert best_gk["Kurtarış"] >= 20
