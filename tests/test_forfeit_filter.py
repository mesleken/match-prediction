"""Tests to verify that forfeit matches (earthquake forfeits) are strictly excluded from model fitting and backtests."""

import duckdb
import numpy as np
import pandas as pd
import pytest
from slmodel.models.dixon_coles import DixonColesModel
from slmodel.evaluate.backtest import WalkForwardBacktest


def test_warehouse_forfeit_status():
    """Verify that 2022/23 earthquake forfeits are marked status='forfeit' in DuckDB."""
    con = duckdb.connect("data/warehouse.duckdb")
    forfeits = con.execute("SELECT COUNT(*) FROM matches WHERE status = 'forfeit'").fetchone()[0]
    played = con.execute("SELECT COUNT(*) FROM matches WHERE status = 'played'").fetchone()[0]
    con.close()

    assert forfeits >= 28, f"Expected at least 28 earthquake forfeit matches, found {forfeits}"
    assert played >= 3000, f"Expected at least 3000 played matches, found {played}"


def test_backtest_data_loader_excludes_forfeits():
    """Verify that WalkForwardBacktest data loader strictly selects status='played'."""
    bt = WalkForwardBacktest(db_path="data/warehouse.duckdb")
    matches_df, _ = bt.load_data()

    assert "forfeit" not in matches_df["status"].values
    assert (matches_df["status"] == "played").all()
    assert matches_df["home_goals"].notna().all()
    assert matches_df["away_goals"].notna().all()


def test_dixon_coles_fit_ignores_forfeits():
    """Verify that DixonColesModel.fit filters out any forfeit rows automatically."""
    dates = pd.date_range("2023-01-01", periods=30, freq="7D")
    base_data = []
    for i in range(25):
        base_data.append({
            "match_id": i + 1,
            "season": "2223",
            "kickoff_utc": dates[i],
            "home_team_id": 1 if i % 2 == 0 else 2,
            "away_team_id": 2 if i % 2 == 0 else 1,
            "home_goals": 2,
            "away_goals": 1,
            "status": "played",
        })

    # Add 5 fake forfeit matches with 3-0 results
    for i in range(25, 30):
        base_data.append({
            "match_id": i + 1,
            "season": "2223",
            "kickoff_utc": dates[i],
            "home_team_id": 1,
            "away_team_id": 2,
            "home_goals": 3,
            "away_goals": 0,
            "status": "forfeit",
        })

    df = pd.DataFrame(base_data)

    model1 = DixonColesModel()
    model1.fit(df)

    # Clean dataset without the forfeit rows at all
    df_clean = df[df["status"] == "played"].copy()
    model2 = DixonColesModel()
    model2.fit(df_clean)

    # Model parameters should be bit-identical because forfeit matches were ignored
    np.testing.assert_allclose(model1.att, model2.att, rtol=1e-10)
    np.testing.assert_allclose(model1.def_, model2.def_, rtol=1e-10)
    assert np.isclose(model1.mu0, model2.mu0, atol=1e-10)
    assert np.isclose(model1.home_adv, model2.home_adv, atol=1e-10)
