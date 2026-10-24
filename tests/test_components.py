"""Unit tests for 4D Marginally Coherent HT/FT, Empirical Bayes side models, Laplace/Bootstrap uncertainty, and cryptographic prediction log."""

import duckdb
import numpy as np
import pandas as pd
import pytest
from slmodel.models.dixon_coles import DixonColesModel
from slmodel.models.htft import predict_ht_ft, predict_half_markets, OUTCOMES_9
from slmodel.models.sidemodels import SideMarketsModel, predict_corners, predict_cards
from slmodel.simulate.score_matrix import score_matrix
from slmodel.simulate.markets import one_x_two
from slmodel.simulate.bootstrap import bayesian_bootstrap_prediction, laplace_posterior_prediction
from slmodel.tracking.prediction_log import PredictionLogger, verify_prediction_log
from slmodel.evaluate.season_stats import compute_season_statistics
from slmodel.ingest.sync import sync_latest_data, get_last_sync_info


def test_htft_exact_marginal_coherence_and_half_markets():
    """
    Verify that 4D HT/FT probabilities sum to 1.0 AND their full-time marginals
    match one_x_two(M_ft) to machine precision (< 1e-12).
    """
    M_ft = score_matrix(1.75, 1.15, -0.07)
    p_ft = one_x_two(M_ft)
    probs = predict_ht_ft(lam_h=1.75, lam_a=1.15, rho=-0.07, M_ft=M_ft)

    assert set(probs.keys()) == set(OUTCOMES_9)
    assert np.isclose(sum(probs.values()), 1.0, atol=1e-12)
    assert all(p >= 0.0 for p in probs.values())

    # Exact marginal coherence check
    ft_home_from_htft = probs["1/1"] + probs["X/1"] + probs["2/1"]
    ft_draw_from_htft = probs["1/X"] + probs["X/X"] + probs["2/X"]
    ft_away_from_htft = probs["1/2"] + probs["X/2"] + probs["2/2"]

    assert np.isclose(ft_home_from_htft, p_ft[0], atol=1e-12)
    assert np.isclose(ft_draw_from_htft, p_ft[1], atol=1e-12)
    assert np.isclose(ft_away_from_htft, p_ft[2], atol=1e-12)

    # Half markets check
    hm = predict_half_markets(lam_h=1.75, lam_a=1.15, rho=-0.07, M_ft=M_ft)
    assert np.isclose(sum(hm["ht_1x2"].values()), 1.0, atol=1e-12)
    assert np.isclose(sum(hm["2h_1x2"].values()), 1.0, atol=1e-12)
    assert np.isclose(sum(hm["highest_scoring_half"].values()), 1.0, atol=1e-12)
    # 2nd half has more goals than 1st half on average (pi ~ 0.44 < 0.56)
    assert hm["highest_scoring_half"]["2H"] > hm["highest_scoring_half"]["1H"]


def test_sidemodels_empirical_bayes_and_bivariate_corners():
    """Verify that corners and cards Bivariate Negative Binomial models produce coherent outputs."""
    M_ft = score_matrix(1.8, 0.9, -0.05)
    corners = predict_corners(lam_h=1.8, lam_a=0.9, M_ft=M_ft)
    assert 7.0 < corners["expected_total"] < 13.0
    assert corners["expected_home"] > corners["expected_away"]
    assert np.isclose(corners["expected_home"] + corners["expected_away"], corners["expected_total"], atol=1e-7)
    assert "over_9.5" in corners["lines"]
    assert np.isclose(corners["lines"]["over_9.5"] + corners["lines"]["under_9.5"], 1.0)
    assert np.isclose(sum(corners["corner_1x2"].values()), 1.0, atol=1e-7)
    assert corners["corner_1x2"]["1"] > corners["corner_1x2"]["2"]

    cards_norm = predict_cards(lam_h=1.5, lam_a=1.2, is_derby=False, M_ft=M_ft)
    cards_derby = predict_cards(lam_h=1.5, lam_a=1.2, is_derby=True, M_ft=M_ft)
    assert cards_derby["expected_total"] > cards_norm["expected_total"]
    assert cards_derby["p_red_card"] > cards_norm["p_red_card"] > 0.0
    assert cards_derby["expected_booking_points"] > cards_norm["expected_booking_points"] > 20.0
    assert np.isclose(cards_norm["lines"]["over_4.5"] + cards_norm["lines"]["under_4.5"], 1.0)


def test_laplace_and_bayesian_bootstrap_intervals():
    """Verify both analytical Laplace posterior sampling and non-parametric Bayesian bootstrap."""
    dates = pd.date_range("2023-01-01", periods=30, freq="7D")
    records = []
    for i in range(30):
        records.append({
            "match_id": i + 1,
            "season": "2223",
            "kickoff_utc": dates[i],
            "home_team_id": 1 if i % 2 == 0 else 2,
            "away_team_id": 2 if i % 2 == 0 else 1,
            "home_goals": 2 if i % 2 == 0 else 1,
            "away_goals": 1 if i % 2 == 0 else 0,
            "status": "played",
        })
    df = pd.DataFrame(records)

    model = DixonColesModel()
    model.fit(df)
    lap = laplace_posterior_prediction(model, home_team_id=1, away_team_id=2, n_samples=200, seed=42)
    assert np.isclose(lap["M_mean"].sum(), 1.0, atol=1e-7)
    for k in ["p_home", "p_draw", "p_away", "p_over25", "p_btts"]:
        p5, p50, p95 = lap["credible_intervals"][k]
        assert 0.0 <= p5 <= p50 <= p95 <= 1.0
        assert lap["posterior_std"][k] > 0.0

    res = bayesian_bootstrap_prediction(df, home_team_id=1, away_team_id=2, n_bootstraps=15, seed=42)
    assert np.isclose(res["M_mean"].sum(), 1.0, atol=1e-7)
    ci = res["credible_intervals"]
    for k in ["p_home", "p_draw", "p_away", "p_over25", "p_btts"]:
        p5, p50, p95 = ci[k]
        assert 0.0 <= p5 <= p50 <= p95 <= 1.0


def test_prediction_logger_hash_chain_and_tamper_detection(tmp_path):
    """Verify append-only hash chain and that any tampering is detected."""
    test_db = str(tmp_path / "test_warehouse.duckdb")
    con = duckdb.connect(test_db)
    con.execute(
        """
        CREATE TABLE prediction_log(
          pred_id BIGINT PRIMARY KEY, match_id INTEGER, predicted_at TIMESTAMP, as_of TIMESTAMP,
          config_hash TEXT, data_hash TEXT, git_commit TEXT, model_version TEXT,
          lam_home DOUBLE, lam_away DOUBLE, rho DOUBLE, score_matrix_json TEXT,
          odds_snapshot_id BIGINT, prev_hash TEXT, row_hash TEXT
        )
        """
    )
    con.close()

    logger = PredictionLogger(db_path=test_db)
    now = pd.Timestamp.now(tz="UTC")

    res1 = logger.log_prediction(
        match_id=101, as_of=now, lam_home=1.5, lam_away=1.0, rho=-0.05,
        score_matrix_arr=np.eye(11) / 11.0,
    )
    assert res1["prev_hash"] == "0" * 64

    res2 = logger.log_prediction(
        match_id=102, as_of=now, lam_home=1.8, lam_away=0.9, rho=-0.04,
        score_matrix_arr=np.eye(11) / 11.0,
    )
    assert res2["prev_hash"] == res1["row_hash"]

    audit = verify_prediction_log(db_path=test_db)
    assert audit["valid"] is True
    assert audit["count"] == 2

    con = duckdb.connect(test_db)
    con.execute("UPDATE prediction_log SET lam_home = 9.99 WHERE pred_id = 1")
    con.close()

    audit_tampered = verify_prediction_log(db_path=test_db)
    assert audit_tampered["valid"] is False
    assert "mismatch" in audit_tampered["error"].lower()


def test_season_2627_statistics_and_xg():
    """Verify that 2026/27 (2627) season statistics, xG/xGA/xPts, and standings are complete and coherent."""
    summary = compute_season_statistics(season="2627")
    assert summary["season"] == "2627"
    assert summary["season_label"] == "2026/27"
    assert summary["matches_played"] == 72
    assert summary["teams_count"] == 18
    assert summary["has_true_xg"] is True
    assert 2.0 < summary["xg_per_match"] < 4.0

    st_df = summary["standings"]
    assert len(st_df) == 18
    assert int(st_df["OM"].sum()) == 72 * 2
    assert int(st_df["AG"].sum()) == int(st_df["YG"].sum())
    assert np.isclose(st_df["xG"].sum(), st_df["xGA"].sum(), atol=1e-6)
    assert np.isclose(st_df["xGD"].sum(), 0.0, atol=1e-6)
    assert all(st_df["xPts"] > 0.0)


def test_sync_module_and_metadata(tmp_path):
    """Verify data synchronization runner and metadata tracking."""
    meta_file = str(tmp_path / "sync_meta.json")
    res = sync_latest_data(seasons=["2627"], sync_fixtures=False, meta_path=meta_file)
    assert res["db_matches"] > 0
    assert "timestamp_utc" in res

    meta = get_last_sync_info(meta_path=meta_file)
    assert meta is not None
    assert meta["timestamp_utc"] == res["timestamp_utc"]

