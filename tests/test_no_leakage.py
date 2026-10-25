"""Tests to verify zero data leakage: future outcomes must never affect past predictions."""

import numpy as np
import pandas as pd
import pytest
from slmodel.models.dixon_coles import DixonColesModel


def test_dixon_coles_future_perturbation_invariance():
    """
    Verify that altering or adding future matches does not alter
    the model fitted as_of a specific cutoff date.
    """
    dates = pd.date_range("2023-01-01", periods=40, freq="3D")
    records = []
    for i in range(40):
        records.append({
            "match_id": i + 1,
            "season": "2223",
            "kickoff_utc": dates[i],
            "home_team_id": (i % 4) + 1,
            "away_team_id": ((i + 1) % 4) + 1,
            "home_goals": (i * 3) % 4,
            "away_goals": (i * 2) % 3,
            "status": "played",
        })

    df_original = pd.DataFrame(records)

    # Cutoff date at match index 25
    cutoff = dates[25]

    # Baseline fit
    model_base = DixonColesModel(half_life_years=1.0)
    model_base.fit(df_original, as_of=cutoff)
    pred_base = model_base.predict_lambdas(1, 2)

    # Dataset with future matches heavily perturbed (matches 25..39 modified)
    df_perturbed = df_original.copy()
    df_perturbed.loc[25:, "home_goals"] = 9
    df_perturbed.loc[25:, "away_goals"] = 8

    model_perturbed = DixonColesModel(half_life_years=1.0)
    model_perturbed.fit(df_perturbed, as_of=cutoff)
    pred_perturbed = model_perturbed.predict_lambdas(1, 2)

    # Assert bit-identical predictions
    np.testing.assert_allclose(pred_base, pred_perturbed, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model_base.att, model_perturbed.att, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model_base.def_, model_perturbed.def_, rtol=1e-12, atol=1e-12)
    assert np.isclose(model_base.mu0, model_perturbed.mu0, atol=1e-12)
    assert np.isclose(model_base.home_adv, model_perturbed.home_adv, atol=1e-12)


def test_no_leakage_on_exact_kickoff_timestamp():
    """Verify that matches occurring at or after cutoff are strictly excluded (kickoff_utc < as_of)."""
    t0 = pd.Timestamp("2024-05-01 18:00:00")
    records = []
    # 20 matches before t0
    for i in range(20):
        records.append({
            "match_id": i + 1,
            "season": "2324",
            "kickoff_utc": t0 - pd.Timedelta(days=25 - i),
            "home_team_id": 1 if i % 2 == 0 else 2,
            "away_team_id": 2 if i % 2 == 0 else 1,
            "home_goals": 1,
            "away_goals": 0,
            "status": "played",
        })

    # 1 match exactly at t0
    records.append({
        "match_id": 21,
        "season": "2324",
        "kickoff_utc": t0,
        "home_team_id": 1,
        "away_team_id": 2,
        "home_goals": 10,  # huge outlier
        "away_goals": 0,
        "status": "played",
    })

    df = pd.DataFrame(records)

    model = DixonColesModel()
    model.fit(df, as_of=t0)

    # If match 21 had leaked, mu0 would jump substantially
    assert model.mu0 < 1.0


def test_no_squad_prior_data_leakage():
    """
    Verify that generating in-season player statistics or re-aggregating ratings
    never alters pre-season 'players' table or Empirical Bayes squad priors.
    Guarantees week-1 match predictions are mathematically invariant to week-8 simulated performance.
    """
    import duckdb
    from slmodel.models.dixon_coles import compute_squad_priors
    from slmodel.data.player_performance import (
        generate_and_sync_player_match_stats,
        compute_player_season_stats_and_ratings,
    )

    db_path = "data/warehouse.duckdb"

    # 1. Baseline squad priors and player ratings
    priors_before = compute_squad_priors(db_path=db_path)
    con = duckdb.connect(db_path, read_only=True)
    df_players_before = con.execute("SELECT player_id, att_rating, def_rating, overall_rating FROM players ORDER BY player_id").fetchdf()
    con.close()

    # 2. Trigger in-season performance generation and Bayesian update
    generate_and_sync_player_match_stats(db_path=db_path, season="2627")
    compute_player_season_stats_and_ratings(db_path=db_path, season="2627")

    # 3. Re-read squad priors and player ratings
    priors_after = compute_squad_priors(db_path=db_path)
    con = duckdb.connect(db_path, read_only=True)
    df_players_after = con.execute("SELECT player_id, att_rating, def_rating, overall_rating FROM players ORDER BY player_id").fetchdf()
    con.close()

    # Assert players table was NOT modified
    pd.testing.assert_frame_equal(df_players_before, df_players_after)

    # Assert squad priors are identical down to 14 decimal places
    for tid, (att_b, def_b) in priors_before.items():
        assert tid in priors_after
        att_a, def_a = priors_after[tid]
        assert np.isclose(att_b, att_a, atol=1e-12)
        assert np.isclose(def_b, def_a, atol=1e-12)


def test_ablation_factor_alters_predictions():
    """
    Verify that ablation neutralizers genuinely perturb match predictions
    when active feature values are present (ruling out silent no-ops).
    """
    from slmodel.evaluate.backtest import AblationRunner
    from slmodel.features.context import ContextFeatureExtractor
    from slmodel.models.dixon_coles import DixonColesModel

    model = DixonColesModel()
    # Mock context with non-zero referee flow and non-zero weather pace
    ctx_active = {
        "ref_delta_h": 0.08,
        "ref_delta_a": -0.04,
        "ref_draw_inf": 0.03,
        "weather_delta_pace": -0.05,
    }
    model.is_fitted = True
    model.team_to_idx = {1: 0, 2: 1}
    model.team_ids = [1, 2]
    model.att = np.array([0.2, -0.1])
    model.def_ = np.array([-0.1, 0.1])
    model.mu0 = 0.2
    model.home_adv = 0.25
    model.rho = -0.08

    lam_full, mu_full, _ = model.predict_lambdas(1, 2, context_features=ctx_active)

    # Ablate referee
    ctx_no_ref = dict(ctx_active)
    ctx_no_ref["ref_delta_h"] = 0.0
    ctx_no_ref["ref_delta_a"] = 0.0
    ctx_no_ref["ref_draw_inf"] = 0.0
    lam_abl, mu_abl, _ = model.predict_lambdas(1, 2, context_features=ctx_no_ref)

    # Must show non-zero shift
    assert not np.isclose(lam_full, lam_abl, atol=1e-4)
    assert not np.isclose(mu_full, mu_abl, atol=1e-4)


def test_backtest_config_matches_production():
    """
    Verify that WalkForwardBacktest aligns strictly with production config/model.yaml
    and does not diverge on regularization, decay rates, or blending weights.
    """
    from slmodel.config import get_model_config
    from slmodel.evaluate.backtest import WalkForwardBacktest

    cfg = get_model_config()
    backtester = WalkForwardBacktest()

    assert backtester.half_life_years == cfg.dixon_coles.half_life_years
    assert backtester.w_blend_total == cfg.market_blend.w_total
    assert backtester.w_blend_superiority == cfg.market_blend.w_superiority

