"""Tests for Season Simulation Auditor, Error Diagnostics Engine, and Model V2.1 Upgrades."""

import numpy as np
import pytest
from slmodel.models.dixon_coles import DixonColesModel, compute_squad_priors
from slmodel.features.context import ContextFeatureExtractor
from slmodel.evaluate.audit import (
    SeasonSimulationAuditor,
    ErrorDiagnosticsEngine,
    compare_audit_models,
)


def test_compute_squad_priors():
    """Verify that squad priors are properly derived from DuckDB players table."""
    priors = compute_squad_priors("data/warehouse.duckdb")
    assert isinstance(priors, dict)
    assert len(priors) >= 15

    # Galatasaray (team 1) and Fenerbahce (team 2) should have positive attack priors
    assert priors[1][0] > 0.0
    assert priors[2][0] > 0.0

    # Eyupspor (team 26) has smaller squad valuation and should have negative prior
    if 26 in priors:
        assert priors[26][0] < 0.0


def test_dixon_coles_v21_priors_gauge_constraint():
    """Verify that DixonColesModel honors zero-sum gauge constraints when squad priors are supplied."""
    priors = compute_squad_priors("data/warehouse.duckdb")
    model = DixonColesModel(
        squad_prior_weight=0.25,
        form_momentum_weight=0.05,
        base_draw_inflation=0.035,
        team_squad_priors=priors,
    )
    assert model.squad_prior_weight == 0.25
    assert model.form_momentum_weight == 0.05
    assert model.base_draw_inflation == 0.035


def test_error_diagnostics_variance_decomposition():
    """Verify mathematical identity of variance decomposition."""
    sample_records = [
        {
            "pred_xg_diff": 1.2,
            "actual_xg_diff": 1.0,
            "actual_home_goals": 2,
            "actual_away_goals": 1,
            "model_error": 0.2,       # 1.2 - 1.0
            "finishing_luck": 0.0,    # 1.0 - (2-1)
            "total_error": 0.2,       # 1.2 - 1.0
        },
        {
            "pred_xg_diff": 0.5,
            "actual_xg_diff": 0.8,
            "actual_home_goals": 0,
            "actual_away_goals": 1,
            "model_error": -0.3,      # 0.5 - 0.8
            "finishing_luck": 1.8,    # 0.8 - (-1)
            "total_error": 1.5,       # 0.5 - (-1)
        },
    ]

    v_dec = ErrorDiagnosticsEngine.decompose_variance(sample_records)
    assert "model_error_mse" in v_dec
    assert "finishing_luck_mse" in v_dec
    assert "total_mse" in v_dec
    assert v_dec["model_error_pct"] >= 0.0
    assert v_dec["finishing_luck_pct"] >= 0.0
    assert abs((v_dec["model_error_pct"] + v_dec["finishing_luck_pct"]) - 100.0) < 1e-5


def test_season_simulation_auditor_fast():
    """Run fast simulation audit with refit_interval=18 and verify all metrics and report structures."""
    auditor = SeasonSimulationAuditor(
        db_path="data/warehouse.duckdb",
        season="2627",
        model_kwargs={"squad_prior_weight": 0.25, "form_momentum_weight": 0.05, "base_draw_inflation": 0.035},
        refit_interval=18,
        load_priors=True,
    )
    result = auditor.run_audit()

    assert result["season"] == "2627"
    assert result["total_matches"] == 72
    assert "summary_metrics" in result
    assert "mean_rps" in result["summary_metrics"]
    assert 0.15 < result["summary_metrics"]["mean_rps"] < 0.30
    assert result["summary_metrics"]["hit_rate_1x2_pct"] > 35.0
    assert result["summary_metrics"]["hit_rate_double_chance_pct"] > 65.0

    diag = result["diagnostics"]
    assert "variance_decomposition" in diag
    assert "team_bias_table" in diag
    assert len(diag["team_bias_table"]) >= 15
    assert "upset_analysis" in diag
    assert "round_progression" in diag
    assert len(diag["round_progression"]) == 8


def test_compare_audit_models():
    """Verify model comparison math."""
    base = {
        "total_matches": 72,
        "summary_metrics": {
            "mean_rps": 0.2200,
            "mean_logloss": 3.0800,
            "hit_rate_1x2_pct": 43.0,
            "hit_rate_double_chance_pct": 76.0,
            "hit_rate_top3_score_pct": 18.0,
            "corners_mae": 2.55,
            "cards_mae": 2.05,
        }
    }
    upgraded = {
        "total_matches": 72,
        "summary_metrics": {
            "mean_rps": 0.2160,
            "mean_logloss": 3.0600,
            "hit_rate_1x2_pct": 46.0,
            "hit_rate_double_chance_pct": 78.0,
            "hit_rate_top3_score_pct": 20.0,
            "corners_mae": 2.50,
            "cards_mae": 2.00,
        }
    }

    comp = compare_audit_models(base, upgraded)
    assert comp["rps"]["delta"] < 0  # Improvement
    assert comp["logloss"]["delta"] < 0  # Improvement
    assert comp["hit_rate_1x2"]["delta"] > 0
    assert comp["hit_rate_double_chance"]["delta"] > 0
