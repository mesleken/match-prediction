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
