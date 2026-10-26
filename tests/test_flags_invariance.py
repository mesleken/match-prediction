"""Strict invariance tests proving flags control context adjustments without silent leaks."""

import numpy as np
import pytest

from slmodel.models.dixon_coles import DixonColesModel


def _build_dummy_fitted_model(**kwargs) -> DixonColesModel:
    model = DixonColesModel(**kwargs)
    model.is_fitted = True
    model.team_to_idx = {1: 0, 2: 1}
    model.team_ids = [1, 2]
    model.att = np.array([0.15, -0.10], dtype=np.float64)
    model.def_ = np.array([-0.05, 0.08], dtype=np.float64)
    model.mu0 = 0.22
    model.home_adv = 0.26
    model.rho = -0.09
    return model


def test_flag_false_invariance_venue_edge():
    """Prove that when venue_edge_enabled=False, varying venue_edge_h changes lambda by exactly 0.0%."""
    model_disabled = _build_dummy_fitted_model(venue_edge_enabled=False)
    lam_0, mu_0, _ = model_disabled.predict_lambdas(1, 2, context_features={"venue_edge_h": 0.0, "venue_edge_a": 0.0})
    lam_1, mu_1, _ = model_disabled.predict_lambdas(1, 2, context_features={"venue_edge_h": 1.0, "venue_edge_a": -1.0})

    assert np.isclose(lam_0, lam_1, atol=1e-12), f"Flag disabled leaked! {lam_0} != {lam_1}"
    assert np.isclose(mu_0, mu_1, atol=1e-12)

    # When enabled, must change
    model_enabled = _build_dummy_fitted_model(venue_edge_enabled=True)
    lam_en_0, _, _ = model_enabled.predict_lambdas(1, 2, context_features={"venue_edge_h": 0.0})
    lam_en_1, _, _ = model_enabled.predict_lambdas(1, 2, context_features={"venue_edge_h": 1.0})
    assert not np.isclose(lam_en_0, lam_en_1, atol=1e-4)


def test_flag_false_invariance_referee():
    """Prove that when referee_enabled=False, varying ref_delta_h changes lambda by exactly 0.0%."""
    model_disabled = _build_dummy_fitted_model(referee_enabled=False)
    lam_0, mu_0, dr_0 = model_disabled.predict_lambdas(1, 2, context_features={"ref_delta_h": 0.0, "ref_delta_a": 0.0, "ref_draw_inf": 0.0})
    lam_1, mu_1, dr_1 = model_disabled.predict_lambdas(1, 2, context_features={"ref_delta_h": 0.08, "ref_delta_a": -0.05, "ref_draw_inf": 0.04})

    assert np.isclose(lam_0, lam_1, atol=1e-12), f"Flag disabled leaked! {lam_0} != {lam_1}"
    assert np.isclose(mu_0, mu_1, atol=1e-12)
    assert np.isclose(dr_0, dr_1, atol=1e-12)

    # When enabled, must change
    model_enabled = _build_dummy_fitted_model(referee_enabled=True)
    lam_en_0, _, _ = model_enabled.predict_lambdas(1, 2, context_features={"ref_delta_h": 0.0})
    lam_en_1, _, _ = model_enabled.predict_lambdas(1, 2, context_features={"ref_delta_h": 0.08})
    assert not np.isclose(lam_en_0, lam_en_1, atol=1e-4)


def test_flag_false_invariance_euro_fatigue():
    """Prove that when euro_fatigue_enabled=False, varying euro_fatigue_h changes lambda by exactly 0.0%."""
    model_disabled = _build_dummy_fitted_model(euro_fatigue_enabled=False)
    lam_0, mu_0, _ = model_disabled.predict_lambdas(1, 2, context_features={"euro_fatigue_h": 0.0, "euro_fatigue_a": 0.0})
    lam_1, mu_1, _ = model_disabled.predict_lambdas(1, 2, context_features={"euro_fatigue_h": -0.025, "euro_fatigue_a": -0.025})

    assert np.isclose(lam_0, lam_1, atol=1e-12), f"Flag disabled leaked! {lam_0} != {lam_1}"
    assert np.isclose(mu_0, mu_1, atol=1e-12)

    # When enabled, must change
    model_enabled = _build_dummy_fitted_model(euro_fatigue_enabled=True)
    lam_en_0, _, _ = model_enabled.predict_lambdas(1, 2, context_features={"euro_fatigue_h": 0.0})
    lam_en_1, _, _ = model_enabled.predict_lambdas(1, 2, context_features={"euro_fatigue_h": -0.025})
    assert not np.isclose(lam_en_0, lam_en_1, atol=1e-4)


def test_flag_false_invariance_weather():
    """Prove that when weather_enabled=False, varying weather_delta_pace changes lambda by exactly 0.0%."""
    model_disabled = _build_dummy_fitted_model(weather_enabled=False)
    lam_0, mu_0, _ = model_disabled.predict_lambdas(1, 2, context_features={"weather_delta_pace": 0.0})
    lam_1, mu_1, _ = model_disabled.predict_lambdas(1, 2, context_features={"weather_delta_pace": -0.035})

    assert np.isclose(lam_0, lam_1, atol=1e-12), f"Flag disabled leaked! {lam_0} != {lam_1}"
    assert np.isclose(mu_0, mu_1, atol=1e-12)

    # When enabled, must change
    model_enabled = _build_dummy_fitted_model(weather_enabled=True)
    lam_en_0, _, _ = model_enabled.predict_lambdas(1, 2, context_features={"weather_delta_pace": 0.0})
    lam_en_1, _, _ = model_enabled.predict_lambdas(1, 2, context_features={"weather_delta_pace": -0.035})
    assert not np.isclose(lam_en_0, lam_en_1, atol=1e-4)


def test_flag_false_invariance_availability():
    """Prove that when availability_enabled=False, varying avail_shift_h changes lambda by exactly 0.0%."""
    model_disabled = _build_dummy_fitted_model(availability_enabled=False)
    lam_0, mu_0, _ = model_disabled.predict_lambdas(1, 2, context_features={"avail_shift_h": 0.0, "avail_shift_a": 0.0})
    lam_1, mu_1, _ = model_disabled.predict_lambdas(1, 2, context_features={"avail_shift_h": -0.04, "avail_shift_a": 0.02})

    assert np.isclose(lam_0, lam_1, atol=1e-12), f"Flag disabled leaked! {lam_0} != {lam_1}"
    assert np.isclose(mu_0, mu_1, atol=1e-12)

    # When enabled, must change
    model_enabled = _build_dummy_fitted_model(availability_enabled=True)
    lam_en_0, _, _ = model_enabled.predict_lambdas(1, 2, context_features={"avail_shift_h": 0.0})
    lam_en_1, _, _ = model_enabled.predict_lambdas(1, 2, context_features={"avail_shift_h": -0.04})
    assert not np.isclose(lam_en_0, lam_en_1, atol=1e-4)
