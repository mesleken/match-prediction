"""Unit tests for mathematical core: score matrix, Gauss-Hermite posterior predictive integration, markets, Shin devig, parameter inversion, and Robust Kelly."""

import numpy as np
import pytest
from slmodel.simulate.score_matrix import (
    score_matrix,
    posterior_predictive_score_matrix,
    rho_bounds,
    tau_matrix,
)
from slmodel.simulate.markets import (
    one_x_two,
    draw_no_bet,
    double_chance,
    over_under,
    both_teams_to_score,
    goal_range_probabilities,
    clean_sheet_and_win_to_nil,
    expected_goals_and_variance,
    match_competitiveness,
    asian_handicap,
    most_likely_scores,
    kelly_stake,
)
from slmodel.models.market import devig_shin, invert_lambdas, blend_lambdas


def test_score_matrix_normalization_and_support():
    """Verify that score matrix generates a valid probability distribution."""
    for lam in [0.5, 1.3, 2.4]:
        for mu in [0.4, 1.1, 2.0]:
            for rho in [-0.10, 0.0, 0.08]:
                M = score_matrix(lam, mu, rho, max_goals=10)
                assert M.shape == (11, 11)
                assert np.all(M >= 0.0)
                assert np.isclose(float(M.sum()), 1.0, atol=1e-7)


def test_posterior_predictive_gauss_hermite_mean_and_overdispersion():
    """
    Verify that 2D Gauss-Hermite Posterior Predictive integration preserves
    exact mean goals E[X] while strictly increasing tail variance Var(X) (overdispersion).
    """
    lam, mu = 1.65, 1.15
    M_point = score_matrix(lam, mu, rho=0.0, max_goals=14)
    M_post = posterior_predictive_score_matrix(
        lam, mu, rho=0.0,
        sigma_eta_h=0.18, sigma_eta_a=0.18, cov_eta_ha=0.04,
        tempo_sigma=0.10, max_goals=14,
    )

    mom_point = expected_goals_and_variance(M_point)
    mom_post = expected_goals_and_variance(M_post)

    # Mean must be preserved within quadrature accuracy
    assert np.isclose(mom_post["mean_home"], mom_point["mean_home"], atol=1e-2)
    assert np.isclose(mom_post["mean_away"], mom_point["mean_away"], atol=1e-2)

    # Variance must be strictly greater due to parameter + tempo uncertainty (Poisson-LogNormal law)
    assert mom_post["var_home"] > mom_point["var_home"]
    assert mom_post["var_away"] > mom_point["var_away"]
    # Positive shared tempo + covariance induces positive goal correlation
    assert mom_post["corr_home_away"] > 0.0


def test_rho_bounds_tau_feasibility():
    """Verify that tau matrix values stay strictly positive within rho bounds."""
    for lam in [0.8, 1.5, 2.5]:
        for mu in [0.7, 1.2, 2.2]:
            lower, upper = rho_bounds(lam, mu)
            assert lower < 0 < upper
            tau_low = tau_matrix(lam, mu, lower + 0.001)
            tau_high = tau_matrix(lam, mu, upper - 0.001)
            assert np.all(tau_low[:2, :2] > 0.0)
            assert np.all(tau_high[:2, :2] > 0.0)


def test_market_derivations_coherence():
    """Verify coherence of all markets derived from the same joint matrix."""
    M = score_matrix(1.65, 1.15, -0.06)

    # 1X2 sums to 1.0
    p1x2 = one_x_two(M)
    assert np.isclose(float(p1x2.sum()), 1.0, atol=1e-7)
    assert p1x2[0] > 0 and p1x2[1] > 0 and p1x2[2] > 0

    # DNB
    dnb = draw_no_bet(M)
    assert np.isclose(dnb["1"] + dnb["2"], 1.0, atol=1e-7)
    assert np.isclose(dnb["1"], p1x2[0] / (p1x2[0] + p1x2[2]), atol=1e-7)

    # Double Chance
    dc = double_chance(M)
    assert np.isclose(dc["1X"], p1x2[0] + p1x2[1], atol=1e-7)
    assert np.isclose(dc["12"], p1x2[0] + p1x2[2], atol=1e-7)
    assert np.isclose(dc["X2"], p1x2[1] + p1x2[2], atol=1e-7)

    # Over/Under & Goal Ranges exact identities
    ou15 = over_under(M, 1.5)
    ou35 = over_under(M, 3.5)
    gr = goal_range_probabilities(M)
    assert np.isclose(sum(gr.values()), 1.0, atol=1e-7)
    assert np.isclose(gr["0-1"], ou15["under"], atol=1e-7)
    assert np.isclose(gr["0-1"] + gr["2-3"], ou35["under"], atol=1e-7)

    # Clean sheet & Win-to-Nil identities
    cs = clean_sheet_and_win_to_nil(M)
    assert np.isclose(cs["home_clean_sheet"], cs["home_win_to_nil"] + float(M[0, 0]), atol=1e-7)
    assert np.isclose(cs["away_clean_sheet"], cs["away_win_to_nil"] + float(M[0, 0]), atol=1e-7)

    # Competitiveness
    comp = match_competitiveness(M)
    assert comp["p_draw"] <= comp["p_close"] <= 1.0
    assert 0.0 <= comp["entropy_1x2"] <= 1.0

    # BTTS
    btts = both_teams_to_score(M)
    assert np.isclose(btts["yes"] + btts["no"], 1.0, atol=1e-7)

    # Top scores
    top10 = most_likely_scores(M, top_n=10)
    assert len(top10) == 10
    probs = [item["probability"] for item in top10]
    assert probs == sorted(probs, reverse=True)


def test_asian_handicap_quarter_line_ev_identity():
    """
    Verify quarter Asian Handicap decomposition.
    A -0.25 line is half stake on 0.0 (DNB) and half on -0.5.
    """
    M = score_matrix(1.5, 1.0, -0.04)

    ah_quarter = asian_handicap(M, -0.25)
    ah_zero = asian_handicap(M, 0.0)
    ah_half = asian_handicap(M, -0.5)

    assert np.isclose(sum(ah_quarter.values()), 1.0, atol=1e-7)
    assert np.isclose(sum(ah_zero.values()), 1.0, atol=1e-7)
    assert np.isclose(sum(ah_half.values()), 1.0, atol=1e-7)

    ev_q = (
        ah_quarter["full_win"] * 1.0
        + ah_quarter["half_win"] * 0.5
        + ah_quarter["push"] * 0.0
        + ah_quarter["half_loss"] * -0.5
        + ah_quarter["full_loss"] * -1.0
    )
    ev_0 = (
        ah_zero["full_win"] * 1.0
        + ah_zero["half_win"] * 0.5
        + ah_zero["push"] * 0.0
        + ah_zero["half_loss"] * -0.5
        + ah_zero["full_loss"] * -1.0
    )
    ev_h = (
        ah_half["full_win"] * 1.0
        + ah_half["half_win"] * 0.5
        + ah_half["push"] * 0.0
        + ah_half["half_loss"] * -0.5
        + ah_half["full_loss"] * -1.0
    )

    expected_q_ev = 0.5 * ev_0 + 0.5 * ev_h
    assert np.isclose(ev_q, expected_q_ev, atol=1e-7)


def test_shin_devig_properties():
    """Verify Shin devigging properties and favorite-longshot adjustment."""
    odds = [2.10, 3.40, 3.60]
    p_shin = devig_shin(odds)

    assert np.isclose(p_shin.sum(), 1.0, atol=1e-7)
    assert all(p > 0 for p in p_shin)

    inv = 1.0 / np.array(odds)
    p_prop = inv / inv.sum()

    assert p_shin[0] > p_prop[0]
    assert p_shin[2] < p_prop[2]


def test_market_lambda_inversion_recovery():
    """Verify that market inversion recovers known true Poisson rates."""
    lam_true = 1.80
    mu_true = 0.95
    rho_true = -0.05

    M_true = score_matrix(lam_true, mu_true, rho_true)
    p1x2_true = one_x_two(M_true)
    p_ou25_true = over_under(M_true, 2.5)["over"]

    lam_rec, mu_rec = invert_lambdas(p1x2_true, p_over25_mkt=p_ou25_true, rho=rho_true)

    assert np.isclose(lam_rec, lam_true, atol=1e-2)
    assert np.isclose(mu_rec, mu_true, atol=1e-2)


def test_blend_lambdas_boundary_behavior():
    """Verify lambda blending weights at extreme bounds."""
    lam_m, mu_m = 1.2, 0.8
    lam_b, mu_b = 2.0, 1.5

    l1, m1 = blend_lambdas(lam_m, mu_m, lam_b, mu_b, w_total=1.0, w_superiority=1.0)
    assert np.isclose(l1, lam_m, atol=1e-4)
    assert np.isclose(m1, mu_m, atol=1e-4)

    l0, m0 = blend_lambdas(lam_m, mu_m, lam_b, mu_b, w_total=0.0, w_superiority=0.0)
    assert np.isclose(l0, lam_b, atol=1e-4)
    assert np.isclose(m0, mu_b, atol=1e-4)


def test_robust_kelly_uncertainty_shrinkage():
    """Verify that Baker-McHale Robust Kelly penalizes high parameter uncertainty."""
    k_certain = kelly_stake(p_model=0.55, odds=2.00, p_std=0.0)
    k_uncertain = kelly_stake(p_model=0.55, odds=2.00, p_std=0.06)
    k_very_uncertain = kelly_stake(p_model=0.55, odds=2.00, p_std=0.12)

    assert k_certain["edge"] > 0
    assert k_certain["robust_kelly"] > k_uncertain["robust_kelly"] > k_very_uncertain["robust_kelly"] > 0.0
    assert k_uncertain["uncertainty_shrinkage"] < 1.0


def test_temperature_scaling_calibration():
    """Verify that calibrate_probabilities sharpens underconfident favorite probabilities."""
    from slmodel.simulate.markets import calibrate_probabilities

    # Uncalibrated heavy favorite
    raw_p = np.array([0.70, 0.20, 0.10])
    cal_p = calibrate_probabilities(raw_p, temperature=0.75)

    assert np.isclose(np.sum(cal_p), 1.0)
    # T < 1.0 sharpens favorite probability towards empirical reality (~78%)
    assert cal_p[0] > raw_p[0]
    assert cal_p[1] < raw_p[1]
    assert cal_p[2] < raw_p[2]

    # Neutral temperature returns original probabilities
    neutral_p = calibrate_probabilities(raw_p, temperature=1.0)
    assert np.allclose(neutral_p, raw_p)
