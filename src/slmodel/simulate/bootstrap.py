"""Uncertainty quantification via Bayesian bootstrap (Rubin 1981) and analytical Laplace posterior sampling."""

from datetime import datetime
from typing import Any, Optional
import numpy as np
import pandas as pd
from slmodel.models.dixon_coles import DixonColesModel
from slmodel.simulate.markets import one_x_two, over_under, both_teams_to_score
from slmodel.simulate.score_matrix import score_matrix


def _ci90(arr: np.ndarray) -> tuple[float, float, float]:
    """Return (5th percentile, median, 95th percentile)."""
    return (
        float(np.percentile(arr, 5)),
        float(np.median(arr)),
        float(np.percentile(arr, 95)),
    )


def laplace_posterior_prediction(
    model: DixonColesModel,
    home_team_id: int,
    away_team_id: int,
    crowd: str = "full",
    context_features: Optional[dict[str, float]] = None,
    n_samples: int = 400,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Fast analytical Laplace posterior sampling using the exact Fisher/Hessian covariance matrix.
    Draws S joint samples of (eta_home, eta_away, rho) in ~10ms to compute the Posterior
    Predictive Score Matrix M_mean, 90% credible intervals, and posterior standard deviations.
    """
    rng = np.random.default_rng(seed)
    unc = model.predict_with_uncertainty(
        home_team_id, away_team_id, crowd=crowd, context_features=context_features
    )
    lam_0 = unc["lam"]
    mu_0 = unc["mu"]
    rho_0 = unc["rho"]
    sig_h = unc["sigma_eta_h"]
    sig_a = unc["sigma_eta_a"]
    cov_ha = unc["cov_eta_ha"]
    draw_inf = unc["draw_inflation"]

    var_h = max(sig_h ** 2, 1e-8)
    var_a = max(sig_a ** 2, 1e-8)
    max_cov = 0.999 * np.sqrt(var_h * var_a)
    cov_ha = float(np.clip(cov_ha, -max_cov, max_cov))
    cov_mat = np.array([[var_h, cov_ha], [cov_ha, var_a]], dtype=np.float64)

    mean_eta = np.array([
        np.log(max(lam_0, 1e-8)) - 0.5 * var_h,
        np.log(max(mu_0, 1e-8)) - 0.5 * var_a,
    ])

    eta_draws = rng.multivariate_normal(mean_eta, cov_mat, size=n_samples)
    rho_draws = np.clip(rng.normal(rho_0, 0.025, size=n_samples), -0.45, 0.45)

    lam_h_arr = np.exp(eta_draws[:, 0])
    lam_a_arr = np.exp(eta_draws[:, 1])

    matrices = []
    p_1x2_list = []
    p_over25_list = []
    p_btts_list = []

    for s in range(n_samples):
        M = score_matrix(
            float(lam_h_arr[s]),
            float(lam_a_arr[s]),
            float(rho_draws[s]),
            draw_inflation=draw_inf,
        )
        matrices.append(M)
        p_1x2_list.append(one_x_two(M))
        p_over25_list.append(over_under(M, 2.5)["over"])
        p_btts_list.append(both_teams_to_score(M)["yes"])

    M_mean = np.mean(matrices, axis=0)
    M_mean /= M_mean.sum()

    p_1x2_arr = np.array(p_1x2_list)
    p_ou_arr = np.array(p_over25_list)
    p_btts_arr = np.array(p_btts_list)

    return {
        "M_mean": M_mean,
        "n_samples": n_samples,
        "credible_intervals": {
            "lam_home": _ci90(lam_h_arr),
            "lam_away": _ci90(lam_a_arr),
            "p_home": _ci90(p_1x2_arr[:, 0]),
            "p_draw": _ci90(p_1x2_arr[:, 1]),
            "p_away": _ci90(p_1x2_arr[:, 2]),
            "p_over25": _ci90(p_ou_arr),
            "p_btts": _ci90(p_btts_arr),
        },
        "posterior_std": {
            "p_home": float(np.std(p_1x2_arr[:, 0], ddof=1)),
            "p_draw": float(np.std(p_1x2_arr[:, 1], ddof=1)),
            "p_away": float(np.std(p_1x2_arr[:, 2], ddof=1)),
            "p_over25": float(np.std(p_ou_arr, ddof=1)),
            "p_btts": float(np.std(p_btts_arr, ddof=1)),
        },
    }


def bayesian_bootstrap_prediction(
    df: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    as_of: Optional[datetime] = None,
    n_bootstraps: int = 50,
    half_life_years: float = 1.25,
    crowd: str = "full",
    context_features: Optional[dict[str, float]] = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Execute non-parametric Bayesian bootstrap (Rubin 1981) with analytical-gradient warm-starting
    to quantify model parameter uncertainty across Dirichlet-reweighted match histories.
    """
    rng = np.random.default_rng(seed)

    valid = df[(df["status"] == "played") & df["home_goals"].notna() & df["away_goals"].notna()].copy()
    if as_of is not None:
        as_of_ts = pd.to_datetime(as_of)
        if as_of_ts.tz is not None:
            as_of_ts = as_of_ts.tz_convert("UTC").tz_localize(None)
        k_series = pd.to_datetime(valid["kickoff_utc"])
        if hasattr(k_series.dt, "tz") and k_series.dt.tz is not None:
            k_series = k_series.dt.tz_convert("UTC").dt.tz_localize(None)
        valid = valid[k_series < as_of_ts]

    n_matches = len(valid)
    if n_matches < 20:
        raise ValueError(f"Insufficient matches for bootstrap: {n_matches}")

    # Fit base model once to warm-start all bootstrap replicates
    base_model = DixonColesModel(half_life_years=half_life_years)
    base_model.fit(valid, as_of=as_of)
    _, _, draw_inf = base_model.compute_context_adjustments(context_features)

    matrices = []
    lambdas_h = []
    lambdas_a = []
    p_1x2_list = []
    p_over25_list = []
    p_btts_list = []

    for _ in range(n_bootstraps):
        exp_samples = rng.exponential(scale=1.0, size=n_matches)
        omega = (exp_samples / exp_samples.sum()) * n_matches

        base_model.fit(valid, as_of=as_of, sample_weights=omega, warm_start=True)
        lam_h, lam_a, rho = base_model.predict_lambdas(
            home_team_id, away_team_id, crowd=crowd, context_features=context_features
        )
        M = score_matrix(lam_h, lam_a, rho, draw_inflation=draw_inf)

        matrices.append(M)
        lambdas_h.append(lam_h)
        lambdas_a.append(lam_a)

        p1x2 = one_x_two(M)
        p_1x2_list.append(p1x2)
        p_over25_list.append(over_under(M, 2.5)["over"])
        p_btts_list.append(both_teams_to_score(M)["yes"])

    M_mean = np.mean(matrices, axis=0)
    M_mean = M_mean / M_mean.sum()

    p_1x2_arr = np.array(p_1x2_list)
    p_ou_arr = np.array(p_over25_list)
    p_btts_arr = np.array(p_btts_list)
    lam_h_arr = np.array(lambdas_h)
    lam_a_arr = np.array(lambdas_a)

    return {
        "M_mean": M_mean,
        "n_bootstraps": n_bootstraps,
        "credible_intervals": {
            "lam_home": _ci90(lam_h_arr),
            "lam_away": _ci90(lam_a_arr),
            "p_home": _ci90(p_1x2_arr[:, 0]),
            "p_draw": _ci90(p_1x2_arr[:, 1]),
            "p_away": _ci90(p_1x2_arr[:, 2]),
            "p_over25": _ci90(p_ou_arr),
            "p_btts": _ci90(p_btts_arr),
        },
        "posterior_std": {
            "p_home": float(np.std(p_1x2_arr[:, 0], ddof=1)),
            "p_draw": float(np.std(p_1x2_arr[:, 1], ddof=1)),
            "p_away": float(np.std(p_1x2_arr[:, 2], ddof=1)),
            "p_over25": float(np.std(p_ou_arr, ddof=1)),
            "p_btts": float(np.std(p_btts_arr, ddof=1)),
        },
    }
