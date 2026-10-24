"""Market layer: Shin devigging, multi-market implied parameter inversion, and dual-weight log-intensity pooling."""

from typing import Any, Optional
import numpy as np
from scipy.optimize import brentq, least_squares, minimize
from slmodel.simulate.markets import one_x_two, over_under
from slmodel.simulate.score_matrix import score_matrix


def devig_shin(odds: np.ndarray | list[float], tol: float = 1e-12) -> np.ndarray:
    """
    Remove bookmaker margin using Shin's (1993) insider-trading model.
    Correctly accounts for the favourite-longshot bias.
    """
    pi = 1.0 / np.asarray(odds, dtype=np.float64)
    S = float(pi.sum())
    if S <= 1.0 + 1e-9:
        return pi / S

    def p_of(z: float) -> np.ndarray:
        rad = np.sqrt(z * z + 4.0 * (1.0 - z) * (pi ** 2) / S)
        return (rad - z) / (2.0 * (1.0 - z))

    try:
        z_star = brentq(lambda z: float(p_of(z).sum() - 1.0), 0.0, 0.999, xtol=tol)
        p = p_of(z_star)
        return p / p.sum()
    except Exception:
        return pi / S


def invert_lambdas(
    p_1x2_mkt: np.ndarray,
    p_over25_mkt: Optional[float] = None,
    rho: float = -0.05,
    x0: tuple[float, float] = (0.35, 0.15),
) -> tuple[float, float]:
    """
    Invert devigged market probabilities (1X2 + optional Over/Under 2.5)
    into implied (lambda_home, lambda_away) under a Dixon-Coles score matrix.
    """
    p_target = np.asarray(p_1x2_mkt, dtype=np.float64)

    def residual(log_params: np.ndarray) -> np.ndarray:
        lam = float(np.exp(np.clip(log_params[0], -2.5, 2.2)))
        mu = float(np.exp(np.clip(log_params[1], -2.5, 2.2)))
        M = score_matrix(lam, mu, rho)
        p = one_x_two(M)
        if p_over25_mkt is not None:
            po = over_under(M, 2.5)["over"]
            return np.array([
                p[0] - p_target[0],
                p[1] - p_target[1],
                p[2] - p_target[2],
                1.2 * (po - p_over25_mkt),
            ], dtype=np.float64)
        return p - p_target

    sol = least_squares(
        residual,
        np.array(x0, dtype=np.float64),
        bounds=([-2.5, -2.5], [2.2, 2.2]),
        xtol=1e-7,
        ftol=1e-7,
    )
    lam_implied = float(np.exp(sol.x[0]))
    mu_implied = float(np.exp(sol.x[1]))
    return lam_implied, mu_implied


def blend_lambdas(
    lam_mod: float,
    mu_mod: float,
    lam_mkt: float,
    mu_mkt: float,
    w_total: float = 0.25,
    w_superiority: float = 0.12,
) -> tuple[float, float]:
    """
    Blend model lambdas with market implied lambdas in orthogonal (total, superiority) log coordinates.

    u = log(lam + mu)  (total match goal pace)
    v = log(lam / mu)  (net team superiority)

    Because the structural model (Dixon-Coles + Shots + Context) is stronger on total pace (u)
    while the closing market captures late lineup/injury superiority (v), separate weights
    (w_total, w_superiority) achieve superior out-of-sample calibration.
    """
    w_u = float(np.clip(w_total, 0.0, 1.0))
    w_v = float(np.clip(w_superiority, 0.0, 1.0))

    tot_mod = max(lam_mod + mu_mod, 1e-6)
    ratio_mod = max(lam_mod / max(mu_mod, 1e-6), 1e-6)
    u_mod = np.log(tot_mod)
    v_mod = np.log(ratio_mod)

    tot_mkt = max(lam_mkt + mu_mkt, 1e-6)
    ratio_mkt = max(lam_mkt / max(mu_mkt, 1e-6), 1e-6)
    u_mkt = np.log(tot_mkt)
    v_mkt = np.log(ratio_mkt)

    u_f = w_u * u_mod + (1.0 - w_u) * u_mkt
    v_f = w_v * v_mod + (1.0 - w_v) * v_mkt

    tot_f = np.exp(u_f)
    exp_v = np.exp(v_f)

    lam_blended = float(tot_f * exp_v / (1.0 + exp_v))
    mu_blended = float(tot_f / (1.0 + exp_v))

    return lam_blended, mu_blended


def learn_blend_weights(
    history_records: list[dict[str, Any]],
    default_w_total: float = 0.25,
    default_w_sup: float = 0.12,
) -> tuple[float, float]:
    """
    Learn optimal (w_total, w_superiority) strictly from historical out-of-sample match records
    by minimizing composite Ranked Probability Score + Score Log-Loss.
    """
    if len(history_records) < 100:
        return default_w_total, default_w_sup

    recent = history_records[-800:]

    def obj(w_vec: np.ndarray) -> float:
        wt = float(np.clip(w_vec[0], 0.0, 1.0))
        ws = float(np.clip(w_vec[1], 0.0, 1.0))
        loss = 0.0
        for r in recent:
            lb, mb = blend_lambdas(
                r["lam_dc"], r["mu_dc"], r["lam_mkt"], r["mu_mkt"],
                w_total=wt, w_superiority=ws,
            )
            M = score_matrix(lb, mb, r["rho_dc"])
            p = one_x_two(M)
            out_idx = r["outcome_idx"]
            cum_p = np.cumsum(p)
            obs = np.zeros(3)
            obs[out_idx] = 1.0
            cum_o = np.cumsum(obs)
            rps_val = 0.5 * ((cum_p[0] - cum_o[0]) ** 2 + (cum_p[1] - cum_o[1]) ** 2)
            hg = min(int(r["home_goals"]), 10)
            ag = min(int(r["away_goals"]), 10)
            ll_val = -np.log(max(float(M[hg, ag]), 1e-9))
            loss += rps_val + 0.02 * ll_val
        return loss / len(recent)

    res = minimize(
        obj,
        np.array([default_w_total, default_w_sup], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(0.0, 0.65), (0.0, 0.50)],
        options={"maxiter": 25},
    )
    return float(res.x[0]), float(res.x[1])
