"""Score matrix generation with Dixon-Coles dependency and Gauss-Hermite Posterior Predictive integration."""

import numpy as np
from scipy.special import gammaln

DEFAULT_MAX_GOALS = 10

# Precomputed 5-point Gauss-Hermite quadrature nodes (for standard normal Z ~ N(0,1)) and weights
_GH_NODES_RAW, _GH_WEIGHTS_RAW = np.polynomial.hermite.hermgauss(5)
_GH_NODES = _GH_NODES_RAW * np.sqrt(2.0)
_GH_WEIGHTS = _GH_WEIGHTS_RAW / np.sqrt(np.pi)


def pois_logpmf(lam: float, max_goals: int = DEFAULT_MAX_GOALS) -> np.ndarray:
    """Compute log Poisson PMF for k in 0..max_goals."""
    g = np.arange(max_goals + 1, dtype=np.float64)
    safe_lam = max(float(lam), 1e-8)
    return g * np.log(safe_lam) - safe_lam - gammaln(g + 1.0)


def rho_bounds(lam: float, mu: float, eps: float = 1e-3) -> tuple[float, float]:
    """
    Feasibility boundaries for rho such that tau(x, y) >= eps for (x, y) in {0,1}^2.
    """
    l = max(float(lam), 1e-6)
    m = max(float(mu), 1e-6)
    lower = max(-(1.0 - eps) / l, -(1.0 - eps) / m)
    upper = min((1.0 - eps) / (l * m), 1.0 - eps)
    return (float(lower), float(upper))


def tau_matrix(
    lam: float,
    mu: float,
    rho: float,
    draw_inflation: float = 0.0,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> np.ndarray:
    """Compute (max_goals+1, max_goals+1) tau adjustment matrix with optional diagonal draw inflation."""
    tau = np.ones((max_goals + 1, max_goals + 1), dtype=np.float64)
    tau[0, 0] = 1.0 - lam * mu * rho
    tau[0, 1] = 1.0 + lam * rho
    tau[1, 0] = 1.0 + mu * rho
    tau[1, 1] = 1.0 - rho
    if draw_inflation != 0.0:
        idx = np.arange(max_goals + 1)
        tau[idx, idx] *= np.exp(float(draw_inflation))
    return np.clip(tau, 1e-9, None)


def score_matrix(
    lam: float,
    mu: float,
    rho: float,
    draw_inflation: float = 0.0,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> np.ndarray:
    """
    Generate normalized (max_goals+1, max_goals+1) probability matrix.
    M[x, y] is P(HomeGoals = x, AwayGoals = y).
    """
    log_p_home = pois_logpmf(lam, max_goals=max_goals)
    log_p_away = pois_logpmf(mu, max_goals=max_goals)
    indep = np.exp(log_p_home[:, None] + log_p_away[None, :])
    t_mat = tau_matrix(lam, mu, rho, draw_inflation=draw_inflation, max_goals=max_goals)
    mat = indep * t_mat
    s = mat.sum()
    if s > 0:
        mat = mat / s
    return mat


def posterior_predictive_score_matrix(
    lam: float,
    mu: float,
    rho: float,
    sigma_eta_h: float = 0.0,
    sigma_eta_a: float = 0.0,
    cov_eta_ha: float = 0.0,
    tempo_sigma: float = 0.0,
    draw_inflation: float = 0.0,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> np.ndarray:
    """
    Compute the Posterior Predictive Score Matrix by integrating over parameter
    estimation covariance and match-level tempo overdispersion via 2D Gauss-Hermite quadrature.

    Preserves exact marginal mean expectations E[X] = lam, E[Y] = mu while capturing
    tail variance inflation Var(X) = lam + (exp(Sigma_11) - 1) * lam^2.
    """
    var_h = float(sigma_eta_h ** 2 + tempo_sigma ** 2)
    var_a = float(sigma_eta_a ** 2 + tempo_sigma ** 2)
    cov_ha = float(cov_eta_ha + tempo_sigma ** 2)

    if var_h <= 1e-10 and var_a <= 1e-10:
        return score_matrix(lam, mu, rho, draw_inflation=draw_inflation, max_goals=max_goals)

    # Construct 2x2 covariance matrix for log-intensities (eta_h, eta_a)
    max_cov = 0.999 * np.sqrt(max(var_h * var_a, 1e-16))
    cov_ha = float(np.clip(cov_ha, -max_cov, max_cov))
    cov_mat = np.array([[max(var_h, 1e-12), cov_ha], [cov_ha, max(var_a, 1e-12)]], dtype=np.float64)

    # Cholesky factor L such that L @ L.T = cov_mat
    L = np.linalg.cholesky(cov_mat)

    # Mean-preserving shift: E[exp(mu_eta - 0.5*var + Z)] = exp(mu_eta)
    eta_h0 = np.log(max(float(lam), 1e-8)) - 0.5 * var_h
    eta_a0 = np.log(max(float(mu), 1e-8)) - 0.5 * var_a

    g = np.arange(max_goals + 1, dtype=np.float64)
    log_fact = gammaln(g + 1.0)

    M_acc = np.zeros((max_goals + 1, max_goals + 1), dtype=np.float64)

    for i, z1 in enumerate(_GH_NODES):
        w1 = _GH_WEIGHTS[i]
        for j, z2 in enumerate(_GH_NODES):
            w = w1 * _GH_WEIGHTS[j]
            delta_h = L[0, 0] * z1
            delta_a = L[1, 0] * z1 + L[1, 1] * z2

            lam_q = float(np.exp(eta_h0 + delta_h))
            mu_q = float(np.exp(eta_a0 + delta_a))

            log_ph = g * np.log(max(lam_q, 1e-8)) - lam_q - log_fact
            log_pa = g * np.log(max(mu_q, 1e-8)) - mu_q - log_fact
            indep = np.exp(log_ph[:, None] + log_pa[None, :])
            t_mat = tau_matrix(lam_q, mu_q, rho, draw_inflation=draw_inflation, max_goals=max_goals)
            mat_q = indep * t_mat
            s_q = mat_q.sum()
            if s_q > 0:
                mat_q /= s_q
            M_acc += w * mat_q

    s_tot = M_acc.sum()
    if s_tot > 0:
        M_acc /= s_tot
    return M_acc


def bivariate_poisson_score_matrix(
    lam: float,
    mu: float,
    rho: float = 0.0,
    lambda_0: float = 0.0,
    draw_inflation: float = 0.0,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> np.ndarray:
    """
    Generate bivariate Poisson score matrix with shared covariance parameter lambda_0
    and Dixon-Coles tau adjustment (Karlis & Ntzoufras 2003).

    P(X=x, Y=y) = exp(-(lam1 + lam2 + lam0)) * sum_{k=0}^{min(x,y)} [ lam1^(x-k)/(x-k)! * lam2^(y-k)/(y-k)! * lam0^k / k! ]
    where lam1 = max(lam - lam0, 1e-6), lam2 = max(mu - lam0, 1e-6), and Cov(X, Y) = lam0.
    """
    if lambda_0 <= 1e-8:
        return score_matrix(lam, mu, rho, draw_inflation=draw_inflation, max_goals=max_goals)

    lam0 = float(min(lambda_0, max(float(lam), float(mu)) * 0.45, 0.50))
    lam1 = max(float(lam) - lam0, 1e-6)
    lam2 = max(float(mu) - lam0, 1e-6)

    g = np.arange(max_goals + 1, dtype=np.float64)
    log_fact = gammaln(g + 1.0)

    M = np.zeros((max_goals + 1, max_goals + 1), dtype=np.float64)
    base_const = np.exp(-(lam1 + lam2 + lam0))

    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            k_max = min(x, y)
            k_arr = np.arange(k_max + 1, dtype=np.float64)
            log_terms = (
                (x - k_arr) * np.log(lam1)
                - log_fact[x - k_arr.astype(int)]
                + (y - k_arr) * np.log(lam2)
                - log_fact[y - k_arr.astype(int)]
                + k_arr * np.log(max(lam0, 1e-12))
                - log_fact[k_arr.astype(int)]
            )
            max_lt = np.max(log_terms)
            sum_val = np.sum(np.exp(log_terms - max_lt)) * np.exp(max_lt)
            M[x, y] = base_const * sum_val

    t_mat = tau_matrix(lam, mu, rho, draw_inflation=draw_inflation, max_goals=max_goals)
    mat = M * t_mat
    s = mat.sum()
    if s > 0:
        mat /= s
    return mat
