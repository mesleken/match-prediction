"""Half-Time / Full-Time (HT/FT) 4D state-conditioned model with exact marginal coherence to Full-Time Score Matrix M."""

from datetime import datetime
from typing import Any, Optional
import numpy as np
import pandas as pd
from scipy.special import gammaln
from slmodel.simulate.score_matrix import score_matrix
from slmodel.simulate.markets import one_x_two, over_under, both_teams_to_score

OUTCOMES_9 = [
    "1/1", "1/X", "1/2",
    "X/1", "X/X", "X/2",
    "2/1", "2/X", "2/2",
]

# Empirical Süper Lig defaults calibrated on 3,129 historical matches:
DEFAULT_PI_H = 0.438
DEFAULT_PI_A = 0.442

# Game-state tactical multipliers for 2nd half intensities:
DEFAULT_GAMMAS = {
    "home": {-1: 1.08, 0: 0.98, 1: 0.94},
    "away": {-1: 0.93, 0: 0.98, 1: 1.12},
}


def calibrate_htft_parameters(
    df: pd.DataFrame,
    as_of: Optional[datetime] = None,
) -> dict[str, Any]:
    """
    Empirically calibrate 1st-half goal shares (pi_h, pi_a) and 2nd-half game-state
    multipliers (gammas) from historical matches strictly prior to as_of.
    """
    valid = df[
        (df["status"] == "played")
        & df["home_goals"].notna()
        & df["away_goals"].notna()
        & df["ht_home_goals"].notna()
        & df["ht_away_goals"].notna()
    ].copy()

    if as_of is not None:
        as_of_ts = pd.to_datetime(as_of)
        if as_of_ts.tz is not None:
            as_of_ts = as_of_ts.tz_convert("UTC").tz_localize(None)
        k_s = pd.to_datetime(valid["kickoff_utc"])
        if hasattr(k_s.dt, "tz") and k_s.dt.tz is not None:
            k_s = k_s.dt.tz_convert("UTC").tz_localize(None)
        valid = valid[k_s < as_of_ts]

    if len(valid) < 100:
        return {"pi_h": DEFAULT_PI_H, "pi_a": DEFAULT_PI_A, "gammas": DEFAULT_GAMMAS}

    tot_h = max(float(valid["home_goals"].sum()), 1.0)
    tot_a = max(float(valid["away_goals"].sum()), 1.0)
    pi_h = float(np.clip(valid["ht_home_goals"].sum() / tot_h, 0.35, 0.55))
    pi_a = float(np.clip(valid["ht_away_goals"].sum() / tot_a, 0.35, 0.55))

    h2 = valid["home_goals"] - valid["ht_home_goals"]
    a2 = valid["away_goals"] - valid["ht_away_goals"]
    mean_h2 = max(float(h2.mean()), 0.1)
    mean_a2 = max(float(a2.mean()), 0.1)

    s_ht = np.sign(valid["ht_home_goals"] - valid["ht_away_goals"]).astype(int)
    gammas: dict[str, dict[int, float]] = {"home": {}, "away": {}}
    for s in [-1, 0, 1]:
        mask = s_ht == s
        if mask.sum() >= 20:
            # Shrink empirical ratio 50% toward 1.0 to avoid overfitting to team-strength selection bias
            raw_gh = float(h2[mask].mean()) / mean_h2
            raw_ga = float(a2[mask].mean()) / mean_a2
            gammas["home"][s] = float(0.5 * raw_gh + 0.5 * DEFAULT_GAMMAS["home"][s])
            gammas["away"][s] = float(0.5 * raw_ga + 0.5 * DEFAULT_GAMMAS["away"][s])
        else:
            gammas["home"][s] = DEFAULT_GAMMAS["home"][s]
            gammas["away"][s] = DEFAULT_GAMMAS["away"][s]

    return {"pi_h": pi_h, "pi_a": pi_a, "gammas": gammas}


def _poisson_pmf_vec(lam: float, max_g: int) -> np.ndarray:
    """Compute Poisson probability vector for 0..max_g."""
    k = np.arange(max_g + 1, dtype=np.float64)
    safe_lam = max(float(lam), 1e-7)
    log_p = k * np.log(safe_lam) - safe_lam - gammaln(k + 1.0)
    p = np.exp(log_p)
    return p / p.sum()


def build_joint_ht_ft_tensor(
    lam_h: float,
    lam_a: float,
    rho: float = -0.05,
    M_ft: Optional[np.ndarray] = None,
    pi_h: float = DEFAULT_PI_H,
    pi_a: float = DEFAULT_PI_A,
    gammas: Optional[dict[str, dict[int, float]]] = None,
) -> np.ndarray:
    """
    Construct the 4D joint distribution P[x1, y1, x2, y2] of 1st-half goals (x1, y1)
    and 2nd-half goals (x2, y2), conditioned on the full-time score matrix M_ft
    so that sum_{x1+x2=x, y1+y2=y} P[x1, y1, x2, y2] == M_ft[x, y] identically.
    """
    if gammas is None:
        gammas = DEFAULT_GAMMAS

    if M_ft is None:
        M_ft = score_matrix(lam_h, lam_a, rho, max_goals=10)

    max_goals = M_ft.shape[0] - 1

    # 1. First half bivariate distribution with half-time Dixon-Coles correlation
    lam_h1 = max(float(lam_h) * pi_h, 1e-6)
    lam_a1 = max(float(lam_a) * pi_a, 1e-6)
    p_ht_joint = score_matrix(lam_h1, lam_a1, rho * 0.5, max_goals=max_goals)

    # 2. Second half state-dependent distributions
    lam_h2_base = max(float(lam_h) * (1.0 - pi_h), 1e-6)
    lam_a2_base = max(float(lam_a) * (1.0 - pi_a), 1e-6)

    p_2h_by_state = {}
    for s in [-1, 0, 1]:
        adj_h = gammas["home"].get(s, 1.0)
        adj_a = gammas["away"].get(s, 1.0)
        p_h2 = _poisson_pmf_vec(lam_h2_base * adj_h, max_g=max_goals)
        p_a2 = _poisson_pmf_vec(lam_a2_base * adj_a, max_g=max_goals)
        p_2h_by_state[s] = np.outer(p_h2, p_a2)

    # 3. Build unconditioned 4D tensor for valid (x1+x2 <= max_goals, y1+y2 <= max_goals)
    tensor = np.zeros((max_goals + 1, max_goals + 1, max_goals + 1, max_goals + 1), dtype=np.float64)
    ft_marginal = np.zeros((max_goals + 1, max_goals + 1), dtype=np.float64)

    for x1 in range(max_goals + 1):
        for y1 in range(max_goals + 1):
            p1 = p_ht_joint[x1, y1]
            if p1 <= 0.0:
                continue
            s = 1 if x1 > y1 else (-1 if x1 < y1 else 0)
            p2_mat = p_2h_by_state[s]

            max_x2 = max_goals - x1
            max_y2 = max_goals - y1
            sub_p2 = p2_mat[: max_x2 + 1, : max_y2 + 1]
            joint_slice = p1 * sub_p2
            tensor[x1, y1, : max_x2 + 1, : max_y2 + 1] = joint_slice
            ft_marginal[x1:, y1:] += joint_slice

    # 4. Exact conditional projection onto M_ft[x, y] for 100% marginal coherence
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            denom = ft_marginal[x, y]
            target_p = M_ft[x, y]
            if denom > 1e-15 and target_p > 0.0:
                scale = target_p / denom
                for x1 in range(x + 1):
                    x2 = x - x1
                    y1_idx = np.arange(y + 1)
                    y2_idx = y - y1_idx
                    tensor[x1, y1_idx, x2, y2_idx] *= scale
            elif target_p > 0.0:
                # Fallback uniform split if denom underflowed
                tensor[0, 0, x, y] = target_p

    tot = tensor.sum()
    if tot > 0:
        tensor /= tot
    return tensor


def predict_ht_ft(
    lam_h: float,
    lam_a: float,
    rho: float = -0.05,
    M_ft: Optional[np.ndarray] = None,
    pi_h: float = DEFAULT_PI_H,
    pi_a: float = DEFAULT_PI_A,
    gammas: Optional[dict[str, dict[int, float]]] = None,
) -> dict[str, float]:
    """
    Compute 9-outcome HT/FT probabilities conditioned on half-time match state
    and projected onto the full-time score matrix M_ft for exact marginal coherence.
    """
    tensor = build_joint_ht_ft_tensor(
        lam_h=lam_h, lam_a=lam_a, rho=rho, M_ft=M_ft, pi_h=pi_h, pi_a=pi_a, gammas=gammas
    )
    max_g = tensor.shape[0] - 1
    htft_probs = {outcome: 0.0 for outcome in OUTCOMES_9}

    for x1 in range(max_g + 1):
        for y1 in range(max_g + 1):
            ht_res = "1" if x1 > y1 else ("2" if x1 < y1 else "X")
            for x2 in range(max_g + 1 - x1):
                for y2 in range(max_g + 1 - y1):
                    p = tensor[x1, y1, x2, y2]
                    if p <= 0.0:
                        continue
                    th = x1 + x2
                    ta = y1 + y2
                    ft_res = "1" if th > ta else ("2" if th < ta else "X")
                    htft_probs[f"{ht_res}/{ft_res}"] += float(p)

    total = sum(htft_probs.values())
    if total > 0:
        htft_probs = {k: v / total for k, v in htft_probs.items()}

    return htft_probs


def predict_half_markets(
    lam_h: float,
    lam_a: float,
    rho: float = -0.05,
    M_ft: Optional[np.ndarray] = None,
    pi_h: float = DEFAULT_PI_H,
    pi_a: float = DEFAULT_PI_A,
    gammas: Optional[dict[str, dict[int, float]]] = None,
) -> dict[str, Any]:
    """
    Derive coherent First-Half (1H), Second-Half (2H), and Highest-Scoring-Half markets
    from the 4D HT/FT joint tensor.
    """
    tensor = build_joint_ht_ft_tensor(
        lam_h=lam_h, lam_a=lam_a, rho=rho, M_ft=M_ft, pi_h=pi_h, pi_a=pi_a, gammas=gammas
    )
    # First-half score matrix M_1h[x1, y1]
    M_1h = tensor.sum(axis=(2, 3))
    # Second-half score matrix M_2h[x2, y2]
    M_2h = tensor.sum(axis=(0, 1))

    p_1h_1x2 = one_x_two(M_1h)
    p_2h_1x2 = one_x_two(M_2h)

    # Highest scoring half
    max_g = tensor.shape[0] - 1
    p_1h_more = 0.0
    p_equal_halves = 0.0
    p_2h_more = 0.0

    for x1 in range(max_g + 1):
        for y1 in range(max_g + 1):
            g1 = x1 + y1
            for x2 in range(max_g + 1 - x1):
                for y2 in range(max_g + 1 - y1):
                    p = tensor[x1, y1, x2, y2]
                    if p <= 0.0:
                        continue
                    g2 = x2 + y2
                    if g1 > g2:
                        p_1h_more += float(p)
                    elif g1 < g2:
                        p_2h_more += float(p)
                    else:
                        p_equal_halves += float(p)

    return {
        "M_1h": M_1h,
        "M_2h": M_2h,
        "ht_1x2": {"1": float(p_1h_1x2[0]), "X": float(p_1h_1x2[1]), "2": float(p_1h_1x2[2])},
        "2h_1x2": {"1": float(p_2h_1x2[0]), "X": float(p_2h_1x2[1]), "2": float(p_2h_1x2[2])},
        "ht_over_0.5": over_under(M_1h, 0.5)["over"],
        "ht_over_1.5": over_under(M_1h, 1.5)["over"],
        "ht_btts": both_teams_to_score(M_1h)["yes"],
        "highest_scoring_half": {
            "1H": p_1h_more,
            "Equal": p_equal_halves,
            "2H": p_2h_more,
        },
    }
