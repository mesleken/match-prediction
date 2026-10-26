"""Market derivation functions from joint score matrix."""

from typing import Any
import numpy as np


def one_x_two(M: np.ndarray) -> np.ndarray:
    """
    Compute 1X2 probabilities.
    Returns: np.ndarray([P(Home), P(Draw), P(Away)])
    """
    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())
    return np.array([p_home, p_draw, p_away], dtype=np.float64)


def calibrate_probabilities(
    probs: np.ndarray,
    temperature: float = 0.75,
) -> np.ndarray:
    """
    Apply temperature scaling to 1X2 probabilities to correct heavy favorite underconfidence.
    T < 1.0 sharpens logits and resolves the empirical residual (e = +0.077) observed on Süper Lig favorites.
    """
    if np.isclose(temperature, 1.0):
        return probs
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    exp_l = np.exp(logits - np.max(logits))
    return exp_l / np.sum(exp_l)


def draw_no_bet(M: np.ndarray) -> dict[str, float]:
    """Draw No Bet (DNB) probabilities (draw excluded)."""
    p1, px, p2 = one_x_two(M)
    denom = p1 + p2
    if denom <= 0:
        return {"1": 0.5, "2": 0.5}
    return {"1": p1 / denom, "2": p2 / denom}


def double_chance(M: np.ndarray) -> dict[str, float]:
    """Double Chance probabilities: 1X, 12, X2."""
    p1, px, p2 = one_x_two(M)
    return {
        "1X": p1 + px,
        "12": p1 + p2,
        "X2": px + p2,
    }


def over_under(M: np.ndarray, line: float) -> dict[str, float]:
    """Over / Under total goals probability for any line (e.g. 0.5, 1.5, 2.5, 3.5, 4.5)."""
    i, j = np.indices(M.shape)
    p_over = float(M[(i + j) > line].sum())
    return {"over": p_over, "under": 1.0 - p_over}


def both_teams_to_score(M: np.ndarray) -> dict[str, float]:
    """Both Teams To Score (BTTS / KG Var-Yok)."""
    p_yes = float(M[1:, 1:].sum())
    return {"yes": p_yes, "no": 1.0 - p_yes}


def team_total_goals(M: np.ndarray, line: float) -> dict[str, dict[str, float]]:
    """Individual team total goals over/under."""
    home_marginal = M.sum(axis=1)
    away_marginal = M.sum(axis=0)
    g = np.arange(len(home_marginal))
    p_home_over = float(home_marginal[g > line].sum())
    p_away_over = float(away_marginal[g > line].sum())
    return {
        "home": {"over": p_home_over, "under": 1.0 - p_home_over},
        "away": {"over": p_away_over, "under": 1.0 - p_away_over},
    }


def goal_range_probabilities(M: np.ndarray) -> dict[str, float]:
    """Total goal bands: 0-1, 2-3, 4-5, 6+ goals."""
    i, j = np.indices(M.shape)
    tot = i + j
    p_0_1 = float(M[tot <= 1].sum())
    p_2_3 = float(M[(tot >= 2) & (tot <= 3)].sum())
    p_4_5 = float(M[(tot >= 4) & (tot <= 5)].sum())
    p_6_plus = float(M[tot >= 6].sum())
    return {
        "0-1": p_0_1,
        "2-3": p_2_3,
        "4-5": p_4_5,
        "6+": p_6_plus,
    }


def clean_sheet_and_win_to_nil(M: np.ndarray) -> dict[str, float]:
    """Clean sheet and Win-to-Nil probabilities derived from matrix marginals."""
    home_cs = float(M[:, 0].sum())  # Away scores 0
    away_cs = float(M[0, :].sum())  # Home scores 0
    home_win_to_nil = float(M[1:, 0].sum())
    away_win_to_nil = float(M[0, 1:].sum())
    return {
        "home_clean_sheet": home_cs,
        "away_clean_sheet": away_cs,
        "home_win_to_nil": home_win_to_nil,
        "away_win_to_nil": away_win_to_nil,
    }


def expected_goals_and_variance(M: np.ndarray) -> dict[str, float]:
    """Compute exact moments (mean, variance, covariance, correlation) of score matrix M."""
    i, j = np.indices(M.shape, dtype=np.float64)
    ex_h = float(np.sum(i * M))
    ex_a = float(np.sum(j * M))
    var_h = float(np.sum(((i - ex_h) ** 2) * M))
    var_a = float(np.sum(((j - ex_a) ** 2) * M))
    cov_ha = float(np.sum((i - ex_h) * (j - ex_a) * M))
    corr_ha = cov_ha / max(np.sqrt(var_h * var_a), 1e-12)
    return {
        "mean_home": ex_h,
        "mean_away": ex_a,
        "mean_total": ex_h + ex_a,
        "var_home": var_h,
        "var_away": var_a,
        "cov_home_away": cov_ha,
        "corr_home_away": float(corr_ha),
    }


def match_competitiveness(M: np.ndarray) -> dict[str, float]:
    """
    Compute match closeness metrics from score matrix M:
    - p_close: P(|X - Y| <= 1) (match within 1 goal)
    - p_draw: P(X = Y)
    - entropy_1x2: Normalized Shannon entropy of 1X2 distribution in [0, 1].
    """
    i, j = np.indices(M.shape)
    diff = np.abs(i - j)
    p_close = float(M[diff <= 1].sum())
    p1x2 = np.clip(one_x_two(M), 1e-12, 1.0)
    entropy = float(-np.sum(p1x2 * np.log(p1x2)) / np.log(3.0))
    return {
        "p_close": p_close,
        "p_draw": float(p1x2[1]),
        "entropy_1x2": entropy,
    }


def asian_handicap(M: np.ndarray, line: float) -> dict[str, float]:
    """
    Asian Handicap settlement distribution for home team with handicap line.
    Handles whole lines (e.g. 0, -1, +1), half lines (e.g. -0.5, +0.5),
    and quarter lines (e.g. -0.25, -0.75, +0.25, +0.75).

    Returns:
        dict with probabilities for:
        full_win, half_win, push, half_loss, full_loss.
    """
    i, j = np.indices(M.shape)
    diff = (i - j).astype(np.float64)

    is_quarter = (abs(line) * 4) % 2 != 0
    subs = [line - 0.25, line + 0.25] if is_quarter else [line]

    res = np.zeros_like(diff)
    for s in subs:
        v = diff + s
        res += np.sign(v) * 1.0
    res = res / len(subs)  # outcome values: +1, +0.5, 0, -0.5, -1

    return {
        "full_win": float(M[res == 1.0].sum()),
        "half_win": float(M[res == 0.5].sum()),
        "push": float(M[res == 0.0].sum()),
        "half_loss": float(M[res == -0.5].sum()),
        "full_loss": float(M[res == -1.0].sum()),
    }


def most_likely_scores(M: np.ndarray, top_n: int = 10, coherent_top1: bool = False) -> list[dict[str, Any]]:
    """
    Return top N most probable exact scorelines.
    When coherent_top1 is True on a realistic match score matrix, ensures the #1 exact scoreline
    is coherent with the match's most probable 1X2 outcome and goal environment.
    """
    scan_n = max(top_n, 12) if coherent_top1 else top_n
    idx = np.argsort(M.ravel())[::-1][:scan_n]
    results = []
    for flat_i in idx:
        h, a = np.unravel_index(flat_i, M.shape)
        results.append({
            "score": f"{h}-{a}",
            "home_goals": int(h),
            "away_goals": int(a),
            "probability": float(M[h, a]),
        })

    if coherent_top1 and len(results) >= 3 and M.shape[0] >= 5 and float(M.max()) < 0.35:
        p1x2 = one_x_two(M)
        fav_idx = int(np.argmax(p1x2))
        p_ov25 = over_under(M, 2.5)["over"]
        p_btts = both_teams_to_score(M)["yes"]

        best_idx = None
        for i, r in enumerate(results):
            h, a = r["home_goals"], r["away_goals"]
            out_i = 0 if h > a else (1 if h == a else 2)
            if out_i != fav_idx:
                continue
            if fav_idx in (0, 2):
                if p_ov25 >= 0.50 and p_btts >= 0.52:
                    if (h + a) > 2.5 and h > 0 and a > 0:
                        best_idx = i
                        break
                elif p_ov25 < 0.50:
                    if (h + a) <= 2.5:
                        best_idx = i
                        break
            else:
                best_idx = i
                break

        if best_idx is None:
            for i, r in enumerate(results):
                h, a = r["home_goals"], r["away_goals"]
                out_i = 0 if h > a else (1 if h == a else 2)
                if out_i == fav_idx:
                    best_idx = i
                    break

        if best_idx is not None and best_idx > 0:
            best_coherent = results.pop(best_idx)
            results.insert(0, best_coherent)

    return results[:top_n]


def kelly_stake(
    p_model: float,
    odds: float,
    p_std: float = 0.0,
    fraction: float = 0.25,
    max_stake: float = 0.05,
) -> dict[str, float]:
    """
    Baker & McHale (2013) Uncertainty-Adjusted Fractional Kelly Criterion.
    Shrinks recommended stake when posterior uncertainty (p_std) of the predicted probability is high.
    """
    p = float(np.clip(p_model, 1e-6, 1.0 - 1e-6))
    o = float(odds)
    edge = p * o - 1.0
    if o <= 1.0 or edge <= 0.0:
        return {"edge": float(edge), "raw_kelly": 0.0, "robust_kelly": 0.0, "uncertainty_shrinkage": 1.0}

    b = o - 1.0
    raw_k = edge / b

    # Baker-McHale shrinkage factor for parameter estimation variance sigma_p^2
    var_p = float(p_std ** 2)
    shrinkage = 1.0 / (1.0 + ((o / b) ** 2) * (var_p / max(p * (1.0 - p), 1e-6)))
    robust_k = min(float(fraction * raw_k * shrinkage), max_stake)

    return {
        "edge": float(edge),
        "raw_kelly": float(raw_k),
        "robust_kelly": float(robust_k),
        "uncertainty_shrinkage": float(shrinkage),
    }
