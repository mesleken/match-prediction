"""Empirical Bayes Bivariate Negative Binomial side models for Corners and Cards coupled with score-matrix competitiveness."""

from datetime import datetime
from typing import Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from slmodel.simulate.markets import match_competitiveness

# Empirical Süper Lig baseline parameters from 3,129 historical matches:
CORNERS_BASE_MEAN = 9.61
CORNERS_HOME_SHARE = 0.545
CORNERS_DISPERSION_ALPHA = 0.014

CARDS_BASE_MEAN = 4.79
CARDS_HOME_SHARE = 0.475
CARDS_DISPERSION_ALPHA = 0.074
RED_CARD_BASE_RATE = 0.251  # ~0.25 red cards per match (~22% of matches have >= 1 red card)


def _nbinom_from_mean_alpha(mu: float, alpha: float):
    """
    Construct scipy.stats.nbinom from mean and dispersion alpha.
    Variance = mu + alpha * mu^2.
    """
    mu = max(float(mu), 0.1)
    alpha = max(float(alpha), 1e-5)
    variance = mu + alpha * (mu ** 2)
    p = mu / variance
    n = (mu ** 2) / (variance - mu)
    return nbinom(n=n, p=p)


def _bivariate_nbinom_matrix(
    mu_h: float,
    mu_a: float,
    alpha: float,
    max_count: int = 20,
) -> np.ndarray:
    """
    Generate a joint (max_count+1, max_count+1) probability matrix for (Count_Home, Count_Away)
    using a shared Gamma match-tempo frailty (Bivariate Negative Binomial) integrated via
    5-point Gauss-Hermite log-normal quadrature.
    """
    sigma_shared = float(np.sqrt(np.log(1.0 + max(alpha, 1e-5))))
    nodes_raw, weights_raw = np.polynomial.hermite.hermgauss(5)
    nodes = nodes_raw * np.sqrt(2.0)
    weights = weights_raw / np.sqrt(np.pi)

    k = np.arange(max_count + 1, dtype=np.float64)
    from scipy.special import gammaln
    log_fact = gammaln(k + 1.0)

    M_joint = np.zeros((max_count + 1, max_count + 1), dtype=np.float64)
    for z, w in zip(nodes, weights):
        # Mean-preserving frailty multiplier
        frailty = float(np.exp(sigma_shared * z - 0.5 * (sigma_shared ** 2)))
        lh = max(mu_h * frailty, 1e-6)
        la = max(mu_a * frailty, 1e-6)
        ph = np.exp(k * np.log(lh) - lh - log_fact)
        pa = np.exp(k * np.log(la) - la - log_fact)
        ph /= ph.sum()
        pa /= pa.sum()
        M_joint += w * np.outer(ph, pa)

    s = M_joint.sum()
    if s > 0:
        M_joint /= s
    return M_joint


class SideMarketsModel:
    """
    Empirical Bayes team-specific estimator for Corners and Cards from historical team_match_stats.
    Strictly respects as_of cutoff to prevent any future leakage.
    """

    def __init__(
        self,
        half_life_years: float = 0.75,
        shrinkage_k: float = 6.0,
        in_season_boost: float = 3.0,
    ) -> None:
        self.half_life_years = half_life_years
        self.shrinkage_k = shrinkage_k
        self.in_season_boost = in_season_boost
        self.corner_for_mult: dict[int, float] = {}
        self.corner_against_mult: dict[int, float] = {}
        self.card_mult: dict[int, float] = {}
        self.base_corners: float = CORNERS_BASE_MEAN
        self.base_cards: float = CARDS_BASE_MEAN
        self.alpha_corners: float = CORNERS_DISPERSION_ALPHA
        self.alpha_cards: float = CARDS_DISPERSION_ALPHA
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame, as_of: Optional[datetime] = None) -> "SideMarketsModel":
        """
        Fit Empirical Bayes team corner and card multipliers from match history.
        Expects columns: home_team_id, away_team_id, kickoff_utc, status, and optional hc, ac, hy, ay, hr, ar.
        """
        valid = df[df["status"] == "played"].copy()
        if as_of is not None:
            as_of_ts = pd.to_datetime(as_of)
            if as_of_ts.tz is not None:
                as_of_ts = as_of_ts.tz_convert("UTC").tz_localize(None)
            k_s = pd.to_datetime(valid["kickoff_utc"])
            if hasattr(k_s.dt, "tz") and k_s.dt.tz is not None:
                k_s = k_s.dt.tz_convert("UTC").dt.tz_localize(None)
            valid = valid[k_s < as_of_ts]

        if len(valid) < 50 or "hc" not in valid.columns or "ac" not in valid.columns:
            self.is_fitted = True
            return self

        valid_c = valid.dropna(subset=["hc", "ac"]).copy()
        if len(valid_c) < 50:
            self.is_fitted = True
            return self

        dt_s = pd.to_datetime(valid_c["kickoff_utc"], utc=True)
        ref_dt = dt_s.max() if as_of is None else pd.to_datetime(as_of, utc=True)
        age_years = np.clip((ref_dt - dt_s).dt.total_seconds().to_numpy() / (365.25 * 86400.0), 0.0, None)
        w = np.power(0.5, age_years / max(self.half_life_years, 0.1))

        if self.in_season_boost != 1.0 and "season" in valid_c.columns and len(valid_c) >= 100:
            seasons_arr = valid_c["season"].astype(str).to_numpy()
            cur_s = seasons_arr[-1]
            w = w * np.where(seasons_arr == cur_s, self.in_season_boost, 1.0)

        hc = valid_c["hc"].to_numpy(dtype=np.float64)
        ac = valid_c["ac"].to_numpy(dtype=np.float64)
        tot_c = hc + ac
        self.base_corners = float(np.average(tot_c, weights=w))
        var_c = float(np.average((tot_c - self.base_corners) ** 2, weights=w))
        self.alpha_corners = float(np.clip((var_c - self.base_corners) / max(self.base_corners ** 2, 1.0), 0.005, 0.15))

        mean_hc = float(np.average(hc, weights=w))
        mean_ac = float(np.average(ac, weights=w))

        teams = set(valid_c["home_team_id"].unique()) | set(valid_c["away_team_id"].unique())
        h_ids = valid_c["home_team_id"].to_numpy()
        a_ids = valid_c["away_team_id"].to_numpy()

        for tid in teams:
            tid_int = int(tid)
            mh = h_ids == tid
            ma = a_ids == tid
            wt = float(np.sum(w[mh]) + np.sum(w[ma]))
            if wt > 0:
                # Corner For ratio relative to venue expectation
                cf_num = float(np.sum(w[mh] * (hc[mh] / max(mean_hc, 0.5))) + np.sum(w[ma] * (ac[ma] / max(mean_ac, 0.5))))
                ca_num = float(np.sum(w[mh] * (ac[mh] / max(mean_ac, 0.5))) + np.sum(w[ma] * (hc[ma] / max(mean_hc, 0.5))))
                shrink = wt / (wt + self.shrinkage_k)
                self.corner_for_mult[tid_int] = float(1.0 + shrink * ((cf_num / wt) - 1.0))
                self.corner_against_mult[tid_int] = float(1.0 + shrink * ((ca_num / wt) - 1.0))

        # Cards Empirical Bayes
        if "hy" in valid_c.columns and "ay" in valid_c.columns:
            hy = valid_c["hy"].fillna(0).to_numpy(dtype=np.float64)
            ay = valid_c["ay"].fillna(0).to_numpy(dtype=np.float64)
            hr = valid_c["hr"].fillna(0).to_numpy(dtype=np.float64) if "hr" in valid_c.columns else np.zeros_like(hy)
            ar = valid_c["ar"].fillna(0).to_numpy(dtype=np.float64) if "ar" in valid_c.columns else np.zeros_like(ay)
            h_cards = hy + hr
            a_cards = ay + ar
            tot_cards = h_cards + a_cards
            raw_base_cards = float(np.average(tot_cards, weights=w))
            cal_factor = 0.90 if self.in_season_boost > 1.0 else 1.00
            self.base_cards = float(cal_factor * raw_base_cards)
            var_k = float(np.average((tot_cards - raw_base_cards) ** 2, weights=w))
            self.alpha_cards = float(np.clip((var_k - raw_base_cards) / max(raw_base_cards ** 2, 1.0), 0.01, 0.25))

            mean_hk = max(float(np.average(h_cards, weights=w)), 0.5)
            mean_ak = max(float(np.average(a_cards, weights=w)), 0.5)
            for tid in teams:
                tid_int = int(tid)
                mh = h_ids == tid
                ma = a_ids == tid
                wt = float(np.sum(w[mh]) + np.sum(w[ma]))
                if wt > 0:
                    ck_num = float(np.sum(w[mh] * (h_cards[mh] / mean_hk)) + np.sum(w[ma] * (a_cards[ma] / mean_ak)))
                    shrink = wt / (wt + self.shrinkage_k)
                    self.card_mult[tid_int] = float(1.0 + shrink * ((ck_num / wt) - 1.0))

        self.is_fitted = True
        return self


def predict_corners(
    lam_h: float,
    lam_a: float,
    base_mean: float = CORNERS_BASE_MEAN,
    alpha: float = CORNERS_DISPERSION_ALPHA,
    lines: tuple[float, ...] = (7.5, 8.5, 9.5, 10.5, 11.5),
    home_team_id: Optional[int] = None,
    away_team_id: Optional[int] = None,
    side_model: Optional[SideMarketsModel] = None,
    M_ft: Optional[np.ndarray] = None,
    referee_profile: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Predict joint and marginal corner distributions using Bivariate Negative Binomial.
    Couples offensive goal intensity (lam_h, lam_a), Empirical Bayes team corner propensities,
    referee game-flow tempo, and score-matrix match competitiveness.
    """
    if side_model is not None and side_model.is_fitted:
        base_mean = side_model.base_corners
        alpha = side_model.alpha_corners
        h_for = side_model.corner_for_mult.get(home_team_id or -1, 1.0)
        h_against = side_model.corner_against_mult.get(home_team_id or -1, 1.0)
        a_for = side_model.corner_for_mult.get(away_team_id or -1, 1.0)
        a_against = side_model.corner_against_mult.get(away_team_id or -1, 1.0)
        team_h_factor = np.sqrt(h_for * a_against)
        team_a_factor = np.sqrt(a_for * h_against)
    else:
        team_h_factor = 1.0
        team_a_factor = 1.0

    total_goals = float(lam_h + lam_a)
    pace_factor = 1.0 + 0.065 * (total_goals - 2.75)

    # Close games generate more late-game attacking pressure and corners
    comp_boost = 1.0
    if M_ft is not None:
        p_close = match_competitiveness(M_ft)["p_close"]
        comp_boost = 1.0 + 0.05 * (p_close - 0.62)

    # Referee flow tempo impact (interrupted games dampen open-play corners)
    ref_pace_boost = 1.0
    if referee_profile:
        tau_r = float(referee_profile.get("whistle_tempo_index", 0.0))
        ref_pace_boost = float(1.0 - 0.045 * np.tanh(tau_r / 1.5))

    # Offensive territorial tilt from expected goal share
    goal_share_h = float(lam_h) / max(total_goals, 1e-6)
    # Blend structural home corner share (0.545) with match goal superiority share
    effective_share_h = 0.55 * CORNERS_HOME_SHARE + 0.45 * goal_share_h

    raw_tot = max(base_mean * pace_factor * comp_boost * ref_pace_boost, 4.0)
    expected_home = max(raw_tot * effective_share_h * team_h_factor, 1.5)
    expected_away = max(raw_tot * (1.0 - effective_share_h) * team_a_factor, 1.2)
    mu_corners = expected_home + expected_away

    dist = _nbinom_from_mean_alpha(mu_corners, alpha)

    ou_probs = {}
    for line in lines:
        k = int(np.floor(line))
        p_under = float(dist.cdf(k))
        ou_probs[f"over_{line}"] = 1.0 - p_under
        ou_probs[f"under_{line}"] = p_under

    # Joint Bivariate Corner Matrix for Corner 1X2 and Team Corner Lines
    M_corn = _bivariate_nbinom_matrix(expected_home, expected_away, alpha, max_count=20)
    p_home_more = float(np.tril(M_corn, -1).sum())
    p_equal_corn = float(np.trace(M_corn))
    p_away_more = float(np.triu(M_corn, 1).sum())

    h_marg = M_corn.sum(axis=1)
    a_marg = M_corn.sum(axis=0)
    idx_c = np.arange(len(h_marg))

    return {
        "expected_total": float(mu_corners),
        "expected_home": float(expected_home),
        "expected_away": float(expected_away),
        "lines": ou_probs,
        "corner_1x2": {
            "1": p_home_more,
            "X": p_equal_corn,
            "2": p_away_more,
        },
        "team_lines": {
            "home_over_4.5": float(h_marg[idx_c > 4.5].sum()),
            "home_over_5.5": float(h_marg[idx_c > 5.5].sum()),
            "away_over_3.5": float(a_marg[idx_c > 3.5].sum()),
            "away_over_4.5": float(a_marg[idx_c > 4.5].sum()),
        },
    }


def predict_cards(
    lam_h: float,
    lam_a: float,
    is_derby: bool = False,
    base_mean: float = CARDS_BASE_MEAN,
    alpha: float = CARDS_DISPERSION_ALPHA,
    lines: tuple[float, ...] = (3.5, 4.5, 5.5, 6.5),
    home_team_id: Optional[int] = None,
    away_team_id: Optional[int] = None,
    side_model: Optional[SideMarketsModel] = None,
    M_ft: Optional[np.ndarray] = None,
    referee_profile: Optional[dict[str, Any]] = None,
    booking_lines: tuple[float, ...] = (35.5, 45.5, 55.5, 65.5),
) -> dict[str, Any]:
    """
    Predict total booking cards (yellow + red), red card probability, and booking points
    using Negative Binomial coupled with score-matrix competitiveness, derby tension,
    and Empirical Bayes referee strictness.
    """
    if side_model is not None and side_model.is_fitted:
        base_mean = side_model.base_cards
        alpha = side_model.alpha_cards
        h_k = side_model.card_mult.get(home_team_id or -1, 1.0)
        a_k = side_model.card_mult.get(away_team_id or -1, 1.0)
    else:
        h_k = 1.0
        a_k = 1.0

    if M_ft is not None:
        comp_metrics = match_competitiveness(M_ft)
        comp_multiplier = 1.0 + 0.06 * (comp_metrics["p_close"] - 0.68)
    else:
        competitiveness = 1.0 - min(abs(lam_h - lam_a) / 3.0, 0.8)
        comp_multiplier = 1.0 + 0.05 * (competitiveness - 0.5)

    derby_multiplier = 1.15 if is_derby else 1.00
    raw_team_mult = 0.5 * (h_k + a_k)
    team_multiplier = 1.0 + 0.60 * (raw_team_mult - 1.0)

    # Referee strictness and whistle impact
    ref_card_mult = 1.0
    ref_red_mult = 1.0
    ref_home_bias = 0.0
    if referee_profile:
        raw_ref_mult = float(referee_profile.get("card_strictness_mult", 1.0))
        ref_card_mult = 1.0 + 0.50 * (raw_ref_mult - 1.0)
        ref_red_mult = float(referee_profile.get("red_card_mult", 1.0))
        ref_home_bias = float(referee_profile.get("home_card_bias", 0.0))

    mu_cards = max(base_mean * comp_multiplier * derby_multiplier * team_multiplier * ref_card_mult, 1.5)

    home_share = (CARDS_HOME_SHARE - 0.5 * ref_home_bias)
    expected_home_cards = mu_cards * home_share * (h_k / max(raw_team_mult, 0.1))
    expected_away_cards = mu_cards - expected_home_cards

    dist = _nbinom_from_mean_alpha(mu_cards, alpha)

    ou_probs = {}
    for line in lines:
        k = int(np.floor(line))
        p_under = float(dist.cdf(k))
        ou_probs[f"over_{line}"] = 1.0 - p_under
        ou_probs[f"under_{line}"] = p_under

    # Red card probability (Poisson/NB rare event scaled by card intensity, derby, and referee)
    expected_reds = RED_CARD_BASE_RATE * (mu_cards / CARDS_BASE_MEAN) * (1.15 if is_derby else 1.0) * ref_red_mult
    p_red_card = float(1.0 - np.exp(-expected_reds))

    # Booking points expectation (Yellow = 10 pts, Red = 25 pts)
    expected_yellows = max(mu_cards - expected_reds, 0.0)
    expected_booking_pts = 10.0 * expected_yellows + 25.0 * expected_reds
    expected_booking_pts_home = 10.0 * (expected_home_cards * 0.95) + 25.0 * (expected_reds * 0.45)
    expected_booking_pts_away = expected_booking_pts - expected_booking_pts_home

    # Booking points lines (35.5, 45.5, 55.5, 65.5)
    bpts_lines = {}
    for bl in booking_lines:
        equiv_k = int(np.floor(bl / 10.0))
        p_und = float(dist.cdf(equiv_k))
        bpts_lines[f"over_{bl}"] = float(np.clip(1.0 - p_und, 0.01, 0.99))
        bpts_lines[f"under_{bl}"] = float(np.clip(p_und, 0.01, 0.99))

    # Card 1X2 Market
    c_diff = expected_away_cards - expected_home_cards
    p_a_more = float(1.0 / (1.0 + np.exp(-1.4 * c_diff + 0.35)))
    p_h_more = float(1.0 / (1.0 + np.exp(1.4 * c_diff + 0.35)))
    p_eq = max(0.05, 1.0 - p_h_more - p_a_more)
    s_norm = p_h_more + p_eq + p_a_more
    card_1x2 = {
        "1": p_h_more / s_norm,
        "X": p_eq / s_norm,
        "2": p_a_more / s_norm,
    }

    return {
        "expected_total": float(mu_cards),
        "expected_home": float(expected_home_cards),
        "expected_away": float(expected_away_cards),
        "expected_red_cards": float(expected_reds),
        "p_red_card": p_red_card,
        "expected_booking_points": float(expected_booking_pts),
        "expected_booking_points_home": float(expected_booking_pts_home),
        "expected_booking_points_away": float(expected_booking_pts_away),
        "is_derby": is_derby,
        "lines": ou_probs,
        "booking_points_lines": bpts_lines,
        "card_1x2": card_1x2,
        "home_mult": float(h_k),
        "away_mult": float(a_k),
    }
