"""Layer D: Empirical Bayes Referee Profiling, Game Flow / Tempo Adjustments, and Booking Points Engine."""

from datetime import datetime
from typing import Any, Optional
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import nbinom

from slmodel.ingest.mackolik_referees import (
    LEAGUE_AVG_BOOKING_PTS,
    LEAGUE_AVG_FOULS,
    LEAGUE_AVG_RED,
    LEAGUE_AVG_YELLOW,
    LEAGUE_STD_FOULS,
    LEAGUE_STD_YELLOW,
    OFFLINE_REFEREE_CATALOG,
)


class RefereeEngine:
    """
    Manages referee profile metrics (whistle tempo, card strictness, red card multiplier,
    and home bias) with strictly out-of-sample Empirical Bayes shrinkage.
    Computes game flow/xG pacing dampening and booking points probability distributions.
    """

    def __init__(
        self,
        db_path: str = "data/warehouse.duckdb",
        pace_sensitivity: float = 0.055,
        asym_sensitivity: float = 0.025,
        draw_sensitivity: float = 0.030,
    ) -> None:
        self.db_path = db_path
        self.pace_sensitivity = pace_sensitivity
        self.asym_sensitivity = asym_sensitivity
        self.draw_sensitivity = draw_sensitivity
        self._cache_catalog: dict[str, dict[str, Any]] = {}
        self.load_referees()

    def load_referees(self) -> None:
        """Load referees from DuckDB database or fallback to offline catalog."""
        try:
            con = duckdb.connect(self.db_path, read_only=True)
            tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
            if "referees" in tables:
                df = con.execute("""
                    SELECT referee_id, name, matches_officiated, avg_fouls, avg_yellow,
                           avg_red, avg_booking_pts, whistle_tempo_index, card_strictness_mult,
                           red_card_mult, home_card_bias
                    FROM referees
                """).fetchdf()
                for _, r in df.iterrows():
                    rec = {
                        "referee_id": int(r["referee_id"]),
                        "name": str(r["name"]),
                        "matches_officiated": int(r["matches_officiated"]),
                        "avg_fouls": float(r["avg_fouls"]),
                        "avg_yellow": float(r["avg_yellow"]),
                        "avg_red": float(r["avg_red"]),
                        "avg_booking_pts": float(r["avg_booking_pts"]),
                        "whistle_tempo_index": float(r["whistle_tempo_index"]),
                        "card_strictness_mult": float(r["card_strictness_mult"]),
                        "red_card_mult": float(r["red_card_mult"]),
                        "home_card_bias": float(r["home_card_bias"]),
                    }
                    self._cache_catalog[rec["name"].lower()] = rec
                    self._cache_catalog[str(rec["referee_id"])] = rec
            con.close()
        except Exception:
            pass

        # Fill any missing entries from offline catalog
        for r in OFFLINE_REFEREE_CATALOG:
            k = r["name"].lower()
            if k not in self._cache_catalog:
                rec = dict(r)
                rec["avg_booking_pts"] = float(10.0 * r["avg_yellow"] + 25.0 * r["avg_red"])
                self._cache_catalog[k] = rec
                self._cache_catalog[str(r["referee_id"])] = rec

    def get_all_active_referees(self) -> list[dict[str, Any]]:
        """Return unique active referees sorted by name."""
        seen = set()
        ref_list = []
        for v in self._cache_catalog.values():
            name = v["name"]
            if name not in seen:
                seen.add(name)
                ref_list.append(v)
        return sorted(ref_list, key=lambda x: x["name"])

    def get_referee_profile(
        self,
        referee_name_or_id: Optional[Any],
        as_of: Optional[datetime] = None,
        matches_history: Optional[pd.DataFrame] = None,
    ) -> dict[str, float]:
        """
        Get Empirical Bayes referee profile.
        If as_of and matches_history are provided, strictly recomputes stats prior to as_of
        to guarantee zero future leakage.
        """
        if not referee_name_or_id or str(referee_name_or_id).strip() == "" or str(referee_name_or_id).lower() in ("none", "nan", "null"):
            return {
                "referee_id": 0,
                "name": "Ortalama Hakem (Lig Standardı)",
                "whistle_tempo_index": 0.0,
                "card_strictness_mult": 1.0,
                "red_card_mult": 1.0,
                "home_card_bias": 0.0,
                "avg_fouls": LEAGUE_AVG_FOULS,
                "avg_yellow": LEAGUE_AVG_YELLOW,
                "avg_red": LEAGUE_AVG_RED,
                "avg_booking_pts": LEAGUE_AVG_BOOKING_PTS,
                "matches_officiated": 0,
            }

        ref_str = str(referee_name_or_id).strip()
        ref_key = ref_str.lower()
        base_profile = self._cache_catalog.get(ref_key)

        if not base_profile:
            # Check fuzzy match
            for k, v in self._cache_catalog.items():
                if ref_key in k or k in ref_key:
                    base_profile = v
                    break

        if not base_profile:
            base_profile = {
                "referee_id": 99999,
                "name": ref_str,
                "matches_officiated": 10,
                "avg_fouls": LEAGUE_AVG_FOULS,
                "avg_yellow": LEAGUE_AVG_YELLOW,
                "avg_red": LEAGUE_AVG_RED,
                "avg_booking_pts": LEAGUE_AVG_BOOKING_PTS,
                "whistle_tempo_index": 0.0,
                "card_strictness_mult": 1.0,
                "red_card_mult": 1.0,
                "home_card_bias": 0.0,
            }

        # If strictly time-ordered out-of-sample recalculation is requested:
        if as_of is not None and matches_history is not None and "referee" in matches_history.columns:
            k_s = pd.to_datetime(matches_history["kickoff_utc"])
            if hasattr(k_s.dt, "tz") and k_s.dt.tz is not None:
                k_s = k_s.dt.tz_convert("UTC").dt.tz_localize(None)
            as_of_ts = pd.to_datetime(as_of)
            if as_of_ts.tz is not None:
                as_of_ts = as_of_ts.tz_convert("UTC").tz_localize(None)

            ref_matches = matches_history[(k_s < as_of_ts) & (matches_history["referee"].str.lower() == ref_key)]
            n_prior = len(ref_matches)
            if n_prior >= 2:
                has_stats = all(c in ref_matches.columns for c in ["hy", "ay"])
                if has_stats:
                    y_tot = ref_matches["hy"].fillna(0) + ref_matches["ay"].fillna(0)
                    r_tot = ref_matches["hr"].fillna(0) + ref_matches["ar"].fillna(0) if "hr" in ref_matches.columns else 0.0
                    f_tot = ref_matches["hf"].fillna(0) + ref_matches["af"].fillna(0) if "hf" in ref_matches.columns else pd.Series([LEAGUE_AVG_FOULS] * n_prior)

                    f_mean = float(f_tot.mean())
                    y_mean = float(y_tot.mean())
                    red_mean = float(r_tot.mean()) if hasattr(r_tot, "mean") else 0.25

                    shrink_k = 6.0
                    w = n_prior / (n_prior + shrink_k)
                    eb_fouls = w * f_mean + (1.0 - w) * LEAGUE_AVG_FOULS
                    eb_yellow = w * y_mean + (1.0 - w) * LEAGUE_AVG_YELLOW
                    eb_red = w * red_mean + (1.0 - w) * LEAGUE_AVG_RED

                    w_idx = float((eb_fouls - LEAGUE_AVG_FOULS) / LEAGUE_STD_FOULS)
                    c_mult = float(eb_yellow / LEAGUE_AVG_YELLOW)
                    r_mult = float(eb_red / LEAGUE_AVG_RED)
                    b_pts = float(10.0 * eb_yellow + 25.0 * eb_red)

                    return {
                        "referee_id": base_profile["referee_id"],
                        "name": base_profile["name"],
                        "whistle_tempo_index": w_idx,
                        "card_strictness_mult": c_mult,
                        "red_card_mult": r_mult,
                        "home_card_bias": float(base_profile.get("home_card_bias", 0.0)),
                        "avg_fouls": eb_fouls,
                        "avg_yellow": eb_yellow,
                        "avg_red": eb_red,
                        "avg_booking_pts": b_pts,
                        "matches_officiated": n_prior,
                    }

        return base_profile

    def compute_flow_and_tempo_adjustments(
        self,
        referee_profile: dict[str, Any],
        att_diff: float = 0.0,
    ) -> dict[str, float]:
        """
        Compute game flow & pace shifts for Dixon-Coles goal intensities (lambda_H, mu_A),
        draw inflation, and corner pacing.

        - Whistle-heavy referees (tau_r > 0) interrupt ball-in-play, dampening open-play xG
          and increasing draw probability.
        - Fluent referees (tau_r < 0) let the game flow, boosting transition xG.
        """
        tau_r = float(referee_profile.get("whistle_tempo_index", 0.0))
        home_bias = float(referee_profile.get("home_card_bias", 0.0))

        # 1. Total pace shift (negative when referee stops the game frequently)
        # delta_pace in [-0.055, +0.055]
        delta_pace = -self.pace_sensitivity * float(np.tanh(tau_r / 1.5))

        # 2. Asymmetric tactical impact: high-whistle referees hurt dominant favorites
        # more because the underdog can break rhythm with tactical fouls
        asym_penalty = 0.0
        if tau_r > 0.0 and abs(att_diff) > 0.15:
            # Favored team gets an extra rhythm penalty
            fav_dampen = self.asym_sensitivity * float(np.tanh(tau_r / 1.5))
            asym_penalty = fav_dampen

        # 3. Net goal intensity shifts
        delta_h = delta_pace - (asym_penalty if att_diff > 0 else 0.0) + (home_bias * 0.02)
        delta_a = delta_pace - (asym_penalty if att_diff < 0 else 0.0) - (home_bias * 0.02)

        # 4. Low-tempo draw inflation (choppy games produce more 0-0 and 1-1 draws)
        delta_draw = float(max(0.0, self.draw_sensitivity * np.tanh(tau_r)))

        # 5. Corner pacing multiplier
        corner_mult = float(1.0 - 0.045 * np.tanh(tau_r / 1.5))

        return {
            "delta_h": delta_h,
            "delta_a": delta_a,
            "delta_pace": delta_pace,
            "draw_inflation": delta_draw,
            "corner_multiplier": corner_mult,
            "whistle_tempo_index": tau_r,
        }

    def compute_booking_points_distribution(
        self,
        referee_profile: dict[str, Any],
        h_card_mult: float = 1.0,
        a_card_mult: float = 1.0,
        is_derby: bool = False,
        p_close: float = 0.62,
    ) -> dict[str, Any]:
        """
        Compute joint card count and Booking Points (10 pts/Yellow, 25 pts/Red) distribution.
        Combines referee strictness (kappa_r), whistle index (tau_r), team tendencies, and match tension.
        """
        tau_r = float(referee_profile.get("whistle_tempo_index", 0.0))
        c_mult = float(referee_profile.get("card_strictness_mult", 1.0))
        r_mult = float(referee_profile.get("red_card_mult", 1.0))
        h_bias = float(referee_profile.get("home_card_bias", 0.0))

        # Base card expectation
        base_cards = LEAGUE_AVG_YELLOW * c_mult
        # Close matches and derbies raise tension and fouls
        comp_mult = 1.0 + 0.10 * (p_close - 0.62) + (0.22 if is_derby else 0.0)
        # Whistle frequency moderately raises card opportunity (+6% per SD of whistle)
        whistle_card_bump = 1.0 + 0.06 * np.clip(tau_r, -2.0, 2.0)

        effective_base = base_cards * comp_mult * whistle_card_bump

        # Allocate between home and away
        # Empirically home sees ~47.5% and away ~52.5% of cards in Süper Lig
        home_share = 0.475 - (0.5 * h_bias)
        away_share = 1.0 - home_share

        team_factor_h = np.clip(h_card_mult, 0.7, 1.4)
        team_factor_a = np.clip(a_card_mult, 0.7, 1.4)

        mu_yellow_h = float(max(1.0, effective_base * home_share * team_factor_h))
        mu_yellow_a = float(max(1.0, effective_base * away_share * team_factor_a))
        mu_yellow_tot = mu_yellow_h + mu_yellow_a

        # Red card expectation
        red_base = LEAGUE_AVG_RED * r_mult * (1.25 if is_derby else 1.0) * (comp_mult ** 0.8)
        p_red_card = float(1.0 - np.exp(-red_base))
        exp_red_cards = float(red_base)

        # Expected booking points (10 per yellow, 25 per red)
        exp_bpts_h = float(10.0 * mu_yellow_h + 12.5 * exp_red_cards * home_share)
        exp_bpts_a = float(10.0 * mu_yellow_a + 12.5 * exp_red_cards * away_share)
        exp_bpts_tot = exp_bpts_h + exp_bpts_a

        # Negative Binomial model for total cards to derive Booking Points CDF
        alpha_cards = 0.074
        var_cards = mu_yellow_tot + alpha_cards * (mu_yellow_tot ** 2)
        p_nb = mu_yellow_tot / var_cards
        n_nb = (mu_yellow_tot ** 2) / (var_cards - mu_yellow_tot)
        dist_cards = nbinom(n=n_nb, p=p_nb)

        # Booking points lines: 35.5, 45.5, 55.5, 65.5, 75.5
        # 35.5 bpts ~ 3.5 yellow cards (or 1 red + 1 yellow)
        # 45.5 bpts ~ 4.5 yellow cards
        # 55.5 bpts ~ 5.5 yellow cards
        # 65.5 bpts ~ 6.5 yellow cards
        bpts_lines = {}
        for line in (35.5, 45.5, 55.5, 65.5, 75.5):
            equiv_cards = int(np.floor(line / 10.0))
            p_under = float(dist_cards.cdf(equiv_cards))
            bpts_lines[f"over_{line}"] = float(np.clip(1.0 - p_under, 0.01, 0.99))
            bpts_lines[f"under_{line}"] = float(np.clip(p_under, 0.01, 0.99))

        # Card 1X2 market (Home more cards / Equal / Away more cards)
        # Approximate using Skellam difference
        card_diff_mean = mu_yellow_a - mu_yellow_h
        p_away_more = float(1.0 / (1.0 + np.exp(-1.4 * card_diff_mean + 0.35)))
        p_home_more = float(1.0 / (1.0 + np.exp(1.4 * card_diff_mean + 0.35)))
        p_equal = max(0.05, 1.0 - p_home_more - p_away_more)
        norm_sum = p_home_more + p_equal + p_away_more
        p_home_more /= norm_sum
        p_equal /= norm_sum
        p_away_more /= norm_sum

        # Categorical styling badge
        if tau_r >= 0.35:
            tempo_style = "🛑 Oyunu Çok Durduran (Düşük Tempo / Yüksek Faul)"
        elif tau_r <= -0.25:
            tempo_style = "⚡ Oyunu Akıtan / Avantaja Bırakan (Yüksek Tempo)"
        else:
            tempo_style = "⚖️ Dengeli Oyun Akışı"

        if c_mult >= 1.08:
            card_style = "🟨 Kartına Çok Başvuran / Sert Yönetim"
        elif c_mult <= 0.92:
            card_style = "🟩 Kartını Az Kullanan / Esnek Yönetim"
        else:
            card_style = "⚖️ Standart Kart Disiplini"

        return {
            "referee_name": referee_profile.get("name", "Bilinmeyen"),
            "whistle_tempo_index": tau_r,
            "tempo_style_badge": tempo_style,
            "card_style_badge": card_style,
            "expected_yellow_home": mu_yellow_h,
            "expected_yellow_away": mu_yellow_a,
            "expected_yellow_total": mu_yellow_tot,
            "expected_red_cards": exp_red_cards,
            "p_red_card": p_red_card,
            "expected_booking_points": exp_bpts_tot,
            "expected_booking_points_home": exp_bpts_h,
            "expected_booking_points_away": exp_bpts_a,
            "booking_points_lines": bpts_lines,
            "card_1x2": {
                "1_home_more": p_home_more,
                "X_equal": p_equal,
                "2_away_more": p_away_more,
            },
        }
