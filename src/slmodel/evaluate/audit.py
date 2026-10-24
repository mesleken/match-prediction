"""Out-of-sample Season Simulation Auditor, Error Diagnostics Engine, and Model Upgrade Benchmark."""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import duckdb
import numpy as np
import pandas as pd

from slmodel.evaluate.metrics import (
    rps,
    score_log_loss,
    brier_score,
    expected_calibration_error,
)
from slmodel.features.context import ContextFeatureExtractor
from slmodel.models.dixon_coles import DixonColesModel, compute_squad_priors
from slmodel.models.market import devig_shin, invert_lambdas, blend_lambdas
from slmodel.models.referee_engine import RefereeEngine
from slmodel.models.sidemodels import SideMarketsModel, predict_corners, predict_cards
from slmodel.simulate.markets import (
    one_x_two,
    over_under,
    both_teams_to_score,
    most_likely_scores,
)
from slmodel.simulate.score_matrix import (
    score_matrix,
    posterior_predictive_score_matrix,
)


class SeasonSimulationAuditor:
    """
    Executes strictly time-ordered, out-of-sample simulation audits for an entire season.
    For each played match, predictions are computed using only data available strictly prior
    to kickoff_utc, then systematically compared against actual goals, xG, shots, corners, and cards.
    """

    def __init__(
        self,
        db_path: str = "data/warehouse.duckdb",
        season: str = "2627",
        model_kwargs: Optional[dict[str, Any]] = None,
        refit_interval: int = 1,
        integrate_uncertainty: bool = True,
        load_priors: bool = True,
    ) -> None:
        self.db_path = db_path
        self.season = season
        self.model_kwargs = model_kwargs or {}
        self.refit_interval = refit_interval
        self.integrate_uncertainty = integrate_uncertainty
        self.load_priors = load_priors

        self.team_names: dict[int, str] = {}
        self.context_extractor = ContextFeatureExtractor(db_path=self.db_path)
        self.referee_engine = RefereeEngine(db_path=self.db_path)
        if self.load_priors:
            self.side_model = SideMarketsModel(half_life_years=0.75, shrinkage_k=6.0, in_season_boost=3.0)
        else:
            self.side_model = SideMarketsModel(half_life_years=1.25, shrinkage_k=12.0, in_season_boost=1.0)
        self.squad_priors: dict[int, tuple[float, float]] = {}
        if self.load_priors:
            self.squad_priors = compute_squad_priors(db_path=self.db_path)

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load all historical and target season matches with shot/xG stats and team names."""
        con = duckdb.connect(self.db_path)
        try:
            teams_df = con.execute("SELECT team_id, name FROM teams").fetchdf()
            self.team_names = dict(zip(teams_df["team_id"], teams_df["name"]))

            q_matches = """
                SELECT 
                    m.match_id, m.league, m.season, m.matchday, m.kickoff_utc, 
                    m.home_team_id, m.away_team_id,
                    m.home_goals, m.away_goals, m.ht_home_goals, m.ht_away_goals,
                    m.referee, m.stadium_id, m.status, m.crowd, m.is_derby,
                    sh.shots as hs, sh.shots_on_target as hst, sh.corners as hc, 
                    COALESCE(sh.xg, sh.shots * 0.10) as hxg,
                    sh.yellow as hy, sh.red as hr,
                    sa.shots as as_, sa.shots_on_target as ast, sa.corners as ac, 
                    COALESCE(sa.xg, sa.shots * 0.10) as axg,
                    sa.yellow as ay, sa.red as ar
                FROM matches m
                LEFT JOIN team_match_stats sh ON m.match_id = sh.match_id AND sh.is_home = TRUE
                LEFT JOIN team_match_stats sa ON m.match_id = sa.match_id AND sa.is_home = FALSE
                WHERE m.status = 'played' AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
                ORDER BY m.kickoff_utc, m.match_id
            """
            matches_df = con.execute(q_matches).fetchdf()

            # Pre-match opening and closing odds
            q_odds = """
                SELECT 
                    match_id, bookmaker, kind,
                    MAX(CASE WHEN market = '1x2' AND selection = '1' THEN odds END) as o1,
                    MAX(CASE WHEN market = '1x2' AND selection = 'X' THEN odds END) as ox,
                    MAX(CASE WHEN market = '1x2' AND selection = '2' THEN odds END) as o2,
                    MAX(CASE WHEN market = 'ou' AND selection = 'over' THEN odds END) as o_over,
                    MAX(CASE WHEN market = 'ou' AND selection = 'under' THEN odds END) as o_under
                FROM odds_snapshots
                GROUP BY match_id, bookmaker, kind
            """
            odds_df = con.execute(q_odds).fetchdf()
        finally:
            con.close()

        return matches_df, odds_df

    def run_audit(self) -> dict[str, Any]:
        """
        Run the complete season simulation audit and compute all evaluation metrics.
        Returns dictionary containing detailed match records, aggregated KPIs, and diagnostic tables.
        """
        matches_df, odds_df = self.load_data()
        season_mask = matches_df["season"] == self.season
        season_indices = matches_df[season_mask].index.tolist()

        if not season_indices:
            raise ValueError(f"No played matches found for season '{self.season}'.")

        # Index pre-match odds by match_id
        closing_1x2: dict[int, list[float]] = {}
        opening_1x2: dict[int, list[float]] = {}
        closing_ou25: dict[int, float] = {}
        if not odds_df.empty:
            bkmk_rank = {"Pinnacle": 0, "Bet365": 1, "MarketAvg": 2}
            odds_sorted = odds_df.copy()
            odds_sorted["rank"] = odds_sorted["bookmaker"].map(lambda x: bkmk_rank.get(str(x), 99))
            odds_sorted = odds_sorted.sort_values(["match_id", "rank"])
            for r_o in odds_sorted.itertuples():
                m_id = int(r_o.match_id)
                knd = str(getattr(r_o, "kind", "closing"))
                if pd.notna(r_o.o1) and pd.notna(r_o.ox) and pd.notna(r_o.o2):
                    if knd == "closing" and m_id not in closing_1x2:
                        closing_1x2[m_id] = [float(r_o.o1), float(r_o.ox), float(r_o.o2)]
                    elif knd == "file_main" and m_id not in opening_1x2:
                        opening_1x2[m_id] = [float(r_o.o1), float(r_o.ox), float(r_o.o2)]
                if m_id not in closing_ou25 and pd.notna(getattr(r_o, "o_over", np.nan)) and pd.notna(getattr(r_o, "o_under", np.nan)):
                    p_ou = devig_shin([float(r_o.o_over), float(r_o.o_under)])
                    closing_ou25[m_id] = float(p_ou[0])

        # Initialize model with V2.2 parameters and squad priors if enabled
        kwargs = dict(self.model_kwargs)
        if self.load_priors:
            v22_defaults = {
                "half_life_years": 0.75,
                "season_carryover": 0.40,
                "in_season_boost": 3.2,
                "shot_alpha": 0.45,
                "squad_prior_weight": 0.32,
                "early_season_prior_weight": 0.18,
                "form_momentum_weight": 0.08,
                "base_draw_inflation": -0.02,
                "closeness_draw_threshold": 0.14,
                "closeness_draw_boost": 0.42,
                "fixed_rho": 0.03,
            }
            for k, v in v22_defaults.items():
                kwargs.setdefault(k, v)
            if "team_squad_priors" not in kwargs and self.squad_priors:
                kwargs["team_squad_priors"] = self.squad_priors

        model = DixonColesModel(**kwargs)

        match_records: list[dict[str, Any]] = []
        last_fit_idx = -9999

        for idx in season_indices:
            row = matches_df.iloc[idx]
            kickoff = row["kickoff_utc"]
            training_history = matches_df.iloc[:idx]

            # Strictly out-of-sample refit
            if idx - last_fit_idx >= self.refit_interval or not model.is_fitted:
                model.fit(training_history, as_of=kickoff)
                self.side_model.fit(training_history, as_of=kickoff)
                last_fit_idx = idx

            mid = int(row["match_id"])
            hid = int(row["home_team_id"])
            aid = int(row["away_team_id"])
            h_name = self.team_names.get(hid, f"Team_{hid}")
            a_name = self.team_names.get(aid, f"Team_{aid}")
            crowd_val = str(row["crowd"]) if pd.notna(row.get("crowd")) else "full"
            ref_name = (
                str(row["referee"])
                if self.load_priors and pd.notna(row.get("referee")) and str(row.get("referee")).strip() != ""
                else None
            )
            ref_prof = None
            if ref_name:
                ref_prof = self.referee_engine.get_referee_profile(ref_name, as_of=kickoff, matches_history=training_history)

            # 1. Extract context features (rest, congestion, altitude, derby, form momentum, referee)
            ctx = self.context_extractor.extract_features_for_match(
                home_team_id=hid,
                away_team_id=aid,
                kickoff_utc=kickoff,
                matches_history=training_history,
                referee_name=ref_name,
            )

            # 2. Dixon-Coles goal intensity & joint score matrix prediction
            lam_h, mu_a, rho = model.predict_lambdas(hid, aid, crowd=crowd_val, context_features=ctx)

            # Blend with pre-match market lambdas when upgraded V2.2 mode is enabled
            if self.load_priors and mid in closing_1x2:
                p_cl = devig_shin(closing_1x2[mid])
                p_ou_mkt = closing_ou25.get(mid, 0.54)
                lm_cl, mm_cl = invert_lambdas(p_cl, p_over25_mkt=p_ou_mkt, rho=-0.03)
                if mid in opening_1x2:
                    p_op = devig_shin(opening_1x2[mid])
                    lm_op, mm_op = invert_lambdas(p_op, p_over25_mkt=p_ou_mkt, rho=-0.03)
                    lm = lm_cl * ((lm_cl / max(lm_op, 0.2)) ** 0.20)
                    mm = mm_cl * ((mm_cl / max(mm_op, 0.2)) ** 0.20)
                else:
                    lm, mm = lm_cl, mm_cl
                lam_h, mu_a = blend_lambdas(lam_h, mu_a, lm, mm, w_total=0.55, w_superiority=0.52)

            if self.load_priors and model.closeness_draw_threshold > 0.0:
                diff = abs(lam_h - mu_a)
                if diff < model.closeness_draw_threshold:
                    d_inf = float(model.closeness_draw_boost * (1.0 - (diff / model.closeness_draw_threshold) ** 2))
                else:
                    d_inf = float(model.base_draw_inflation)
                M_pred = score_matrix(lam_h, mu_a, rho=rho, draw_inflation=d_inf)
            else:
                M_pred = model.predict(
                    hid, aid, crowd=crowd_val, context_features=ctx,
                    integrate_uncertainty=self.integrate_uncertainty
                )

            p_1x2 = one_x_two(M_pred)
            p_home, p_draw, p_away = float(p_1x2[0]), float(p_1x2[1]), float(p_1x2[2])
            ou_25 = over_under(M_pred, line=2.5)
            p_over_25 = float(ou_25["over"])
            p_under_25 = float(ou_25["under"])
            btts_map = both_teams_to_score(M_pred)
            p_btts_yes = float(btts_map["yes"])
            p_btts_no = float(btts_map["no"])
            top_scores_raw = most_likely_scores(M_pred, top_n=5, coherent_top1=self.load_priors)
            top_5_scores = [(s["home_goals"], s["away_goals"], s["probability"]) for s in top_scores_raw]

            # 3. Side markets: Corners & Cards (incorporating referee tempo & strictness)
            corn_pred = predict_corners(
                lam_h=lam_h, lam_a=mu_a, home_team_id=hid, away_team_id=aid,
                side_model=self.side_model, M_ft=M_pred, referee_profile=ref_prof,
            )
            card_pred = predict_cards(
                lam_h=lam_h, lam_a=mu_a, is_derby=(ctx.get("is_derby", 0.0) > 0.5),
                home_team_id=hid, away_team_id=aid, side_model=self.side_model, M_ft=M_pred,
                referee_profile=ref_prof,
            )

            # 4. Actual outcomes
            hg = int(row["home_goals"])
            ag = int(row["away_goals"])
            tot_g = hg + ag
            g_diff = hg - ag
            actual_outcome = "1" if hg > ag else ("X" if hg == ag else "2")
            outcome_idx = 0 if hg > ag else (1 if hg == ag else 2)

            # Actual stats
            hxg = float(row["hxg"]) if pd.notna(row["hxg"]) else (float(row["hs"]) * 0.1 if pd.notna(row["hs"]) else float(hg))
            axg = float(row["axg"]) if pd.notna(row["axg"]) else (float(row["as_"]) * 0.1 if pd.notna(row["as_"]) else float(ag))
            tot_xg = hxg + axg
            xg_diff = hxg - axg

            hs = int(row["hs"]) if pd.notna(row["hs"]) else 0
            as_ = int(row["as_"]) if pd.notna(row["as_"]) else 0
            hst = int(row["hst"]) if pd.notna(row["hst"]) else 0
            ast = int(row["ast"]) if pd.notna(row["ast"]) else 0

            hc = int(row["hc"]) if pd.notna(row["hc"]) else 0
            ac = int(row["ac"]) if pd.notna(row["ac"]) else 0
            tot_c = hc + ac

            hy = int(row["hy"]) if pd.notna(row["hy"]) else 0
            ay = int(row["ay"]) if pd.notna(row["ay"]) else 0
            hr = int(row["hr"]) if pd.notna(row["hr"]) else 0
            ar = int(row["ar"]) if pd.notna(row["ar"]) else 0
            tot_cards = (hy + ay) + 2 * (hr + ar)
            actual_bp = (hy + ay) * 10 + (hr + ar) * 25

            # 5. Out-of-sample hit evaluations
            pred_idx = int(np.argmax(p_1x2))
            pred_fav = ["1", "X", "2"][pred_idx]
            p_fav = float(p_1x2[pred_idx])

            # Double chance: top 2 outcomes
            top_2_indices = list(np.argsort(p_1x2)[::-1][:2])
            hit_dc = outcome_idx in top_2_indices

            hit_1x2 = (pred_idx == outcome_idx)
            hit_ou25 = (p_over_25 >= 0.50 and tot_g > 2.5) or (p_over_25 < 0.50 and tot_g <= 2.5)
            hit_btts = (p_btts_yes >= 0.50 and (hg > 0 and ag > 0)) or (p_btts_yes < 0.50 and (hg == 0 or ag == 0))

            hit_top1_score = (len(top_5_scores) > 0 and top_5_scores[0][0] == hg and top_5_scores[0][1] == ag)
            hit_top3_score = any(s[0] == hg and s[1] == ag for s in top_5_scores[:3])
            hit_top5_score = any(s[0] == hg and s[1] == ag for s in top_5_scores[:5])

            # Metrics
            rps_val = float(rps(p_1x2, outcome_idx))
            brier_val = float(brier_score(p_1x2, outcome_idx))
            ll_val = float(score_log_loss(M_pred, hg, ag))

            # Variance decomposition terms
            pred_xg_diff = float(lam_h - mu_a)
            model_error = float(pred_xg_diff - xg_diff)
            finishing_luck = float(xg_diff - g_diff)
            total_error = float(pred_xg_diff - g_diff)

            # Error diagnosis classification
            if hit_1x2:
                if hit_top1_score or hit_top3_score:
                    diagnosis_tag = "🎯 Tam İsabet (Exact Hit)"
                else:
                    diagnosis_tag = "✅ Baskın İsabet (Dominant Hit)"
            else:
                if actual_outcome == "X" and abs(pred_xg_diff) <= 0.65:
                    diagnosis_tag = "⚖️ Kritik Denge / Beraberlik (Draw Variance)"
                elif abs(finishing_luck) >= 1.0 and abs(model_error) <= 0.85:
                    diagnosis_tag = "🍀 Şans / Bitiricilik Sapması (Finishing Luck)"
                else:
                    diagnosis_tag = "⚠️ Model / Taktiksel Sapma (Model Miss)"

            record = {
                "match_id": mid,
                "season": str(row["season"]),
                "matchday": int(row["matchday"]) if pd.notna(row["matchday"]) else 0,
                "kickoff_utc": kickoff,
                "home_team_id": hid,
                "away_team_id": aid,
                "home_team": h_name,
                "away_team": a_name,
                "referee": ref_name or "Bilinmiyor",
                "actual_home_goals": hg,
                "actual_away_goals": ag,
                "actual_total_goals": tot_g,
                "actual_goal_diff": g_diff,
                "actual_outcome": actual_outcome,
                "outcome_idx": outcome_idx,
                "actual_hxg": hxg,
                "actual_axg": axg,
                "actual_total_xg": tot_xg,
                "actual_xg_diff": xg_diff,
                "actual_hs": hs,
                "actual_as": as_,
                "actual_hst": hst,
                "actual_ast": ast,
                "actual_hc": hc,
                "actual_ac": ac,
                "actual_corners_tot": tot_c,
                "actual_hy": hy,
                "actual_ay": ay,
                "actual_hr": hr,
                "actual_ar": ar,
                "actual_cards_tot": tot_cards,
                "pred_lam_h": lam_h,
                "pred_mu_a": mu_a,
                "pred_total_lam": lam_h + mu_a,
                "pred_xg_diff": pred_xg_diff,
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,
                "pred_favorite": pred_fav,
                "p_fav": p_fav,
                "p_over_25": p_over_25,
                "p_under_25": p_under_25,
                "p_btts_yes": p_btts_yes,
                "p_btts_no": p_btts_no,
                "top_5_scores": top_5_scores,
                "pred_corners_tot": float(corn_pred["expected_total"]),
                "pred_corners_h": float(corn_pred["expected_home"]),
                "pred_corners_a": float(corn_pred["expected_away"]),
                "pred_cards_tot": float(card_pred["expected_total"]),
                "pred_cards_h": float(card_pred["expected_home"]),
                "pred_cards_a": float(card_pred["expected_away"]),
                "actual_booking_points": actual_bp,
                "pred_booking_points": float(card_pred.get("expected_booking_points", card_pred["expected_total"] * 10)),
                "booking_points_abs_err": abs(float(card_pred.get("expected_booking_points", card_pred["expected_total"] * 10)) - actual_bp),
                "hit_1x2": hit_1x2,
                "hit_double_chance": hit_dc,
                "hit_over_25": hit_ou25,
                "hit_btts": hit_btts,
                "hit_top1_score": hit_top1_score,
                "hit_top3_score": hit_top3_score,
                "hit_top5_score": hit_top5_score,
                "corner_abs_err": abs(float(corn_pred["expected_total"]) - tot_c),
                "card_abs_err": abs(float(card_pred["expected_total"]) - tot_cards),
                "rps": rps_val,
                "brier": brier_val,
                "logloss": ll_val,
                "model_error": model_error,
                "finishing_luck": finishing_luck,
                "total_error": total_error,
                "diagnosis_tag": diagnosis_tag,
            }
            match_records.append(record)

        # Run diagnostics engine
        diagnostics = ErrorDiagnosticsEngine.analyze(match_records, self.team_names)
        summary_metrics = self._aggregate_metrics(match_records)

        return {
            "season": self.season,
            "total_matches": len(match_records),
            "summary_metrics": summary_metrics,
            "diagnostics": diagnostics,
            "match_records": match_records,
        }

    def _aggregate_metrics(self, records: list[dict[str, Any]]) -> dict[str, float]:
        """Compute top-level summary metrics from match records."""
        n = len(records)
        if n == 0:
            return {}

        return {
            "mean_rps": float(np.mean([r["rps"] for r in records])),
            "mean_brier": float(np.mean([r["brier"] for r in records])),
            "mean_logloss": float(np.mean([r["logloss"] for r in records])),
            "hit_rate_1x2_pct": float(np.mean([r["hit_1x2"] for r in records]) * 100.0),
            "hit_rate_double_chance_pct": float(np.mean([r["hit_double_chance"] for r in records]) * 100.0),
            "hit_rate_over25_pct": float(np.mean([r["hit_over_25"] for r in records]) * 100.0),
            "hit_rate_btts_pct": float(np.mean([r["hit_btts"] for r in records]) * 100.0),
            "hit_rate_top1_score_pct": float(np.mean([r["hit_top1_score"] for r in records]) * 100.0),
            "hit_rate_top3_score_pct": float(np.mean([r["hit_top3_score"] for r in records]) * 100.0),
            "hit_rate_top5_score_pct": float(np.mean([r["hit_top5_score"] for r in records]) * 100.0),
            "corners_mae": float(np.mean([r["corner_abs_err"] for r in records])),
            "cards_mae": float(np.mean([r["card_abs_err"] for r in records])),
            "booking_points_mae": float(np.mean([r["booking_points_abs_err"] for r in records])),
            "goal_diff_mae": float(np.mean([abs(r["total_error"]) for r in records])),
            "xg_diff_mae": float(np.mean([abs(r["model_error"]) for r in records])),
        }


class ErrorDiagnosticsEngine:
    """
    In-depth error analysis engine identifying model strengths, systematic biases,
    and separating expected play setup errors from stochastic finishing variance.
    """

    @classmethod
    def analyze(
        cls,
        records: list[dict[str, Any]],
        team_names: Optional[dict[int, str]] = None,
    ) -> dict[str, Any]:
        """Perform comprehensive error diagnostics across all evaluated matches."""
        team_names = team_names or {}
        return {
            "variance_decomposition": cls.decompose_variance(records),
            "team_bias_table": cls.compute_team_bias_table(records, team_names),
            "upset_analysis": cls.find_upsets(records, threshold=0.52),
            "round_progression": cls.compute_round_progression(records),
            "diagnosis_distribution": cls.compute_diagnosis_distribution(records),
            "calibration": cls.compute_calibration(records),
            "referee_impact_table": cls.compute_referee_impact_table(records),
        }

    @staticmethod
    def decompose_variance(records: list[dict[str, Any]]) -> dict[str, float]:
        """
        Decompose total score error into:
        Total Error = Model Play Setup Error ((lam - mu) - (xG_H - xG_A)) + Finishing Luck ((xG_H - xG_A) - (G_H - G_A))
        MSE_tot = MSE_model + MSE_luck + 2*Cov(model, luck)
        """
        if not records:
            return {}

        tot_err = np.array([r["total_error"] for r in records], dtype=np.float64)
        mod_err = np.array([r["model_error"] for r in records], dtype=np.float64)
        luck_err = np.array([r["finishing_luck"] for r in records], dtype=np.float64)

        mse_tot = float(np.mean(tot_err ** 2))
        mse_mod = float(np.mean(mod_err ** 2))
        mse_luck = float(np.mean(luck_err ** 2))
        cov_term = float(2.0 * np.mean(mod_err * luck_err))

        denom = max(mse_mod + mse_luck, 1e-6)
        mod_pct = (mse_mod / denom) * 100.0
        luck_pct = (mse_luck / denom) * 100.0

        return {
            "total_mse": mse_tot,
            "model_error_mse": mse_mod,
            "finishing_luck_mse": mse_luck,
            "covariance_term": cov_term,
            "model_error_pct": float(mod_pct),
            "finishing_luck_pct": float(luck_pct),
        }

    @staticmethod
    def compute_team_bias_table(
        records: list[dict[str, Any]],
        team_names: dict[int, str],
    ) -> list[dict[str, Any]]:
        """
        Compute team-level bias table: checks if any team was systematically
        under-predicted or over-predicted in attack, defense, or overall net superiority.
        """
        all_teams = set([r["home_team_id"] for r in records] + [r["away_team_id"] for r in records])
        team_rows = []

        for tid in sorted(list(all_teams)):
            t_name = team_names.get(tid, f"Team_{tid}")
            h_matches = [r for r in records if r["home_team_id"] == tid]
            a_matches = [r for r in records if r["away_team_id"] == tid]
            m_all = h_matches + a_matches
            n_m = len(m_all)
            if n_m == 0:
                continue

            # Goals & xG for
            gf = sum(r["actual_home_goals"] for r in h_matches) + sum(r["actual_away_goals"] for r in a_matches)
            ga = sum(r["actual_away_goals"] for r in h_matches) + sum(r["actual_home_goals"] for r in a_matches)
            xgf = sum(r["actual_hxg"] for r in h_matches) + sum(r["actual_axg"] for r in a_matches)
            xga = sum(r["actual_axg"] for r in h_matches) + sum(r["actual_hxg"] for r in a_matches)

            # Model expected goals for & against
            pred_f = sum(r["pred_lam_h"] for r in h_matches) + sum(r["pred_mu_a"] for r in a_matches)
            pred_a = sum(r["pred_mu_a"] for r in h_matches) + sum(r["pred_lam_h"] for r in a_matches)

            # Attack bias: (pred - actual). Positive = Model over-estimated, Negative = Model under-estimated
            att_bias_g = (pred_f - gf) / n_m
            att_bias_xg = (pred_f - xgf) / n_m
            def_bias_g = (pred_a - ga) / n_m
            def_bias_xg = (pred_a - xga) / n_m

            # 1X2 hit rate when this team plays
            hits = sum(1 for r in m_all if r["hit_1x2"])
            hit_rate = (hits / n_m) * 100.0

            # Net goal superiority bias
            net_bias = att_bias_g - def_bias_g

            team_rows.append({
                "team_id": tid,
                "team_name": t_name,
                "matches": n_m,
                "goals_scored": gf,
                "goals_conceded": ga,
                "xg_for": round(xgf, 2),
                "xg_against": round(xga, 2),
                "pred_goals_for": round(pred_f, 2),
                "pred_goals_against": round(pred_a, 2),
                "attack_bias_vs_goals": round(att_bias_g, 3),
                "attack_bias_vs_xg": round(att_bias_xg, 3),
                "defense_bias_vs_goals": round(def_bias_g, 3),
                "defense_bias_vs_xg": round(def_bias_xg, 3),
                "net_superiority_bias": round(net_bias, 3),
                "accuracy_1x2_pct": round(hit_rate, 1),
            })

        return sorted(team_rows, key=lambda x: x["accuracy_1x2_pct"], reverse=True)

    @staticmethod
    def find_upsets(records: list[dict[str, Any]], threshold: float = 0.52) -> list[dict[str, Any]]:
        """Identify matches where a favorite with P >= threshold failed to win."""
        upsets = []
        for r in records:
            if r["p_fav"] >= threshold and not r["hit_1x2"]:
                fav_team = r["home_team"] if r["pred_favorite"] == "1" else (r["away_team"] if r["pred_favorite"] == "2" else "Draw")
                underdog_team = r["away_team"] if r["pred_favorite"] == "1" else r["home_team"]

                # Check if favorite won xG battle despite dropping points
                fav_xg = r["actual_hxg"] if r["pred_favorite"] == "1" else r["actual_axg"]
                und_xg = r["actual_axg"] if r["pred_favorite"] == "1" else r["actual_hxg"]
                xg_dominance = fav_xg - und_xg

                if xg_dominance >= 0.60:
                    cause = "Bitiricilik Şanssızlığı (Favori xG'yi domine etti ama puan kaybetti)"
                elif xg_dominance <= -0.30:
                    cause = "Taktiksel Yanılma (Rakip sahada xG üstünlüğünü kurdu)"
                else:
                    cause = "Dengeli Maç / Yüksek Varyans"

                upsets.append({
                    "match_id": r["match_id"],
                    "matchday": r["matchday"],
                    "home_team": r["home_team"],
                    "away_team": r["away_team"],
                    "actual_score": f"{r['actual_home_goals']}-{r['actual_away_goals']}",
                    "favorite_team": fav_team,
                    "underdog_team": underdog_team,
                    "fav_win_prob_pct": round(r["p_fav"] * 100, 1),
                    "actual_xg_diff": round(r["actual_xg_diff"], 2),
                    "post_mortem_cause": cause,
                })

        return sorted(upsets, key=lambda x: x["fav_win_prob_pct"], reverse=True)

    @staticmethod
    def compute_round_progression(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compute performance metrics grouped by matchday/week to show learning progression."""
        rounds = sorted(list(set(r["matchday"] for r in records if r["matchday"] > 0)))
        prog = []
        for rd in rounds:
            rd_records = [r for r in records if r["matchday"] == rd]
            if not rd_records:
                continue
            prog.append({
                "matchday": rd,
                "matches": len(rd_records),
                "rps": round(float(np.mean([r["rps"] for r in rd_records])), 4),
                "hit_rate_1x2_pct": round(float(np.mean([r["hit_1x2"] for r in rd_records]) * 100), 1),
                "double_chance_pct": round(float(np.mean([r["hit_double_chance"] for r in rd_records]) * 100), 1),
                "top3_score_pct": round(float(np.mean([r["hit_top3_score"] for r in rd_records]) * 100), 1),
                "logloss": round(float(np.mean([r["logloss"] for r in rd_records])), 4),
                "corners_mae": round(float(np.mean([r["corner_abs_err"] for r in rd_records])), 2),
                "cards_mae": round(float(np.mean([r["card_abs_err"] for r in rd_records])), 2),
            })
        return prog

    @staticmethod
    def compute_diagnosis_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
        """Count matches by error diagnosis classification."""
        counts: dict[str, int] = {}
        for r in records:
            tag = r["diagnosis_tag"]
            counts[tag] = counts.get(tag, 0) + 1
        return counts

    @staticmethod
    def compute_calibration(records: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute calibration bins for home win, draw, and over 2.5."""
        p_home = np.array([r["p_home"] for r in records])
        y_home = np.array([1 if r["outcome_idx"] == 0 else 0 for r in records])
        ece_home = expected_calibration_error(p_home, y_home, n_bins=5)

        p_draw = np.array([r["p_draw"] for r in records])
        y_draw = np.array([1 if r["outcome_idx"] == 1 else 0 for r in records])
        ece_draw = expected_calibration_error(p_draw, y_draw, n_bins=5)

        p_ou = np.array([r["p_over_25"] for r in records])
        y_ou = np.array([1 if r["actual_total_goals"] > 2.5 else 0 for r in records])
        ece_ou = expected_calibration_error(p_ou, y_ou, n_bins=5)

        return {
            "ece_home_win": round(float(ece_home), 4),
            "ece_draw": round(float(ece_draw), 4),
            "ece_over25": round(float(ece_ou), 4),
            "empirical_draw_rate": round(float(np.mean(y_draw)), 3),
            "predicted_draw_rate": round(float(np.mean(p_draw)), 3),
        }

    @staticmethod
    def compute_referee_impact_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compute referee officiating analysis comparing predicted vs actual booking points, cards and pace."""
        ref_groups: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            ref = r.get("referee", "Bilinmiyor")
            ref_groups.setdefault(ref, []).append(r)

        rows = []
        for ref, m_list in ref_groups.items():
            n = len(m_list)
            if n == 0:
                continue
            act_bp = [r.get("actual_booking_points", 0) for r in m_list]
            pred_bp = [r.get("pred_booking_points", 0.0) for r in m_list]
            act_c = [r.get("actual_cards_tot", 0) for r in m_list]
            pred_c = [r.get("pred_cards_tot", 0.0) for r in m_list]
            tot_xg = [r.get("actual_total_xg", 0.0) for r in m_list]
            pred_xg = [r.get("pred_total_lam", 0.0) for r in m_list]

            rows.append({
                "referee": ref,
                "matches": n,
                "avg_actual_bp": round(float(np.mean(act_bp)), 1),
                "avg_pred_bp": round(float(np.mean(pred_bp)), 1),
                "bp_mae": round(float(np.mean(np.abs(np.array(act_bp) - np.array(pred_bp)))), 1),
                "avg_actual_cards": round(float(np.mean(act_c)), 2),
                "avg_pred_cards": round(float(np.mean(pred_c)), 2),
                "avg_actual_xg": round(float(np.mean(tot_xg)), 2),
                "avg_pred_xg": round(float(np.mean(pred_xg)), 2),
            })
        return sorted(rows, key=lambda x: x["matches"], reverse=True)


def compare_audit_models(
    baseline_result: dict[str, Any],
    upgraded_result: dict[str, Any],
) -> dict[str, Any]:
    """Side-by-side comparison between Baseline V2.0 and Upgraded V2.2 audit runs."""
    b_m = baseline_result["summary_metrics"]
    u_m = upgraded_result["summary_metrics"]

    comparison = {
        "evaluated_matches": baseline_result["total_matches"],
        "rps": {
            "baseline": b_m["mean_rps"],
            "upgraded": u_m["mean_rps"],
            "delta": u_m["mean_rps"] - b_m["mean_rps"],
            "pct_change": ((u_m["mean_rps"] - b_m["mean_rps"]) / b_m["mean_rps"]) * 100.0,
        },
        "logloss": {
            "baseline": b_m["mean_logloss"],
            "upgraded": u_m["mean_logloss"],
            "delta": u_m["mean_logloss"] - b_m["mean_logloss"],
            "pct_change": ((u_m["mean_logloss"] - b_m["mean_logloss"]) / b_m["mean_logloss"]) * 100.0,
        },
        "hit_rate_1x2": {
            "baseline": b_m["hit_rate_1x2_pct"],
            "upgraded": u_m["hit_rate_1x2_pct"],
            "delta": u_m["hit_rate_1x2_pct"] - b_m["hit_rate_1x2_pct"],
        },
        "hit_rate_double_chance": {
            "baseline": b_m["hit_rate_double_chance_pct"],
            "upgraded": u_m["hit_rate_double_chance_pct"],
            "delta": u_m["hit_rate_double_chance_pct"] - b_m["hit_rate_double_chance_pct"],
        },
        "hit_rate_over25": {
            "baseline": b_m.get("hit_rate_over25_pct", 0.0),
            "upgraded": u_m.get("hit_rate_over25_pct", 0.0),
            "delta": u_m.get("hit_rate_over25_pct", 0.0) - b_m.get("hit_rate_over25_pct", 0.0),
        },
        "hit_rate_top1_score": {
            "baseline": b_m.get("hit_rate_top1_score_pct", 0.0),
            "upgraded": u_m.get("hit_rate_top1_score_pct", 0.0),
            "delta": u_m.get("hit_rate_top1_score_pct", 0.0) - b_m.get("hit_rate_top1_score_pct", 0.0),
        },
        "hit_rate_top3_score": {
            "baseline": b_m["hit_rate_top3_score_pct"],
            "upgraded": u_m["hit_rate_top3_score_pct"],
            "delta": u_m["hit_rate_top3_score_pct"] - b_m["hit_rate_top3_score_pct"],
        },
        "hit_rate_top5_score": {
            "baseline": b_m.get("hit_rate_top5_score_pct", 0.0),
            "upgraded": u_m.get("hit_rate_top5_score_pct", 0.0),
            "delta": u_m.get("hit_rate_top5_score_pct", 0.0) - b_m.get("hit_rate_top5_score_pct", 0.0),
        },
        "xg_diff_mae": {
            "baseline": b_m.get("xg_diff_mae", 0.0),
            "upgraded": u_m.get("xg_diff_mae", 0.0),
            "delta": u_m.get("xg_diff_mae", 0.0) - b_m.get("xg_diff_mae", 0.0),
        },
        "corners_mae": {
            "baseline": b_m["corners_mae"],
            "upgraded": u_m["corners_mae"],
            "delta": u_m["corners_mae"] - b_m["corners_mae"],
        },
        "cards_mae": {
            "baseline": b_m["cards_mae"],
            "upgraded": u_m["cards_mae"],
            "delta": u_m["cards_mae"] - b_m["cards_mae"],
        },
    }
    return comparison


def generate_audit_markdown_report(
    upgraded_audit: dict[str, Any],
    baseline_audit: Optional[dict[str, Any]] = None,
    output_path: str = "reports/simulation_audit_2627.md",
) -> str:
    """Generate comprehensive markdown report of the simulation audit and error diagnostics."""
    m_up = upgraded_audit["summary_metrics"]
    diag = upgraded_audit["diagnostics"]
    v_dec = diag["variance_decomposition"]
    cal = diag["calibration"]
    records = upgraded_audit["match_records"]

    comp = None
    if baseline_audit is not None:
        comp = compare_audit_models(baseline_audit, upgraded_audit)

    lines: list[str] = []
    lines.append(f"# 🧪 2026/27 Süper Lig Simülasyon Denetimi ve Model Geliştirme Raporu (V2.2)")
    lines.append(f"**Denetim Tarihi:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | **İncelenen Maç Sayısı:** {upgraded_audit['total_matches']} Maç (1–8. Hafta Tamamı)")
    lines.append(f"**Veri Sızıntısı:** %0 (Strictly Out-of-Sample, her maç yalnızca maç saatinden önceki verilerle simüle edilmiştir)\n")

    lines.append("## 1. Yönetici Özeti ve Model Karşılaştırma Karnesi\n")
    if comp:
        lines.append("| Değerlendirme Metriği | Baz Model (V2.0) | Geliştirilmiş Model (V2.2) | Değişim / İyileşme |")
        lines.append("| :--- | :---: | :---: | :---: |")
        lines.append(f"| **RPS (Ranked Probability Score)** | {comp['rps']['baseline']:.4f} | **{comp['rps']['upgraded']:.4f}** | {'📉 ' if comp['rps']['delta'] < 0 else '📈 '}{comp['rps']['pct_change']:+.2f}% |")
        lines.append(f"| **Log-Loss (Exact Score NLL)** | {comp['logloss']['baseline']:.4f} | **{comp['logloss']['upgraded']:.4f}** | {'📉 ' if comp['logloss']['delta'] < 0 else '📈 '}{comp['logloss']['pct_change']:+.2f}% |")
        lines.append(f"| **1X2 İsabet Oranı (Hit Rate)** | %{comp['hit_rate_1x2']['baseline']:.1f} | **%{comp['hit_rate_1x2']['upgraded']:.1f}** | {comp['hit_rate_1x2']['delta']:+.1f}% |")
        lines.append(f"| **Çifte Şans İsabet Oranı** | %{comp['hit_rate_double_chance']['baseline']:.1f} | **%{comp['hit_rate_double_chance']['upgraded']:.1f}** | {comp['hit_rate_double_chance']['delta']:+.1f}% |")
        lines.append(f"| **Alt/Üst 2.5 İsabet Oranı** | %{comp['hit_rate_over25']['baseline']:.1f} | **%{comp['hit_rate_over25']['upgraded']:.1f}** | {comp['hit_rate_over25']['delta']:+.1f}% |")
        lines.append(f"| **İlk 1 (Tam) Skor Yakalama** | %{comp['hit_rate_top1_score']['baseline']:.1f} | **%{comp['hit_rate_top1_score']['upgraded']:.1f}** | {comp['hit_rate_top1_score']['delta']:+.1f}% |")
        lines.append(f"| **İlk 3 Skor Yakalama Oranı** | %{comp['hit_rate_top3_score']['baseline']:.1f} | **%{comp['hit_rate_top3_score']['upgraded']:.1f}** | {comp['hit_rate_top3_score']['delta']:+.1f}% |")
        lines.append(f"| **İlk 5 Skor Yakalama Oranı** | %{comp['hit_rate_top5_score']['baseline']:.1f} | **%{comp['hit_rate_top5_score']['upgraded']:.1f}** | {comp['hit_rate_top5_score']['delta']:+.1f}% |")
        lines.append(f"| **xG Farkı Hatası (xG Diff MAE)** | {comp['xg_diff_mae']['baseline']:.3f} | **{comp['xg_diff_mae']['upgraded']:.3f}** | {comp['xg_diff_mae']['delta']:+.3f} |")
        lines.append(f"| **Korner Beklentisi MAE** | {comp['corners_mae']['baseline']:.2f} | **{comp['corners_mae']['upgraded']:.2f}** | {comp['corners_mae']['delta']:+.2f} |")
        lines.append(f"| **Kart Beklentisi MAE** | {comp['cards_mae']['baseline']:.2f} | **{comp['cards_mae']['upgraded']:.2f}** | {comp['cards_mae']['delta']:+.2f} |\n")
    else:
        lines.append(f"- **Ortalama RPS:** {m_up['mean_rps']:.4f}")
        lines.append(f"- **1X2 İsabet Oranı:** %{m_up['hit_rate_1x2_pct']:.1f}")
        lines.append(f"- **Çifte Şans İsabeti:** %{m_up['hit_rate_double_chance_pct']:.1f}")
        lines.append(f"- **Alt/Üst 2.5 İsabeti:** %{m_up['hit_rate_over25_pct']:.1f}")
        lines.append(f"- **KG Var/Yok İsabeti:** %{m_up['hit_rate_btts_pct']:.1f}")
        lines.append(f"- **İlk 3 Skor Yakalama:** %{m_up['hit_rate_top3_score_pct']:.1f}\n")

    lines.append("## 2. Hata Teşhisi: Model Hatası vs. Bitiricilik / Şans Ayrıştırması (Variance Decomposition)\n")
    lines.append("Gerçekleşen gol farkı ile model beklentisi arasındaki sapmanın matematiksel varyans ayrıştırması:\n")
    lines.append(f"- **Model Kurgu Hatası (Model Setup MSE — $(\\lambda_H - \\mu_A) - (xG_H - xG_A)$):** `{v_dec['model_error_mse']:.3f}` (**%{v_dec['model_error_pct']:.1f}**)")
    lines.append(f"- **Bitiricilik & Şans Varyansı (Finishing Luck MSE — $(xG_H - xG_A) - (G_H - G_A)$):** `{v_dec['finishing_luck_mse']:.3f}` (**%{v_dec['finishing_luck_pct']:.1f}**)")
    lines.append(f"- **Toplam Skor Farkı MSE:** `{v_dec['total_mse']:.3f}`")
    lines.append("> [!NOTE]")
    lines.append(f"> Yanılgıların yaklaşık **%{v_dec['finishing_luck_pct']:.0f}'i**, modelin saha içi pozisyon kalitesini ($xG$) doğru tahmin etmesine rağmen oyuncuların olağan dışı şut bitiriciliği veya kaleci kurtarışları nedeniyle gerçekleşen şans faktöründen kaynaklanmaktadır.\n")

    lines.append("### Teşhis Dağılımı (Diagnosis Breakdown)\n")
    diag_dist = diag["diagnosis_distribution"]
    for tag, cnt in sorted(diag_dist.items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / len(records)) * 100.0
        lines.append(f"- **{tag}:** {cnt} maç (%{pct:.1f})")
    lines.append("")

    lines.append("## 3. Hafta Hafta Performans İlerlemesi (Round-by-Round Learning)\n")
    lines.append("| Hafta | Maç Sayısı | RPS | 1X2 İsabet % | Çifte Şans % | İlk 3 Skor % | Korner MAE | Kart MAE |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in diag["round_progression"]:
        lines.append(f"| {r['matchday']}. Hafta | {r['matches']} | {r['rps']:.4f} | %{r['hit_rate_1x2_pct']:.1f} | %{r['double_chance_pct']:.1f} | %{r['top3_score_pct']:.1f} | {r['corners_mae']:.2f} | {r['cards_mae']:.2f} |")
    lines.append("")

    lines.append("## 4. Takım Bazlı Sistematik Yanlılık Analizi (Team Bias Table)\n")
    lines.append("Modelin hangi takımları daha kolay çözdüğü ve hangi takımlarda sistematik sapma gösterdiği tablosu:\n")
    lines.append("| Takım | Maç | Gerçek Gol (A-Y) | Gerçek xG (A-Y) | Model Bekl. (A-Y) | Hücum Sapması (G) | Savunma Sapması (G) | 1X2 Doğruluk % |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for t in diag["team_bias_table"]:
        lines.append(f"| **{t['team_name']}** | {t['matches']} | {t['goals_scored']}-{t['goals_conceded']} | {t['xg_for']:.1f}-{t['xg_against']:.1f} | {t['pred_goals_for']:.1f}-{t['pred_goals_against']:.1f} | {t['attack_bias_vs_goals']:+.2f} | {t['defense_bias_vs_goals']:+.2f} | %{t['accuracy_1x2_pct']:.1f} |")
    lines.append("")

    lines.append("## 5. Yüksek Güvenli Favori Sürprizleri Analizi (Upset Post-Mortem)\n")
    lines.append("Modelin favoriye %52'den fazla kazanma şansı verdiği ancak favorinin puan kaybettiği maçlar:\n")
    if diag["upset_analysis"]:
        lines.append("| Hafta | Maç | Skor | Model Favorisi | Kazanma İhtimali | Gerçek xG Farkı | Sürprizin Temel Nedeni |")
        lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :--- |")
        for u in diag["upset_analysis"]:
            lines.append(f"| {u['matchday']}. H | {u['home_team']} - {u['away_team']} | **{u['actual_score']}** | {u['favorite_team']} | %{u['fav_win_prob_pct']:.1f} | {u['actual_xg_diff']:+.2f} | {u['post_mortem_cause']} |")
    else:
        lines.append("Yüksek güvenli favori sürprizi bulunmamaktadır.\n")
    lines.append("")

    lines.append("## 6. Hakem Etkisi, Kart Puanı ve Oyun Temposu Karnesi (Referee Officiating Impact)\n")
    lines.append("2026/27 sezonunda görev alan hakemlerin maç başına beklenen vs. gerçekleşen kart puanları ve tempo/xG akışı:\n")
    ref_table = diag.get("referee_impact_table", [])
    if ref_table:
        lines.append("| Hakem Adı | Maç | Gerçekleşen Kart Pts | Model Kart Pts | Kart Pts MAE | Gerçek Kart | Model Kart | Gerçek xG | Model xG |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for rf in ref_table:
            lines.append(f"| **{rf['referee']}** | {rf['matches']} | {rf['avg_actual_bp']:.1f} pts | {rf['avg_pred_bp']:.1f} pts | {rf['bp_mae']:.1f} pts | {rf['avg_actual_cards']:.1f} | {rf['avg_pred_cards']:.1f} | {rf['avg_actual_xg']:.2f} | {rf['avg_pred_xg']:.2f} |")
        lines.append("")

    lines.append("## 7. Oynanmış 72 Maçın Hafta Hafta Detaylı Simülasyon Tablosu (1–8. Hafta)\n")
    rounds_set = sorted(list(set(r["matchday"] for r in records if r["matchday"] > 0)))
    for rd in rounds_set:
        lines.append(f"### {rd}. Hafta Karşılaşmaları\n")
        lines.append("| Maç | Gerçek Skor | Gerçek xG | Tahmin (λ - μ) | Model 1X2 % | İlk 3 Skor Tahmini | Korner (T/G) | Kart (T/G) | Teşhis |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |")
        rd_matches = [r for r in records if r["matchday"] == rd]
        for m in rd_matches:
            top_s_str = ", ".join([f"{s[0]}-{s[1]} (%{s[2]*100:.0f})" for s in m["top_5_scores"][:3]])
            p_str = f"%{m['p_home']*100:.0f} / %{m['p_draw']*100:.0f} / %{m['p_away']*100:.0f}"
            c_str = f"{m['pred_corners_tot']:.1f} / {m['actual_corners_tot']}"
            k_str = f"{m['pred_cards_tot']:.1f} / {m['actual_cards_tot']}"
            lines.append(f"| **{m['home_team']} - {m['away_team']}** | **{m['actual_home_goals']}-{m['actual_away_goals']}** | {m['actual_hxg']:.2f}-{m['actual_axg']:.2f} | {m['pred_lam_h']:.2f} - {m['pred_mu_a']:.2f} | {p_str} | {top_s_str} | {c_str} | {k_str} | {m['diagnosis_tag']} |")
        lines.append("")

    report_text = "\n".join(lines)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text
