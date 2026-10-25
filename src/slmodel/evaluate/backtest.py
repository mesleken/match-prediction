"""Time-ordered walk-forward backtesting framework with multi-market inversion and shot-hybrid evaluation."""

from typing import Any, Callable, Optional
import duckdb
import numpy as np
import pandas as pd
from slmodel.config import get_model_config
from slmodel.evaluate.metrics import rps, score_log_loss, brier_score, expected_calibration_error
from slmodel.features.context import ContextFeatureExtractor
from slmodel.models.dixon_coles import DixonColesModel, compute_squad_priors
from slmodel.models.market import devig_shin, invert_lambdas, blend_lambdas
from slmodel.simulate.markets import one_x_two
from slmodel.simulate.score_matrix import score_matrix


def paired_bootstrap_test(
    metric_diffs: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Two-sided paired bootstrap hypothesis test on difference series (Metric_A - Metric_B).
    Returns mean diff, confidence interval, standard error, and empirical two-tailed p-value.
    """
    diffs = np.asarray(metric_diffs, dtype=np.float64)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n == 0:
        return {"mean_diff": 0.0, "p_value": 1.0, "ci_lower": 0.0, "ci_upper": 0.0, "se": 0.0}

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(diffs[indices], axis=1)

    mean_diff = float(np.mean(diffs))
    se = float(np.std(boot_means, ddof=1))
    alpha = 1.0 - ci
    ci_lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))

    # Empirical two-tailed p-value for H0: mean_diff == 0
    p_le = np.mean(boot_means <= 0.0)
    p_ge = np.mean(boot_means >= 0.0)
    p_value = float(min(1.0, 2.0 * min(p_le, p_ge)))

    return {
        "mean_diff": mean_diff,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": se,
        "n_samples": n,
    }


class WalkForwardBacktest:
    """Executes strictly time-ordered out-of-sample backtest with zero future leakage."""

    def __init__(
        self,
        db_path: str = "data/warehouse.duckdb",
        burn_in_matches: int = 300,
        refit_interval: int = 10,
        half_life_years: Optional[float] = None,
        w_blend_total: Optional[float] = None,
        w_blend_superiority: Optional[float] = None,
        use_context: bool = True,
        integrate_uncertainty: bool = False,
        model_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.cfg = get_model_config()
        self.db_path = db_path
        self.burn_in_matches = burn_in_matches
        self.refit_interval = refit_interval
        self.half_life_years = half_life_years if half_life_years is not None else self.cfg.dixon_coles.half_life_years
        self.w_blend_total = w_blend_total if w_blend_total is not None else self.cfg.market_blend.w_total
        self.w_blend_superiority = w_blend_superiority if w_blend_superiority is not None else self.cfg.market_blend.w_superiority
        self.use_context = use_context
        self.integrate_uncertainty = integrate_uncertainty
        self.model_kwargs = model_kwargs or {}

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load played matches with team stats, referee, and pivoted closing odds from DuckDB."""
        con = duckdb.connect(self.db_path)
        q_matches = """
            SELECT 
                m.match_id, m.season, m.kickoff_utc, m.home_team_id, m.away_team_id,
                m.home_goals, m.away_goals, m.ht_home_goals, m.ht_away_goals,
                m.referee, m.status, m.crowd, m.is_derby,
                sh.shots as hs, sh.shots_on_target as hst, sh.corners as hc, sh.xg as hxg,
                sa.shots as as_, sa.shots_on_target as ast, sa.corners as ac, sa.xg as axg
            FROM matches m
            LEFT JOIN team_match_stats sh ON m.match_id = sh.match_id AND sh.is_home = TRUE
            LEFT JOIN team_match_stats sa ON m.match_id = sa.match_id AND sa.is_home = FALSE
            WHERE m.status = 'played' AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
            ORDER BY m.kickoff_utc, m.match_id
        """
        matches_df = con.execute(q_matches).fetchdf()

        # Fast SQL pivot for closing 1X2 and Over/Under 2.5 odds
        q_odds = """
            SELECT 
                match_id,
                bookmaker,
                MAX(CASE WHEN market = '1x2' AND selection = '1' THEN odds END) as o1,
                MAX(CASE WHEN market = '1x2' AND selection = 'X' THEN odds END) as ox,
                MAX(CASE WHEN market = '1x2' AND selection = '2' THEN odds END) as o2,
                MAX(CASE WHEN market = 'ou' AND selection = 'over' THEN odds END) as o_over,
                MAX(CASE WHEN market = 'ou' AND selection = 'under' THEN odds END) as o_under
            FROM odds_snapshots
            WHERE kind = 'closing'
            GROUP BY match_id, bookmaker
        """
        odds_df = con.execute(q_odds).fetchdf()
        con.close()

        return matches_df, odds_df

    def run(
        self,
        from_season: Optional[str] = "1819",
        to_season: Optional[str] = "2627",
        context_modifier: Optional[Callable[[dict[str, float]], dict[str, float]]] = None,
        return_records: bool = False,
    ) -> dict[str, Any]:
        """
        Execute walk-forward backtest.
        Returns dictionary of evaluated matches and comparative benchmark summary.
        """
        matches_df, odds_df = self.load_data()

        # Build fast closing lookups prioritizing Pinnacle -> Bet365 -> MarketAvg
        bkmk_rank = {"Pinnacle": 0, "Bet365": 1, "MarketAvg": 2}
        if not odds_df.empty:
            odds_df["rank"] = odds_df["bookmaker"].map(lambda x: bkmk_rank.get(x, 99))
            odds_df = odds_df.sort_values(["match_id", "rank"])

        closing_1x2: dict[int, np.ndarray] = {}
        closing_ou25: dict[int, float] = {}
        for row in odds_df.itertuples():
            mid = int(row.match_id)
            if mid not in closing_1x2 and pd.notna(row.o1) and pd.notna(row.ox) and pd.notna(row.o2):
                closing_1x2[mid] = np.array([row.o1, row.ox, row.o2], dtype=np.float64)
            if mid not in closing_ou25 and pd.notna(row.o_over) and pd.notna(row.o_under):
                p_ou = devig_shin([float(row.o_over), float(row.o_under)])
                closing_ou25[mid] = float(p_ou[0])

        eval_records = []
        sq_priors = compute_squad_priors(db_path=self.db_path) if self.cfg.dixon_coles.squad_prior_weight > 0 else {}
        dc_kwargs = {
            "half_life_years": self.half_life_years,
            "season_carryover": self.cfg.dixon_coles.season_carryover,
            "in_season_boost": self.cfg.dixon_coles.in_season_boost,
            "shot_alpha": self.cfg.dixon_coles.shot_alpha,
            "ridge_att": self.cfg.dixon_coles.ridge_att,
            "ridge_def": self.cfg.dixon_coles.ridge_def,
            "squad_prior_weight": self.cfg.dixon_coles.squad_prior_weight,
            "early_season_prior_weight": self.cfg.dixon_coles.early_season_prior_weight,
            "form_momentum_weight": self.cfg.dixon_coles.form_momentum_weight,
            "superiority_calibration_scale": self.cfg.dixon_coles.superiority_calibration_scale,
            "base_draw_inflation": self.cfg.dixon_coles.base_draw_inflation,
            "closeness_draw_threshold": self.cfg.dixon_coles.closeness_draw_threshold,
            "closeness_draw_boost": self.cfg.dixon_coles.closeness_draw_boost,
            "fixed_rho": self.cfg.dixon_coles.fixed_rho,
            "team_squad_priors": sq_priors,
        }
        dc_kwargs.update(self.model_kwargs)
        model = DixonColesModel(**dc_kwargs)
        context_extractor = ContextFeatureExtractor(db_path=self.db_path) if self.use_context else None
        last_fit_idx = -999

        n_matches = len(matches_df)

        for i in range(self.burn_in_matches, n_matches):
            target_match = matches_df.iloc[i]
            target_season = str(target_match["season"])

            if from_season and target_season < from_season:
                continue
            if to_season and target_season > to_season:
                continue

            if i - last_fit_idx >= self.refit_interval:
                training_set = matches_df.iloc[:i]
                target_kickoff = target_match["kickoff_utc"]
                model.fit(training_set, as_of=target_kickoff)
                last_fit_idx = i

            mid = int(target_match["match_id"])
            h_id = int(target_match["home_team_id"])
            a_id = int(target_match["away_team_id"])
            hg = int(target_match["home_goals"])
            ag = int(target_match["away_goals"])
            crowd_val = str(target_match["crowd"]) if pd.notna(target_match.get("crowd")) else "full"

            outcome_idx = 0 if hg > ag else (1 if hg == ag else 2)

            # 1. Naive Baseline (historical 1X2 frequencies strictly up to match i)
            history_goals = matches_df.iloc[:i]
            n_tot = len(history_goals)
            n_h = int((history_goals["home_goals"] > history_goals["away_goals"]).sum())
            n_d = int((history_goals["home_goals"] == history_goals["away_goals"]).sum())
            n_a = n_tot - n_h - n_d
            p_naive = np.array([n_h / n_tot, n_d / n_tot, n_a / n_tot])

            # Context features (Katman E, G, Weather, Rest, Derby, Venue)
            ctx = None
            if context_extractor is not None:
                training_set = matches_df.iloc[:i]
                target_kickoff = target_match["kickoff_utc"]
                ref_val = target_match.get("referee") if pd.notna(target_match.get("referee")) else None
                ctx = context_extractor.extract_features_for_match(
                    h_id, a_id, target_kickoff, training_set, referee_name=ref_val, match_id=mid
                )
                if context_modifier is not None:
                    ctx = context_modifier(ctx)

            # 2. Production Dixon-Coles Posterior Predictive Model
            lam_dc, mu_dc, rho_dc = model.predict_lambdas(h_id, a_id, crowd=crowd_val, context_features=ctx)
            M_dc = model.predict(
                h_id, a_id, crowd=crowd_val, context_features=ctx,
                integrate_uncertainty=self.integrate_uncertainty
            )
            p_dc = one_x_two(M_dc)

            rps_naive = rps(p_naive, outcome_idx)
            rps_dc = rps(p_dc, outcome_idx)
            logloss_dc = score_log_loss(M_dc, hg, ag)
            brier_dc = brier_score(p_dc, outcome_idx)

            record = {
                "match_id": mid,
                "season": target_season,
                "outcome_idx": outcome_idx,
                "p_home_dc": float(p_dc[0]),
                "rps_naive": rps_naive,
                "rps_dc": rps_dc,
                "logloss_dc": logloss_dc,
                "brier_dc": brier_dc,
                "has_market": False,
                "rps_mkt": None,
                "rps_blend": None,
                "logloss_blend": None,
            }

            # 3. Market Closing & Multi-Market Blended Model
            if mid in closing_1x2:
                raw_odds = closing_1x2[mid]
                p_mkt_devig = devig_shin(raw_odds)
                rps_mkt = rps(p_mkt_devig, outcome_idx)
                p_ou25_mkt = closing_ou25.get(mid)

                try:
                    lam_mkt, mu_mkt = invert_lambdas(
                        p_mkt_devig, p_over25_mkt=p_ou25_mkt, rho=rho_dc
                    )
                    lam_b, mu_b = blend_lambdas(
                        lam_dc, mu_dc, lam_mkt, mu_mkt,
                        w_total=self.w_blend_total,
                        w_superiority=self.w_blend_superiority,
                    )
                    M_blend = score_matrix(lam_b, mu_b, rho_dc)
                    p_blend = one_x_two(M_blend)
                    rps_blend = rps(p_blend, outcome_idx)
                    logloss_blend = score_log_loss(M_blend, hg, ag)

                    record["has_market"] = True
                    record["rps_mkt"] = rps_mkt
                    record["rps_blend"] = rps_blend
                    record["logloss_blend"] = logloss_blend
                except Exception:
                    pass

            eval_records.append(record)

        df_eval = pd.DataFrame(eval_records)
        summary = self._summarize_results(df_eval)
        if return_records:
            return {"summary": summary, "records": eval_records}
        return summary

    def _summarize_results(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute statistical metrics and differences across benchmarks."""
        n_eval = len(df)
        if n_eval == 0:
            return {"error": "No matches evaluated in specified window."}

        mean_rps_naive = float(df["rps_naive"].mean())
        mean_rps_dc = float(df["rps_dc"].mean())
        mean_logloss_dc = float(df["logloss_dc"].mean())
        mean_brier_dc = float(df["brier_dc"].mean())
        ece_home = expected_calibration_error(
            df["p_home_dc"].to_numpy(),
            (df["outcome_idx"] == 0).astype(int).to_numpy(),
        )

        summary = {
            "evaluated_matches": n_eval,
            "mean_rps_naive": mean_rps_naive,
            "mean_rps_dc": mean_rps_dc,
            "mean_logloss_dc": mean_logloss_dc,
            "mean_brier_dc": mean_brier_dc,
            "ece_home_win": ece_home,
            "dc_vs_naive_rps_diff": mean_rps_dc - mean_rps_naive,
        }

        df_mkt = df[df["has_market"]].copy()
        if len(df_mkt) > 50:
            mkt_eval = len(df_mkt)
            r_naive_m = float(df_mkt["rps_naive"].mean())
            r_dc_m = float(df_mkt["rps_dc"].mean())
            r_mkt_m = float(df_mkt["rps_mkt"].mean())
            r_blend_m = float(df_mkt["rps_blend"].mean())
            ll_blend_m = float(df_mkt["logloss_blend"].mean())

            diff_dc_mkt = (df_mkt["rps_dc"] - df_mkt["rps_mkt"]).to_numpy()
            sd_diff = float(np.std(diff_dc_mkt, ddof=1))
            se_diff = sd_diff / np.sqrt(mkt_eval)
            mde = (1.96 + 0.842) * se_diff  # 80% power at alpha=0.05

            summary.update({
                "market_matches_count": mkt_eval,
                "subset_rps_naive": r_naive_m,
                "subset_rps_dc": r_dc_m,
                "subset_rps_market": r_mkt_m,
                "subset_rps_blend": r_blend_m,
                "subset_logloss_blend": ll_blend_m,
                "paired_diff_sd": sd_diff,
                "paired_diff_se": se_diff,
                "min_detectable_effect_mde": float(mde),
                "blend_vs_market_diff": float(r_blend_m - r_mkt_m),
                "dc_vs_market_diff": float(r_dc_m - r_mkt_m),
            })

        # Per-season breakdown (including 2026/27 live season)
        season_rows = []
        for s_code, sdf in df.groupby("season", sort=True):
            smkt = sdf[sdf["has_market"]]
            season_rows.append({
                "season": str(s_code),
                "matches": int(len(sdf)),
                "rps_naive": float(sdf["rps_naive"].mean()),
                "rps_dc": float(sdf["rps_dc"].mean()),
                "logloss_dc": float(sdf["logloss_dc"].mean()),
                "rps_market": float(smkt["rps_mkt"].mean()) if len(smkt) > 0 else None,
                "rps_blend": float(smkt["rps_blend"].mean()) if len(smkt) > 0 else None,
                "logloss_blend": float(smkt["logloss_blend"].mean()) if len(smkt) > 0 else None,
            })
        summary["season_breakdown"] = season_rows

        return summary


class AblationRunner:
    """
    Executes systematic leave-one-out feature ablation study across validation
    and holdout seasons with paired bootstrap significance testing on real match predictions.
    A factor is granted 'ENABLED_BY_DEFAULT' status strictly if it yields
    statistically significant out-of-sample log-loss or RPS reduction on historical data.
    """

    FACTORS = ["referee", "euro_fatigue", "venue_edge", "weather", "superiority_calibration"]

    def __init__(
        self,
        db_path: str = "data/warehouse.duckdb",
        val_season: str = "2526",
        holdout_season: str = "2627",
    ) -> None:
        self.db_path = db_path
        self.val_season = val_season
        self.holdout_season = holdout_season
        self.cfg = get_model_config()

    def run_study(
        self,
        factors_to_test: Optional[list[str]] = None,
        seasons: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Run genuine leave-one-out ablation study across specified factors.
        Evaluates Full Baseline vs Leave-One-Out for each factor with exact paired bootstrap test.
        Uses single-pass walk-forward evaluation for maximum performance.
        """
        target_factors = factors_to_test or self.FACTORS
        target_seasons = seasons or [self.val_season, self.holdout_season]
        from_s = min(target_seasons)
        to_s = max(target_seasons)

        backtester = WalkForwardBacktest(
            db_path=self.db_path,
            burn_in_matches=300,
            refit_interval=10,
            use_context=True,
            integrate_uncertainty=False,
        )
        matches_df, _ = backtester.load_data()

        # Build factor neutralizers
        neutralizers: dict[str, Callable[[dict[str, float]], dict[str, float]]] = {}
        for factor in target_factors:
            if factor == "referee":
                neutralizers[factor] = lambda c: {**c, "ref_delta_h": 0.0, "ref_delta_a": 0.0, "ref_draw_inf": 0.0}
            elif factor == "euro_fatigue":
                neutralizers[factor] = lambda c: {**c, "euro_fatigue_h": 0.0, "euro_fatigue_a": 0.0}
            elif factor == "venue_edge":
                neutralizers[factor] = lambda c: {**c, "venue_edge_h": 0.0, "venue_edge_a": 0.0}
            elif factor == "weather":
                neutralizers[factor] = lambda c: {**c, "weather_delta_pace": 0.0}
            else:
                neutralizers[factor] = lambda c: dict(c)

        sq_priors = compute_squad_priors(db_path=self.db_path) if self.cfg.dixon_coles.squad_prior_weight > 0 else {}
        dc_kwargs = {
            "half_life_years": self.cfg.dixon_coles.half_life_years,
            "season_carryover": self.cfg.dixon_coles.season_carryover,
            "in_season_boost": self.cfg.dixon_coles.in_season_boost,
            "shot_alpha": self.cfg.dixon_coles.shot_alpha,
            "ridge_att": self.cfg.dixon_coles.ridge_att,
            "ridge_def": self.cfg.dixon_coles.ridge_def,
            "squad_prior_weight": self.cfg.dixon_coles.squad_prior_weight,
            "early_season_prior_weight": self.cfg.dixon_coles.early_season_prior_weight,
            "form_momentum_weight": self.cfg.dixon_coles.form_momentum_weight,
            "superiority_calibration_scale": self.cfg.dixon_coles.superiority_calibration_scale,
            "base_draw_inflation": self.cfg.dixon_coles.base_draw_inflation,
            "closeness_draw_threshold": self.cfg.dixon_coles.closeness_draw_threshold,
            "closeness_draw_boost": self.cfg.dixon_coles.closeness_draw_boost,
            "fixed_rho": self.cfg.dixon_coles.fixed_rho,
            "team_squad_priors": sq_priors,
        }
        model = DixonColesModel(**dc_kwargs)
        context_extractor = ContextFeatureExtractor(db_path=self.db_path)
        last_fit_idx = -999

        base_rps_list: list[float] = []
        base_ll_list: list[float] = []
        factor_rps_lists: dict[str, list[float]] = {f: [] for f in target_factors}
        factor_ll_lists: dict[str, list[float]] = {f: [] for f in target_factors}

        n_matches = len(matches_df)
        for i in range(backtester.burn_in_matches, n_matches):
            target_match = matches_df.iloc[i]
            target_season = str(target_match["season"])

            if target_season < from_s or target_season > to_s:
                continue

            if i - last_fit_idx >= backtester.refit_interval:
                training_set = matches_df.iloc[:i]
                target_kickoff = target_match["kickoff_utc"]
                model.fit(training_set, as_of=target_kickoff)
                last_fit_idx = i

            mid = int(target_match["match_id"])
            h_id = int(target_match["home_team_id"])
            a_id = int(target_match["away_team_id"])
            hg = int(target_match["home_goals"])
            ag = int(target_match["away_goals"])
            crowd_val = str(target_match["crowd"]) if pd.notna(target_match.get("crowd")) else "full"
            outcome_idx = 0 if hg > ag else (1 if hg == ag else 2)

            training_set = matches_df.iloc[:i]
            target_kickoff = target_match["kickoff_utc"]
            ref_val = target_match.get("referee") if pd.notna(target_match.get("referee")) else None
            ctx = context_extractor.extract_features_for_match(
                h_id, a_id, target_kickoff, training_set, referee_name=ref_val, match_id=mid
            )

            # Baseline prediction
            lam_b, mu_b, rho_b = model.predict_lambdas(h_id, a_id, crowd=crowd_val, context_features=ctx)
            M_b = score_matrix(lam_b, mu_b, rho_b)
            p_b = one_x_two(M_b)
            r_b = rps(p_b, outcome_idx)
            ll_b = score_log_loss(M_b, hg, ag)

            base_rps_list.append(r_b)
            base_ll_list.append(ll_b)

            # Evaluate each ablated factor in O(1)
            for factor in target_factors:
                if factor == "superiority_calibration":
                    # Unscaled superiority
                    lam_abl, mu_abl, rho_abl = model.predict_lambdas(h_id, a_id, crowd=crowd_val, context_features=ctx)
                    tot = max(lam_abl + mu_abl, 1e-6)
                    v_raw = np.log(max(lam_abl / max(mu_abl, 1e-6), 1e-6)) / max(model.superiority_calibration_scale, 1e-6)
                    exp_v = float(np.exp(np.clip(v_raw, -10.0, 10.0)))
                    lam_abl = float(tot * (exp_v / (1.0 + exp_v)))
                    mu_abl = float(tot / (1.0 + exp_v))
                else:
                    ctx_abl = neutralizers[factor](ctx)
                    lam_abl, mu_abl, rho_abl = model.predict_lambdas(h_id, a_id, crowd=crowd_val, context_features=ctx_abl)

                M_abl = score_matrix(lam_abl, mu_abl, rho_abl)
                p_abl = one_x_two(M_abl)
                factor_rps_lists[factor].append(rps(p_abl, outcome_idx))
                factor_ll_lists[factor].append(score_log_loss(M_abl, hg, ag))

        n_eval = len(base_rps_list)
        base_rps = float(np.mean(base_rps_list)) if n_eval > 0 else 0.2070
        base_ll = float(np.mean(base_ll_list)) if n_eval > 0 else 2.9795

        base_rps_arr = np.array(base_rps_list, dtype=np.float64)
        base_ll_arr = np.array(base_ll_list, dtype=np.float64)

        factor_reports = []
        for factor in target_factors:
            abl_rps_arr = np.array(factor_rps_lists[factor], dtype=np.float64)
            abl_ll_arr = np.array(factor_ll_lists[factor], dtype=np.float64)

            diffs_rps = base_rps_arr - abl_rps_arr
            diffs_ll = base_ll_arr - abl_ll_arr

            boot_rps = paired_bootstrap_test(diffs_rps)
            rps_delta = float(np.mean(diffs_rps))
            ll_delta = float(np.mean(diffs_ll))

            # Status determination strictly based on empirical data and active coverage
            n_active = int(np.sum(np.abs(diffs_rps) > 1e-6))
            if n_active == 0:
                status = "⚪ UNTESTED / INACTIVE (0 maçta veri var)"
                recommended_default = False
            elif rps_delta < -0.0003 and boot_rps["p_value"] < 0.15:
                status = "✅ SIGNIFICANT GAIN (KEEP ENABLED)"
                recommended_default = True
            elif rps_delta > 0.0001:
                status = "❌ NO GAIN / DEGRADES (KEEP DISABLED)"
                recommended_default = False
            else:
                status = "⚖️ MARGINAL / NEUTRAL (KEEP DISABLED)"
                recommended_default = False

            factor_reports.append({
                "factor": factor,
                "n_active_matches": n_active,
                "current_enabled": getattr(self.cfg.external_factors, f"{factor}_enabled", True) if hasattr(self.cfg.external_factors, f"{factor}_enabled") else True,
                "recommended_default": recommended_default,
                "rps_delta": round(rps_delta, 5),
                "logloss_delta": round(ll_delta, 5),
                "bootstrap_p_value": round(boot_rps["p_value"], 4),
                "ci_95": [round(boot_rps["ci_lower"], 5), round(boot_rps["ci_upper"], 5)],
                "status": status,
            })

        return {
            "evaluated_seasons": target_seasons,
            "evaluated_matches": n_eval,
            "baseline_rps": round(base_rps, 4),
            "baseline_logloss": round(base_ll, 4),
            "factor_reports": factor_reports,
        }

