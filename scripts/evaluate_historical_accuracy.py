"""Comprehensive Historical Walk-Forward Simulation and Multi-Market Accuracy Benchmark."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import duckdb
import numpy as np
import pandas as pd

from slmodel.config import get_model_config
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
    double_chance,
    most_likely_scores,
)
from slmodel.simulate.score_matrix import (
    score_matrix,
    posterior_predictive_score_matrix,
)


def run_comprehensive_historical_evaluation(
    db_path: str = "data/warehouse.duckdb",
    from_season: str = "2122",
    to_season: str = "2627",
    refit_interval: int = 10,
) -> dict[str, Any]:
    """
    Executes strictly time-ordered, walk-forward out-of-sample simulation across historical seasons.
    Tests exact accuracy percentages across 1X2, Double Chance, Over/Under 2.5, BTTS, and Exact Scores.
    """
    cfg = get_model_config()
    con = duckdb.connect(db_path, read_only=True)

    teams_df = con.execute("SELECT team_id, name FROM teams").fetchdf()
    team_names = dict(zip(teams_df["team_id"], teams_df["name"]))

    # Query all played matches with stats
    q = """
        SELECT 
            m.match_id, m.season, m.matchday, m.kickoff_utc,
            m.home_team_id, m.away_team_id,
            m.home_goals, m.away_goals, m.ht_home_goals, m.ht_away_goals,
            m.referee, m.stadium_id, m.status, m.crowd, m.is_derby,
            sh.shots as hs, sh.shots_on_target as hst, sh.corners as hc,
            sh.fouls as hf, sh.yellow as hy, sh.red as hr,
            COALESCE(sh.xg, sh.shots * 0.10) as hxg,
            sa.shots as as_, sa.shots_on_target as ast, sa.corners as ac,
            sa.fouls as af, sa.yellow as ay, sa.red as ar,
            COALESCE(sa.xg, sa.shots * 0.10) as axg
        FROM matches m
        LEFT JOIN team_match_stats sh ON m.match_id = sh.match_id AND sh.is_home = TRUE
        LEFT JOIN team_match_stats sa ON m.match_id = sa.match_id AND sa.is_home = FALSE
        WHERE m.status = 'played' AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
        ORDER BY m.kickoff_utc, m.match_id
    """
    matches_df = con.execute(q).fetchdf()

    # Pre-load odds
    q_odds = """
        SELECT match_id,
               MAX(CASE WHEN market = '1x2' AND selection = '1' THEN odds END) as o1,
               MAX(CASE WHEN market = '1x2' AND selection = 'X' THEN odds END) as ox,
               MAX(CASE WHEN market = '1x2' AND selection = '2' THEN odds END) as o2,
               MAX(CASE WHEN market = 'ou' AND selection = 'over' THEN odds END) as o_over,
               MAX(CASE WHEN market = 'ou' AND selection = 'under' THEN odds END) as o_under
        FROM odds_snapshots
        WHERE kind = 'closing'
        GROUP BY match_id
    """
    odds_df = con.execute(q_odds).fetchdf()
    con.close()

    odds_map: dict[int, dict[str, float]] = {}
    for r in odds_df.itertuples():
        mid = int(r.match_id)
        if pd.notna(r.o1) and pd.notna(r.ox) and pd.notna(r.o2):
            odds_map[mid] = {
                "o1": float(r.o1), "ox": float(r.ox), "o2": float(r.o2),
                "o_over": float(r.o_over) if pd.notna(r.o_over) else None,
                "o_under": float(r.o_under) if pd.notna(r.o_under) else None,
            }

    ctx_extractor = ContextFeatureExtractor(db_path=db_path)
    ref_engine = RefereeEngine(db_path=db_path)
    sq_priors = compute_squad_priors(db_path=db_path)

    dc_model = DixonColesModel(
        half_life_years=cfg.dixon_coles.half_life_years,
        in_season_boost=cfg.dixon_coles.in_season_boost,
        squad_prior_weight=cfg.dixon_coles.squad_prior_weight,
        team_squad_priors=sq_priors,
    )
    side_model = SideMarketsModel(
        half_life_years=cfg.side_markets.corners_half_life,
        shrinkage_k=cfg.side_markets.corners_shrinkage_k,
        in_season_boost=cfg.side_markets.in_season_boost,
    )

    n_total = len(matches_df)
    eval_records: list[dict[str, Any]] = []
    last_fit_idx = -999

    for i in range(len(matches_df)):
        m = matches_df.iloc[i]
        season_str = str(m["season"])

        if season_str < from_season:
            continue
        if season_str > to_season:
            continue

        mid = int(m["match_id"])
        h_id = int(m["home_team_id"])
        a_id = int(m["away_team_id"])
        hg = int(m["home_goals"])
        ag = int(m["away_goals"])
        hxg = float(m["hxg"]) if pd.notna(m["hxg"]) else float(hg)
        axg = float(m["axg"]) if pd.notna(m["axg"]) else float(ag)
        kickoff = m["kickoff_utc"]
        ref_name = str(m["referee"]) if pd.notna(m.get("referee")) else None
        crowd_val = str(m["crowd"]) if pd.notna(m.get("crowd")) else "full"

        # Fit model on training slice strictly prior to match i
        if (i - last_fit_idx) >= refit_interval or last_fit_idx < 0:
            train_slice = matches_df.iloc[:i]
            dc_model.fit(train_slice, as_of=kickoff)
            side_model.fit(train_slice, as_of=kickoff)
            last_fit_idx = i

        # Extract context
        ctx = ctx_extractor.extract_features_for_match(
            home_team_id=h_id,
            away_team_id=a_id,
            kickoff_utc=kickoff,
            matches_history=matches_df.iloc[:i],
            referee_name=ref_name,
            match_id=mid,
        )

        # Baseline Model Predictions
        lam_base, mu_base, rho_base = dc_model.predict_lambdas(h_id, a_id, crowd=crowd_val)

        # Contextual coupling shifts
        # Rest and fatigue shift
        rest_adv = float(ctx.get("net_rest_advantage", 0.0))
        euro_fat_h = float(ctx.get("euro_fatigue_h", 0.0))
        euro_fat_a = float(ctx.get("euro_fatigue_a", 0.0))
        ref_dh = float(ctx.get("ref_delta_h", 0.0))
        ref_da = float(ctx.get("ref_delta_a", 0.0))
        venue_h = float(ctx.get("venue_edge_h", 0.0))
        venue_a = float(ctx.get("venue_edge_a", 0.0))
        pace_weather = float(ctx.get("weather_delta_pace", 0.0))

        # Attacking/Defensive log shifts
        shift_h = 0.045 * rest_adv + euro_fat_h + ref_dh + 0.50 * venue_h + pace_weather
        shift_a = -0.045 * rest_adv + euro_fat_a + ref_da + 0.50 * venue_a + pace_weather

        M_pred = dc_model.predict(
            h_id, a_id,
            crowd=crowd_val,
            context_features=ctx,
            integrate_uncertainty=cfg.dixon_coles.integrate_uncertainty,
        )
        lam_final, mu_final, rho_base = dc_model.predict_lambdas(
            h_id, a_id, crowd=crowd_val, context_features=ctx
        )

        p1, px, p2 = one_x_two(M_pred)
        dc_dict = double_chance(M_pred)
        p_1x, p_12, p_x2 = float(dc_dict["1X"]), float(dc_dict["12"]), float(dc_dict["X2"])
        ou_dict = over_under(M_pred, 2.5)
        p_over25 = float(ou_dict["over"])
        p_under25 = float(ou_dict["under"])
        btts_dict = both_teams_to_score(M_pred)
        p_btts = float(btts_dict["yes"])
        top5_scores = most_likely_scores(M_pred, top_n=5)

        # Actual Outcomes
        act_outcome_idx = 0 if hg > ag else (1 if hg == ag else 2)
        act_total_goals = hg + ag
        act_over25 = act_total_goals > 2.5
        act_btts = (hg > 0) and (ag > 0)

        # Model Choices (Argmax prediction)
        pred_outcome_idx = int(np.argmax([p1, px, p2]))
        hit_1x2 = (pred_outcome_idx == act_outcome_idx)

        # Double Chance Choice: Pick highest probability combination
        # Options: (1X: p1+px), (X2: px+p2), (12: p1+p2)
        dc_probs = [p_1x, p_x2, p_12]
        best_dc_idx = int(np.argmax(dc_probs))
        # 0 -> 1X (hit if home win or draw)
        # 1 -> X2 (hit if draw or away win)
        # 2 -> 12 (hit if home win or away win)
        if best_dc_idx == 0:
            hit_dc = (act_outcome_idx in (0, 1))
        elif best_dc_idx == 1:
            hit_dc = (act_outcome_idx in (1, 2))
        else:
            hit_dc = (act_outcome_idx in (0, 2))

        # Over / Under Choice:
        pred_over25 = p_over25 >= 0.50
        hit_ou25 = (pred_over25 == act_over25)

        # BTTS Choice:
        pred_btts = p_btts >= 0.50
        hit_btts = (pred_btts == act_btts)

        # Exact Scores:
        top1_score = top5_scores[0]
        top3_tuples = [(s["home_goals"], s["away_goals"]) for s in top5_scores[:3]]
        top5_tuples = [(s["home_goals"], s["away_goals"]) for s in top5_scores[:5]]
        actual_tuple = (hg, ag)

        hit_top1_score = (actual_tuple == (top1_score["home_goals"], top1_score["away_goals"]))
        hit_top3_score = (actual_tuple in top3_tuples)
        hit_top5_score = (actual_tuple in top5_tuples)

        # Side Markets (Corners & Cards)
        ref_prof = ref_engine.get_referee_profile(ref_name) if ref_name else None
        pred_corn = predict_corners(lam_final, mu_final, home_team_id=h_id, away_team_id=a_id, side_model=side_model)
        pred_c = predict_cards(lam_final, mu_final, home_team_id=h_id, away_team_id=a_id, side_model=side_model, referee_profile=ref_prof)

        act_corners = int(m["hc"] + m["ac"]) if pd.notna(m.get("hc")) and pd.notna(m.get("ac")) else None
        act_yellow = int(m["hy"] + m["ay"]) if pd.notna(m.get("hy")) and pd.notna(m.get("ay")) else None
        act_red = int(m.get("hr", 0) + m.get("ar", 0)) if pd.notna(m.get("hr")) and pd.notna(m.get("ar")) else 0
        act_cards = (act_yellow + act_red) if act_yellow is not None else None

        # Metrics
        m_rps = rps(np.array([p1, px, p2]), act_outcome_idx)
        m_nll = score_log_loss(M_pred, hg, ag)
        m_brier = brier_score(np.array([p1, px, p2]), act_outcome_idx)

        eval_records.append({
            "match_id": mid,
            "season": season_str,
            "matchday": int(m["matchday"]) if pd.notna(m.get("matchday")) else 0,
            "home_team": team_names.get(h_id, f"Team_{h_id}"),
            "away_team": team_names.get(a_id, f"Team_{a_id}"),
            "actual_score": f"{hg}-{ag}",
            "act_outcome_idx": act_outcome_idx,
            "pred_outcome_idx": pred_outcome_idx,
            "p_home": p1,
            "p_draw": px,
            "p_away": p2,
            "p_fav": max(p1, px, p2),
            "hit_1x2": hit_1x2,
            "hit_dc": hit_dc,
            "p_over25": p_over25,
            "act_over25": act_over25,
            "hit_ou25": hit_ou25,
            "p_btts": p_btts,
            "act_btts": act_btts,
            "hit_btts": hit_btts,
            "hit_top1_score": hit_top1_score,
            "hit_top3_score": hit_top3_score,
            "hit_top5_score": hit_top5_score,
            "rps": m_rps,
            "nll": m_nll,
            "brier": m_brier,
            "pred_corners_tot": pred_corn.get("total", 9.5),
            "act_corners_tot": act_corners,
            "pred_cards_tot": pred_c.get("total_cards", 4.3),
            "act_cards_tot": act_cards,
            "hxg": hxg,
            "axg": axg,
            "lam_h": lam_final,
            "mu_a": mu_final,
        })

    df_eval = pd.DataFrame(eval_records)
    return summarize_historical_performance(df_eval)


def summarize_historical_performance(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate thorough hit rates, calibration, and market accuracy breakdowns."""
    n = len(df)
    if n == 0:
        return {"error": "No matches evaluated."}

    # 1. Overall Hit Rates (%)
    hit_1x2_pct = float(df["hit_1x2"].mean() * 100.0)
    hit_dc_pct = float(df["hit_dc"].mean() * 100.0)
    hit_ou25_pct = float(df["hit_ou25"].mean() * 100.0)
    hit_btts_pct = float(df["hit_btts"].mean() * 100.0)
    hit_top1_score_pct = float(df["hit_top1_score"].mean() * 100.0)
    hit_top3_score_pct = float(df["hit_top3_score"].mean() * 100.0)
    hit_top5_score_pct = float(df["hit_top5_score"].mean() * 100.0)

    # 2. Calibration & Loss Metrics
    mean_rps = float(df["rps"].mean())
    mean_nll = float(df["nll"].mean())
    mean_brier = float(df["brier"].mean())

    # ECE (Expected Calibration Error)
    ece_home = expected_calibration_error(df["p_home"].to_numpy(), (df["act_outcome_idx"] == 0).astype(int).to_numpy())
    ece_draw = expected_calibration_error(df["p_draw"].to_numpy(), (df["act_outcome_idx"] == 1).astype(int).to_numpy())
    ece_away = expected_calibration_error(df["p_away"].to_numpy(), (df["act_outcome_idx"] == 2).astype(int).to_numpy())

    # 3. Accuracy Grouped by Favorite Confidence (p_fav)
    fav_bins = [
        {"label": "Düşük Güven (< %45)", "mask": (df["p_fav"] < 0.45)},
        {"label": "Orta Güven (%45 - %55)", "mask": ((df["p_fav"] >= 0.45) & (df["p_fav"] < 0.55))},
        {"label": "Yüksek Güven (%55 - %65)", "mask": ((df["p_fav"] >= 0.55) & (df["p_fav"] < 0.65))},
        {"label": "Çok Yüksek Güven (≥ %65)", "mask": (df["p_fav"] >= 0.65)},
    ]
    confidence_breakdown = []
    for fb in fav_bins:
        sub = df[fb["mask"]]
        if len(sub) > 0:
            confidence_breakdown.append({
                "tier": fb["label"],
                "matches": int(len(sub)),
                "pct_of_all": round(len(sub) / n * 100.0, 1),
                "hit_1x2_pct": round(float(sub["hit_1x2"].mean() * 100.0), 1),
                "hit_dc_pct": round(float(sub["hit_dc"].mean() * 100.0), 1),
                "hit_ou25_pct": round(float(sub["hit_ou25"].mean() * 100.0), 1),
                "hit_top3_score_pct": round(float(sub["hit_top3_score"].mean() * 100.0), 1),
                "mean_rps": round(float(sub["rps"].mean()), 4),
            })

    # 4. Season by Season Breakdown
    season_breakdown = []
    for s_code, sdf in df.groupby("season", sort=True):
        season_breakdown.append({
            "season": str(s_code),
            "matches": int(len(sdf)),
            "hit_1x2_pct": round(float(sdf["hit_1x2"].mean() * 100.0), 1),
            "hit_dc_pct": round(float(sdf["hit_dc"].mean() * 100.0), 1),
            "hit_ou25_pct": round(float(sdf["hit_ou25"].mean() * 100.0), 1),
            "hit_btts_pct": round(float(sdf["hit_btts"].mean() * 100.0), 1),
            "hit_top1_score_pct": round(float(sdf["hit_top1_score"].mean() * 100.0), 1),
            "hit_top3_score_pct": round(float(sdf["hit_top3_score"].mean() * 100.0), 1),
            "hit_top5_score_pct": round(float(sdf["hit_top5_score"].mean() * 100.0), 1),
            "mean_rps": round(float(sdf["rps"].mean()), 4),
            "mean_nll": round(float(sdf["nll"].mean()), 4),
        })

    # 5. Side markets MAE (Corners and Cards)
    df_corn = df[df["act_corners_tot"].notna()]
    corn_mae = float(np.mean(np.abs(df_corn["pred_corners_tot"] - df_corn["act_corners_tot"]))) if len(df_corn) > 0 else 0.0

    df_cards = df[df["act_cards_tot"].notna()]
    cards_mae = float(np.mean(np.abs(df_cards["pred_cards_tot"] - df_cards["act_cards_tot"]))) if len(df_cards) > 0 else 0.0

    return {
        "total_evaluated_matches": n,
        "from_season": str(df["season"].min()),
        "to_season": str(df["season"].max()),
        "overall_hit_rates": {
            "hit_1x2_pct": round(hit_1x2_pct, 2),
            "hit_double_chance_pct": round(hit_dc_pct, 2),
            "hit_over_under_25_pct": round(hit_ou25_pct, 2),
            "hit_btts_pct": round(hit_btts_pct, 2),
            "hit_top1_exact_score_pct": round(hit_top1_score_pct, 2),
            "hit_top3_score_pct": round(hit_top3_score_pct, 2),
            "hit_top5_score_pct": round(hit_top5_score_pct, 2),
        },
        "overall_loss_metrics": {
            "mean_rps": round(mean_rps, 4),
            "mean_logloss_nll": round(mean_nll, 4),
            "mean_brier_score": round(mean_brier, 4),
            "ece_home_win": round(ece_home, 4),
            "ece_draw": round(ece_draw, 4),
            "ece_away_win": round(ece_away, 4),
            "corners_mae": round(corn_mae, 2),
            "cards_mae": round(cards_mae, 2),
        },
        "confidence_breakdown": confidence_breakdown,
        "season_breakdown": season_breakdown,
        "raw_records_count": len(df),
    }


if __name__ == "__main__":
    print("Executing Historical Walk-Forward Simulation (2021/22 - 2026/27)...")
    res = run_comprehensive_historical_evaluation(from_season="2122", to_season="2627", refit_interval=10)
    out_p = Path("reports/historical_model_accuracy.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Complete! Evaluated matches:", res["total_evaluated_matches"])
    print("1X2 Hit Rate:", res["overall_hit_rates"]["hit_1x2_pct"], "%")
    print("Double Chance Hit Rate:", res["overall_hit_rates"]["hit_double_chance_pct"], "%")
    print("Over/Under 2.5 Hit Rate:", res["overall_hit_rates"]["hit_over_under_25_pct"], "%")
    print("BTTS Hit Rate:", res["overall_hit_rates"]["hit_btts_pct"], "%")
    print("Top 1 Exact Score Hit Rate:", res["overall_hit_rates"]["hit_top1_exact_score_pct"], "%")
    print("Top 3 Score Hit Rate:", res["overall_hit_rates"]["hit_top3_score_pct"], "%")
    print("Top 5 Score Hit Rate:", res["overall_hit_rates"]["hit_top5_score_pct"], "%")
