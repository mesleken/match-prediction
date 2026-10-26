"""
Matchday prospective forecasting runner and Closing Line Value (CLV) evaluation pipeline.
Implements automated matchday batch inference, tamper-proof audit logging, and CLV monitoring.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional
import duckdb
import numpy as np
import pandas as pd

from slmodel.config import get_model_config
from slmodel.models.market import devig_shin
from slmodel.services.prediction_service import MatchPredictionService
from slmodel.tracking.prediction_log import PredictionLogger


class MatchdayRunner:
    """Orchestrates end-to-end prospective fixture forecasting, logging, and CLV tracking."""

    def __init__(self, db_path: str = "data/warehouse.duckdb") -> None:
        self.db_path = db_path
        self.cfg = get_model_config()
        self.service = MatchPredictionService(db_path=db_path)
        self.logger = PredictionLogger(db_path=db_path)

    def record_matchday_forecasts(
        self,
        season: str,
        matchday: int,
        as_of: Optional[datetime] = None,
        run_monte_carlo: bool = True,
        n_mc: int = 500,
        output_dir: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Execute prospective forecasting for all fixtures in a designated matchday.
        Logs each prediction into the tamper-proof ledger and persists human-readable markdown summaries.
        """
        as_of_dt = as_of or datetime.now(timezone.utc)
        con = duckdb.connect(self.db_path, read_only=True)
        fixtures_df = con.execute("""
            SELECT m.match_id, m.season, m.matchday, m.kickoff_utc,
                   m.home_team_id, th.name as home_team,
                   m.away_team_id, ta.name as away_team,
                   m.referee, m.status
            FROM matches m
            JOIN teams th ON m.home_team_id = th.team_id
            JOIN teams ta ON m.away_team_id = ta.team_id
            WHERE m.season = ? AND m.matchday = ?
            ORDER BY m.kickoff_utc ASC, m.match_id ASC
        """, [season, matchday]).fetchdf()

        if fixtures_df.empty:
            con.close()
            return []

        # Fetch available odds snapshot if present
        m_ids = fixtures_df["match_id"].tolist()
        odds_df = con.execute("""
            SELECT match_id,
                   MAX(CASE WHEN market = '1x2' AND selection = '1' THEN odds END) as o1,
                   MAX(CASE WHEN market = '1x2' AND selection = 'X' THEN odds END) as ox,
                   MAX(CASE WHEN market = '1x2' AND selection = '2' THEN odds END) as o2,
                   MAX(snapshot_id) as snapshot_id
            FROM odds_snapshots
            WHERE match_id IN (SELECT UNNEST(?))
            GROUP BY match_id
        """, [m_ids]).fetchdf()
        con.close()

        odds_lookup = {}
        for r in odds_df.itertuples():
            if pd.notna(r.o1) and pd.notna(r.ox) and pd.notna(r.o2):
                odds_lookup[int(r.match_id)] = {
                    "odds": {"1": float(r.o1), "X": float(r.ox), "2": float(r.o2)},
                    "snapshot_id": int(r.snapshot_id) if pd.notna(r.snapshot_id) else None,
                }

        results: list[dict[str, Any]] = []
        out_path = Path(output_dir or f"reports/matchdays/season_{season}_md_{matchday}")
        out_path.mkdir(parents=True, exist_ok=True)

        for _, fix in fixtures_df.iterrows():
            mid = int(fix["match_id"])
            h_id = int(fix["home_team_id"])
            a_id = int(fix["away_team_id"])
            ref_name = str(fix["referee"]) if pd.notna(fix["referee"]) else None

            odds_info = odds_lookup.get(mid, None)
            closing_odds = odds_info["odds"] if odds_info else None
            snapshot_id = odds_info["snapshot_id"] if odds_info else None

            # Execute end-to-end match forecast via unified MatchPredictionService
            analysis = self.service.analyze_match(
                home_team_id=h_id,
                away_team_id=a_id,
                as_of=as_of_dt,
                closing_odds=closing_odds,
                n_monte_carlo=n_mc if run_monte_carlo else 50,
                referee_name=ref_name,
            )

            # Build score matrix array for logging
            from slmodel.simulate.score_matrix import score_matrix
            M_score = score_matrix(analysis.lam_h, analysis.lam_a, analysis.rho)

            # Record in tamper-proof prediction ledger
            log_entry = self.logger.log_prediction(
                match_id=mid,
                as_of=as_of_dt,
                lam_home=analysis.lam_h,
                lam_away=analysis.lam_a,
                rho=analysis.rho,
                score_matrix_arr=M_score,
                odds_snapshot_id=snapshot_id,
                model_version="0.3.0",
                avail_shift_h=0.0,
                avail_shift_a=0.0,
                lineup_known=False,
            )

            # Generate individual markdown dossier
            md_doc = analysis.markdown_report
            fixture_file = out_path / f"match_{mid}_{fix['home_team']}_vs_{fix['away_team']}.md".replace(" ", "_")
            fixture_file.write_text(md_doc, encoding="utf-8")

            results.append({
                "match_id": mid,
                "home_team": fix["home_team"],
                "away_team": fix["away_team"],
                "kickoff_utc": str(fix["kickoff_utc"]),
                "lam_h": analysis.lam_h,
                "lam_a": analysis.lam_a,
                "p_football_1x2": analysis.p_football_1x2,
                "p_calibrated_1x2": analysis.p_calibrated_1x2,
                "pred_id": log_entry["pred_id"],
                "row_hash": log_entry["row_hash"],
                "markdown_path": str(fixture_file),
            })

        # Save matchday index json
        index_json = out_path / "matchday_forecasts.json"
        index_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        return results

    def evaluate_matchday_clv(
        self,
        season: str,
        matchday: int,
        output_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Calculate Closing Line Value (CLV) and prediction calibration for recorded fixtures.
        Compares model-predicted true probabilities against closing Shin devigged market probabilities.
        """
        con = duckdb.connect(self.db_path, read_only=True)
        # Fetch fixtures with their logged predictions and closing odds
        query = """
            SELECT m.match_id, m.season, m.matchday, m.status,
                   m.home_goals, m.away_goals,
                   th.name as home_team, ta.name as away_team,
                   p.pred_id, p.predicted_at, p.as_of, p.lam_home, p.lam_away, p.rho,
                   p.score_matrix_json,
                   o.o1, o.ox, o.o2
            FROM matches m
            JOIN teams th ON m.home_team_id = th.team_id
            JOIN teams ta ON m.away_team_id = ta.team_id
            JOIN (
                SELECT *,
                       ROW_NUMBER() OVER(PARTITION BY match_id ORDER BY pred_id DESC) as rn
                FROM prediction_log
            ) p ON m.match_id = p.match_id AND p.rn = 1
            LEFT JOIN (
                SELECT match_id,
                       MAX(CASE WHEN selection = '1' THEN odds END) as o1,
                       MAX(CASE WHEN selection = 'X' THEN odds END) as ox,
                       MAX(CASE WHEN selection = '2' THEN odds END) as o2
                FROM odds_snapshots
                WHERE market = '1x2' AND kind = 'closing'
                GROUP BY match_id
            ) o ON m.match_id = o.match_id
            WHERE m.season = ? AND m.matchday = ?
            ORDER BY m.kickoff_utc ASC
        """
        df = con.execute(query, [season, matchday]).fetchdf()
        con.close()

        if df.empty:
            return {"status": "empty", "message": f"No logged predictions found for season {season} matchday {matchday}"}

        fixture_clv: list[dict[str, Any]] = []
        clv_edges_1: list[float] = []
        clv_edges_x: list[float] = []
        clv_edges_2: list[float] = []

        from slmodel.simulate.markets import one_x_two, calibrate_probabilities
        from slmodel.evaluate.metrics import rps, brier_score

        rps_list: list[float] = []
        brier_list: list[float] = []

        for _, r in df.iterrows():
            # Extract logged score matrix
            sm = np.array(json.loads(r["score_matrix_json"]), dtype=np.float64)
            p_raw = one_x_two(sm)
            p_cal = calibrate_probabilities(p_raw, temperature=self.cfg.calibration.default_temperature)

            p_mkt = None
            clv_dict = {"1": None, "X": None, "2": None}
            if pd.notna(r["o1"]) and pd.notna(r["ox"]) and pd.notna(r["o2"]):
                p_mkt_arr = devig_shin([float(r["o1"]), float(r["ox"]), float(r["o2"])])
                p_mkt = [round(float(p), 4) for p in p_mkt_arr]

                # CLV defined as Ratio of Model True Prob to Closing Market True Prob minus 1
                # Positive CLV on selection k indicates model identified an edge exceeding closing consensus
                clv_1 = float(p_cal[0] / p_mkt_arr[0] - 1.0)
                clv_x = float(p_cal[1] / p_mkt_arr[1] - 1.0)
                clv_2 = float(p_cal[2] / p_mkt_arr[2] - 1.0)
                clv_dict = {"1": round(clv_1, 4), "X": round(clv_x, 4), "2": round(clv_2, 4)}

                clv_edges_1.append(clv_1)
                clv_edges_x.append(clv_x)
                clv_edges_2.append(clv_2)

            # If match was played, evaluate realization
            eval_dict = {}
            if r["status"] == "played" and pd.notna(r["home_goals"]) and pd.notna(r["away_goals"]):
                hg = int(r["home_goals"])
                ag = int(r["away_goals"])
                act_idx = 0 if hg > ag else (1 if hg == ag else 2)
                r_rps = rps(p_cal, act_idx)
                r_brier = brier_score(p_cal, act_idx)
                rps_list.append(r_rps)
                brier_list.append(r_brier)
                eval_dict = {
                    "actual_score": f"{hg}-{ag}",
                    "actual_outcome": "1" if act_idx == 0 else ("X" if act_idx == 1 else "2"),
                    "rps": round(r_rps, 4),
                    "brier": round(r_brier, 4),
                }

            fixture_clv.append({
                "match_id": int(r["match_id"]),
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "p_calibrated": [round(float(p), 4) for p in p_cal],
                "p_market_closing": p_mkt,
                "clv_relative_edge": clv_dict,
                "realization": eval_dict or None,
            })

        all_edges = clv_edges_1 + clv_edges_2
        summary = {
            "status": "success",
            "season": season,
            "matchday": matchday,
            "total_fixtures": len(df),
            "fixtures_with_closing_odds": len(clv_edges_1),
            "mean_clv_home": round(float(np.mean(clv_edges_1)), 4) if clv_edges_1 else 0.0,
            "mean_clv_away": round(float(np.mean(clv_edges_2)), 4) if clv_edges_2 else 0.0,
            "positive_clv_rate": round(float(np.mean([e > 0 for e in all_edges])), 4) if all_edges else 0.0,
            "mean_rps_realized": round(float(np.mean(rps_list)), 6) if rps_list else None,
            "mean_brier_realized": round(float(np.mean(brier_list)), 6) if brier_list else None,
            "fixtures": fixture_clv,
        }

        # Write markdown report
        out_path = Path(output_dir or f"reports/matchdays/season_{season}_md_{matchday}")
        out_path.mkdir(parents=True, exist_ok=True)
        report_file = out_path / "clv_evaluation_report.md"

        md = f"""# Maç Günü CLV (Closing Line Value) ve Gerçekleşme Raporu

**Sezon:** {season} | **Hafta:** {matchday}  
**Değerlendirilen Maç Sayısı:** {len(df)}  
**Kapanış Oranı Bulunan Karşılaşma:** {len(clv_edges_1)}  
**Pozitif CLV Oranı (1/2 Seçimleri):** %{summary['positive_clv_rate']*100:.1f}  
**Gerçekleşen Ortalama RPS:** {f"{summary['mean_rps_realized']:.6f}" if summary['mean_rps_realized'] else "Maçlar Henüz Oynanmadı"}  

---

## 1. Karşılaşma Bazlı CLV ve Olasılık Dağılımı

| Karşılaşma | Model (1/X/2) | Kapanış Piyasa (1/X/2) | Göreceli CLV (1 / X / 2) | Sonuç | RPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
        for fix in fixture_clv:
            p_m = "/".join(f"{p*100:.1f}%" for p in fix["p_calibrated"])
            p_c = "/".join(f"{p*100:.1f}%" for p in fix["p_market_closing"]) if fix["p_market_closing"] else "N/A"
            clv_str = "/".join(f"{v:+.1%}" if v is not None else "N/A" for v in fix["clv_relative_edge"].values())
            res_str = fix["realization"]["actual_score"] if fix["realization"] else "Bekleniyor"
            rps_str = f"{fix['realization']['rps']:.4f}" if fix["realization"] else "—"
            md += f"| **{fix['home_team']} - {fix['away_team']}** | {p_m} | {p_c} | {clv_str} | {res_str} | {rps_str} |\n"

        report_file.write_text(md, encoding="utf-8")
        return summary
