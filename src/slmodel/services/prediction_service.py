"""
Prediction and match analysis service conforming to the Section 14 roadmap standard.
Delivers decoupled pure model, market baseline, hybrid blend, event Monte Carlo simulation,
and economic expected value (+EV) analysis.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import duckdb
import numpy as np

from slmodel.config import get_model_config
from slmodel.features.context import ContextFeatureExtractor
from slmodel.models.dixon_coles import create_dixon_coles_model, compute_squad_priors
from slmodel.models.market import devig_shin, blend_lambdas, invert_lambdas
from slmodel.simulate.markets import one_x_two, double_chance, over_under, both_teams_to_score, most_likely_scores, calibrate_probabilities
from slmodel.simulate.monte_carlo import run_monte_carlo_simulation, MonteCarloResult
from slmodel.simulate.tactics import TacticalProfile
from slmodel.data.loaders import load_match_history


@dataclass
class MatchAnalysisResult:
    home_team: str
    away_team: str
    as_of: datetime
    p_football_1x2: tuple[float, float, float]
    p_market_1x2: Optional[tuple[float, float, float]]
    p_hybrid_1x2: Optional[tuple[float, float, float]]
    lam_h: float
    lam_a: float
    ci_90_lam_h: tuple[float, float]
    ci_90_lam_a: tuple[float, float]
    ev_outcomes: dict[str, float]
    monte_carlo: MonteCarloResult
    top_scores: list[dict[str, Any]]
    markdown_report: str
    p_calibrated_1x2: Optional[tuple[float, float, float]] = None


class MatchPredictionService:
    """End-to-end service coordinating forecasting, market devigging, and Monte Carlo simulation."""

    def __init__(self, db_path: str = "data/warehouse.duckdb") -> None:
        self.db_path = db_path
        self.cfg = get_model_config()

    def analyze_match(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of: Optional[datetime] = None,
        closing_odds: Optional[dict[str, float]] = None,
        n_monte_carlo: int = 1000,
        referee_name: Optional[str] = None,
        seed: int = 42,
    ) -> MatchAnalysisResult:
        """Execute complete forecasting and simulation workflow for a fixture."""
        as_of_dt = as_of or datetime.now(timezone.utc)
        con = duckdb.connect(self.db_path, read_only=True)
        teams = dict(con.execute("SELECT team_id, name FROM teams").fetchall())
        home_name = teams.get(home_team_id, f"Team_{home_team_id}")
        away_name = teams.get(away_team_id, f"Team_{away_team_id}")

        # Training history strictly prior to as_of
        matches_df = load_match_history(con=con, as_of=as_of_dt)
        con.close()

        # 1. Pre-Match Forecasting (Dixon-Coles)
        is_season_2627 = as_of_dt >= datetime(2026, 7, 1, tzinfo=timezone.utc)
        sq_priors = compute_squad_priors(db_path=self.db_path) if is_season_2627 else {}
        model = create_dixon_coles_model(self.cfg, team_squad_priors=sq_priors)
        if not is_season_2627:
            model.squad_prior_weight = 0.0
            model.early_season_prior_weight = 0.0
        if not matches_df.empty:
            model.fit(matches_df, as_of=as_of_dt)

        ctx_extractor = ContextFeatureExtractor(db_path=self.db_path)
        ctx = ctx_extractor.extract_features_for_match(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            kickoff_utc=as_of_dt,
            matches_history=matches_df,
            referee_name=referee_name,
        )

        unc = model.predict_with_uncertainty(home_team_id, away_team_id, context_features=ctx)
        lam_h = unc["lam"]
        lam_a = unc["mu"]
        sig_h = unc["sigma_eta_h"]
        sig_a = unc["sigma_eta_a"]

        ci_90_h = (float(np.exp(np.log(lam_h) - 1.645 * sig_h)), float(np.exp(np.log(lam_h) + 1.645 * sig_h)))
        ci_90_a = (float(np.exp(np.log(lam_a) - 1.645 * sig_a)), float(np.exp(np.log(lam_a) + 1.645 * sig_a)))

        M_pred = model.predict(home_team_id, away_team_id, context_features=ctx, integrate_uncertainty=True)
        p_mod_1, p_mod_x, p_mod_2 = one_x_two(M_pred)
        p_football = (round(float(p_mod_1), 3), round(float(p_mod_x), 3), round(float(p_mod_2), 3))

        # Temperature Calibration (Faz 4.1 proven favorite scaling)
        p_calibrated = None
        if hasattr(self.cfg, "calibration") and self.cfg.calibration.temperature_enabled:
            cal_t = self.cfg.calibration.default_temperature
            p_cal_raw = calibrate_probabilities(list(p_football), temperature=cal_t)
            p_calibrated = (round(float(p_cal_raw[0]), 3), round(float(p_cal_raw[1]), 3), round(float(p_cal_raw[2]), 3))

        # 2. Market Intelligence & Devigging
        p_market = None
        p_hybrid = None
        ev_outcomes: dict[str, float] = {}

        if closing_odds and "1" in closing_odds and "X" in closing_odds and "2" in closing_odds:
            o1, ox, o2 = float(closing_odds["1"]), float(closing_odds["X"]), float(closing_odds["2"])
            p_mkt_raw = devig_shin([o1, ox, o2])
            p_market = (round(float(p_mkt_raw[0]), 3), round(float(p_mkt_raw[1]), 3), round(float(p_mkt_raw[2]), 3))

            # Calibrated blend
            w_mkt = self.cfg.market_blend.w_total
            w_mod = 1.0 - w_mkt
            p_base_for_blend = p_calibrated if p_calibrated is not None else p_football
            p_hyb_raw = [w_mod * p_base_for_blend[0] + w_mkt * p_mkt_raw[0],
                         w_mod * p_base_for_blend[1] + w_mkt * p_mkt_raw[1],
                         w_mod * p_base_for_blend[2] + w_mkt * p_mkt_raw[2]]
            tot_h = sum(p_hyb_raw)
            p_hybrid = (round(p_hyb_raw[0] / tot_h, 3), round(p_hyb_raw[1] / tot_h, 3), round(p_hyb_raw[2] / tot_h, 3))

            # Economic Expected Value (EV = p * O - 1)
            p_for_ev = p_calibrated if p_calibrated is not None else p_football
            ev_outcomes = {
                "1": round(float(p_for_ev[0] * o1 - 1.0), 3),
                "X": round(float(p_for_ev[1] * ox - 1.0), 3),
                "2": round(float(p_for_ev[2] * o2 - 1.0), 3),
            }

        # 3. Event-Based Monte Carlo Simulation
        if model.is_fitted and home_team_id in model.team_to_idx and away_team_id in model.team_to_idx:
            raw_att_h = float(model.att[model.team_to_idx[home_team_id]])
            raw_def_h = float(model.def_[model.team_to_idx[home_team_id]])
            raw_att_a = float(model.att[model.team_to_idx[away_team_id]])
            raw_def_a = float(model.def_[model.team_to_idx[away_team_id]])
            h_adv = float(getattr(model, "home_adv", 0.25))

            att_h_score = float(np.clip(0.20 + 0.15 * (raw_att_h + 0.5 * h_adv), 0.05, 0.45))
            def_h_score = float(np.clip(0.20 - 0.15 * raw_def_h + 0.05 * h_adv, 0.05, 0.45))
            att_a_score = float(np.clip(0.20 + 0.15 * raw_att_a, 0.05, 0.45))
            def_a_score = float(np.clip(0.20 - 0.15 * raw_def_a, 0.05, 0.45))
        else:
            att_h_score = 0.22
            def_h_score = 0.21
            att_a_score = 0.19
            def_a_score = 0.19

        mc_res = run_monte_carlo_simulation(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            att_h=att_h_score,
            def_h=def_h_score,
            att_a=att_a_score,
            def_a=def_a_score,
            lam_h=lam_h,
            lam_a=lam_a,
            n_simulations=n_monte_carlo,
            seed=seed,
        )

        top_scores = most_likely_scores(M_pred, top_n=5)

        # 4. Generate Section 14 Markdown Report
        report_md = self._render_section_14_report(
            home_name=home_name,
            away_name=away_name,
            p_football=p_football,
            p_calibrated=p_calibrated,
            p_market=p_market,
            p_hybrid=p_hybrid,
            lam_h=lam_h,
            lam_a=lam_a,
            ci_90_h=ci_90_h,
            ci_90_a=ci_90_a,
            ev_outcomes=ev_outcomes,
            closing_odds=closing_odds,
            mc=mc_res,
            top_scores=top_scores,
        )

        return MatchAnalysisResult(
            home_team=home_name,
            away_team=away_name,
            as_of=as_of_dt,
            p_football_1x2=p_football,
            p_calibrated_1x2=p_calibrated,
            p_market_1x2=p_market,
            p_hybrid_1x2=p_hybrid,
            lam_h=round(lam_h, 2),
            lam_a=round(lam_a, 2),
            ci_90_lam_h=(round(ci_90_h[0], 2), round(ci_90_h[1], 2)),
            ci_90_lam_a=(round(ci_90_a[0], 2), round(ci_90_a[1], 2)),
            ev_outcomes=ev_outcomes,
            monte_carlo=mc_res,
            top_scores=top_scores,
            markdown_report=report_md,
        )

    def _render_section_14_report(
        self,
        home_name: str,
        away_name: str,
        p_football: tuple[float, float, float],
        p_calibrated: Optional[tuple[float, float, float]],
        p_market: Optional[tuple[float, float, float]],
        p_hybrid: Optional[tuple[float, float, float]],
        lam_h: float,
        lam_a: float,
        ci_90_h: tuple[float, float],
        ci_90_a: tuple[float, float],
        ev_outcomes: dict[str, float],
        closing_odds: Optional[dict[str, float]],
        mc: MonteCarloResult,
        top_scores: list[dict[str, Any]],
    ) -> str:
        p_active = p_calibrated if p_calibrated is not None else p_football
        diff_h_str = f"{(p_active[0] - p_market[0]) * 100.0:+.1f}%" if p_market else "—"
        diff_x_str = f"{(p_active[1] - p_market[1]) * 100.0:+.1f}%" if p_market else "—"
        diff_a_str = f"{(p_active[2] - p_market[2]) * 100.0:+.1f}%" if p_market else "—"

        p_mkt_h_str = f"%{p_market[0]*100.0:.1f}" if p_market else "—"
        p_mkt_x_str = f"%{p_market[1]*100.0:.1f}" if p_market else "—"
        p_mkt_a_str = f"%{p_market[2]*100.0:.1f}" if p_market else "—"

        p_hyb_h_str = f"%{p_hybrid[0]*100.0:.1f}" if p_hybrid else "—"
        p_hyb_x_str = f"%{p_hybrid[1]*100.0:.1f}" if p_hybrid else "—"
        p_hyb_a_str = f"%{p_hybrid[2]*100.0:.1f}" if p_hybrid else "—"

        ev_1_str = f"{ev_outcomes.get('1', 0.0):+.1%}" if closing_odds else "—"
        ev_x_str = f"{ev_outcomes.get('X', 0.0):+.1%}" if closing_odds else "—"
        ev_2_str = f"{ev_outcomes.get('2', 0.0):+.1%}" if closing_odds else "—"

        cal_row = ""
        if p_calibrated:
            cal_row = f"| **Futbol Modeli (Kalibre)** | %{p_calibrated[0]*100.0:.1f} | %{p_calibrated[1]*100.0:.1f} | %{p_calibrated[2]*100.0:.1f} | Faz 4.1 sıcaklık kalibrasyonu (favori sapması düzeltilmiş) |\n"

        md = f"""# Maç Öncesi Analiz: {home_name} vs {away_name}

---

### 1. Sonuç Olasılıkları (1X2)

| Kaynak | Ev Sahibi ({home_name}) | Beraberlik | Deplasman ({away_name}) |
| :--- | :---: | :---: | :---: |
| **Model B (Dixon-Coles Ham)** | %{p_football[0]*100.0:.1f} | %{p_football[1]*100.0:.1f} | %{p_football[2]*100.0:.1f} |
"""
        if p_calibrated:
            md += f"| **Model B (Sıcaklık Kalibreli)** | **%{p_calibrated[0]*100.0:.1f}** | **%{p_calibrated[1]*100.0:.1f}** | **%{p_calibrated[2]*100.0:.1f}** |\n"

        md += f"""
---

### 2. Beklenen Gol ve Skor Dağılımı

* **Ev Sahibi xG Beklentisi:** **{lam_h:.2f}** [%90 GA: {ci_90_h[0]:.2f} – {ci_90_h[1]:.2f}]
* **Deplasman xG Beklentisi:** **{lam_a:.2f}** [%90 GA: {ci_90_a[0]:.2f} – {ci_90_a[1]:.2f}]

**En Olası Skorlar:**
"""
        for s in top_scores:
            md += f"* **{s['home_goals']} - {s['away_goals']}**: %{s['probability']*100.0:.1f}\n"

        md += f"""
---

### 3. Model ve Piyasa Karşılaştırması

| Gösterge | Ev Sahibi (1) | Beraberlik (X) | Deplasman (2) | Notlar |
| :--- | :---: | :---: | :---: | :--- |
| **Futbol Modeli (Ham)** | %{p_football[0]*100.0:.1f} | %{p_football[1]*100.0:.1f} | %{p_football[2]*100.0:.1f} | Saf Dixon-Coles istatistiksel tahmini |
{cal_row}| **Piyasa (Shin Devigged)** | {p_mkt_h_str} | {p_mkt_x_str} | {p_mkt_a_str} | Kapanış/Canlı oran marjından arındırılmış |
| **Hibrit Model** | {p_hyb_h_str} | {p_hyb_x_str} | {p_hyb_a_str} | İki kaynağın kalibre edilmiş birleşimi |
| **Fark (Model - Piyasa)** | {diff_h_str} | {diff_x_str} | {diff_a_str} | Pozitif fark potansiyel değer sinyali |
| **Beklenen Getiri (EV)** | {ev_1_str} | {ev_x_str} | {ev_2_str} | Teorik EV = p * Oran - 1 |

---

### 4. Olay Tabanlı Monte Carlo Simülasyon Özeti ({mc.n_simulations:,} Tekrar)

* **Simülasyon 1X2 Olasılıkları:** Ev: **%{mc.p_home*100.0:.1f}** | Beraberlik: **%{mc.p_draw*100.0:.1f}** | Deplasman: **%{mc.p_away*100.0:.1f}**
* **Ortalama Maç İçi xG:** {home_name}: **{mc.mean_xg_h:.2f}** | {away_name}: **{mc.mean_xg_a:.2f}**
* **Alt / Üst 2.5:** Üst 2.5: **%{mc.p_over_25*100.0:.1f}** | Alt 2.5: **%{(1.0 - mc.p_over_25)*100.0:.1f}**
* **Karşılıklı Gol (KG):** KG Var: **%{mc.p_btts_yes*100.0:.1f}** | KG Yok: **%{mc.p_btts_no*100.0:.1f}**
* **Beklenen Toplam Korner:** **{mc.mean_corners_tot:.1f}**
* **Beklenen Toplam Kart:** **{mc.mean_cards_tot:.1f}**
* **Geriye Düşenin Puan Kurtarma Oranı (Comeback):** **%{mc.comeback_pct:.1f}**

**Gol Zamanı Dağılımı (Simülasyon Yoğunluğu):**
| Periyot | 0-15' | 15-30' | 30-45' | 45-60' | 60-75' | 75-90+' |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **{home_name}** | %{mc.goal_timing_h.get('0-15', 0)*100.0:.1f} | %{mc.goal_timing_h.get('15-30', 0)*100.0:.1f} | %{mc.goal_timing_h.get('30-45', 0)*100.0:.1f} | %{mc.goal_timing_h.get('45-60', 0)*100.0:.1f} | %{mc.goal_timing_h.get('60-75', 0)*100.0:.1f} | %{mc.goal_timing_h.get('75-90+', 0)*100.0:.1f} |
| **{away_name}** | %{mc.goal_timing_a.get('0-15', 0)*100.0:.1f} | %{mc.goal_timing_a.get('15-30', 0)*100.0:.1f} | %{mc.goal_timing_a.get('30-45', 0)*100.0:.1f} | %{mc.goal_timing_a.get('45-60', 0)*100.0:.1f} | %{mc.goal_timing_a.get('60-75', 0)*100.0:.1f} | %{mc.goal_timing_a.get('75-90+', 0)*100.0:.1f} |

---

### 5. Geçmiş Performans & Metodolojik Uyarısı

> [!NOTE]
> Bu rapor, `match-prediction` modelleme sistemi tarafından üretilmiştir.
> Model tek bir maç için kesin sonuç veya garanti getiri vadetmez. Olasılık tahminleri geçmiş 1.793 walk-forward maç üzerinde test edilmiştir (Out-of-Sample Model B RPS: 0.2022 vs Naif Taban RPS: 0.2276; Naif Tabana Karşı RPS Beceri Skoru: %11.2; Kapanış Piyasası Model A Referansı RPS: 0.1908). MatchEngine simülasyon çıktıları taktiksel ve olay tabanlı senaryo dağılımını temsil eder.
"""
        return md
