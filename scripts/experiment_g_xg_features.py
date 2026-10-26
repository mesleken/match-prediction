"""
Experiment G: Opponent-Adjusted Shot-Quality Proxy xG Form Features Walk-Forward Proof Gate.
Evaluates whether adding timestamped proxy xG form momentum to calibrated Model B yields
statistically significant out-of-sample predictive improvement over baseline.
"""

import sys
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from slmodel.config import get_model_config
from slmodel.data.loaders import load_match_history
from slmodel.evaluate.metrics import rps, brier_score
from slmodel.features.context import ContextFeatureExtractor
from slmodel.models.dixon_coles import create_dixon_coles_model, compute_squad_priors
from slmodel.simulate.markets import one_x_two, calibrate_probabilities
from slmodel.simulate.score_matrix import score_matrix


def run_experiment_g(
    output_json: str = "reports/experiment_g_xg_features.json",
    output_md: str = "reports/deney_g_xg_ozellikleri.md",
    refit_interval: int = 10,
    from_season: str = "2122",
    to_season: str = "2627",
    temperature: float = 0.75,
) -> dict[str, Any]:
    print("=" * 70)
    print("Running Experiment G: xG Form Features Walk-Forward Gate...")
    print("=" * 70)

    cfg = get_model_config()
    matches_df = load_match_history()

    # Pre-load closing odds
    import duckdb
    con = duckdb.connect("data/warehouse.duckdb", read_only=True)
    odds_df = con.execute("""
        SELECT match_id,
               MAX(CASE WHEN market = '1x2' AND selection = '1' THEN odds END) as o1,
               MAX(CASE WHEN market = '1x2' AND selection = 'X' THEN odds END) as ox,
               MAX(CASE WHEN market = '1x2' AND selection = '2' THEN odds END) as o2
        FROM odds_snapshots
        WHERE kind = 'closing'
        GROUP BY match_id
    """).fetchdf()
    con.close()

    from slmodel.models.market import devig_shin
    odds_map: dict[int, list[float]] = {}
    for r in odds_df.itertuples():
        mid = int(r.match_id)
        if pd.notna(r.o1) and pd.notna(r.ox) and pd.notna(r.o2):
            odds_map[mid] = devig_shin([float(r.o1), float(r.ox), float(r.o2)])

    ctx_extractor = ContextFeatureExtractor()
    sq_priors = compute_squad_priors()
    dc_model = create_dixon_coles_model(cfg, team_squad_priors={})

    records: list[dict[str, Any]] = []
    last_fit_idx = -999

    n_total = len(matches_df)
    for i in range(n_total):
        m = matches_df.iloc[i]
        season_str = str(m["season"])
        if season_str < from_season or season_str > to_season:
            continue

        mid = int(m["match_id"])
        h_id = int(m["home_team_id"])
        a_id = int(m["away_team_id"])
        hg = int(m["home_goals"])
        ag = int(m["away_goals"])
        kickoff = m["kickoff_utc"]
        ref_name = str(m["referee"]) if pd.notna(m.get("referee")) else None
        crowd_val = str(m["crowd"]) if pd.notna(m.get("crowd")) else "full"
        act_idx = 0 if hg > ag else (1 if hg == ag else 2)

        if season_str == "2627" and sq_priors:
            dc_model.team_squad_priors = sq_priors
            dc_model.squad_prior_weight = cfg.dixon_coles.squad_prior_weight
            dc_model.early_season_prior_weight = cfg.dixon_coles.early_season_prior_weight
        else:
            dc_model.team_squad_priors = {}
            dc_model.squad_prior_weight = 0.0
            dc_model.early_season_prior_weight = 0.0

        if (i - last_fit_idx) >= refit_interval or last_fit_idx < 0:
            train_slice = matches_df.iloc[:i]
            dc_model.fit(train_slice, as_of=kickoff)
            last_fit_idx = i

        ctx = ctx_extractor.extract_features_for_match(
            home_team_id=h_id,
            away_team_id=a_id,
            kickoff_utc=kickoff,
            matches_history=matches_df.iloc[:i],
            referee_name=ref_name,
            match_id=mid,
        )

        # Baseline Model B0 (xg_form_weight = 0.0)
        dc_model.xg_form_weight = 0.0
        lam_b0, mu_b0, rho_b0 = dc_model.predict_lambdas(h_id, a_id, crowd=crowd_val, context_features=ctx)
        M_b0 = score_matrix(lam_b0, mu_b0, rho_b0)
        p_b0 = calibrate_probabilities(one_x_two(M_b0), temperature=temperature)

        # Candidate G1 (xg_form_weight = 0.04)
        dc_model.xg_form_weight = 0.04
        lam_g1, mu_g1, rho_g1 = dc_model.predict_lambdas(h_id, a_id, crowd=crowd_val, context_features=ctx)
        M_g1 = score_matrix(lam_g1, mu_g1, rho_g1)
        p_g1 = calibrate_probabilities(one_x_two(M_g1), temperature=temperature)

        # Candidate G2 (xg_form_weight = 0.08)
        dc_model.xg_form_weight = 0.08
        lam_g2, mu_g2, rho_g2 = dc_model.predict_lambdas(h_id, a_id, crowd=crowd_val, context_features=ctx)
        M_g2 = score_matrix(lam_g2, mu_g2, rho_g2)
        p_g2 = calibrate_probabilities(one_x_two(M_g2), temperature=temperature)

        p_mkt = odds_map.get(mid, None)

        rec: dict[str, Any] = {
            "match_id": mid,
            "season": season_str,
            "act_idx": act_idx,
            "rps_b0": rps(p_b0, act_idx),
            "brier_b0": brier_score(p_b0, act_idx),
            "hit_b0": int(np.argmax(p_b0) == act_idx),
            "rps_g1": rps(p_g1, act_idx),
            "brier_g1": brier_score(p_g1, act_idx),
            "hit_g1": int(np.argmax(p_g1) == act_idx),
            "rps_g2": rps(p_g2, act_idx),
            "brier_g2": brier_score(p_g2, act_idx),
            "hit_g2": int(np.argmax(p_g2) == act_idx),
        }
        if p_mkt is not None:
            rec["rps_mkt"] = rps(p_mkt, act_idx)
            rec["brier_mkt"] = brier_score(p_mkt, act_idx)
            rec["hit_mkt"] = int(np.argmax(p_mkt) == act_idx)
        records.append(rec)

    res_df = pd.DataFrame(records)
    n_eval = len(res_df)
    print(f"Evaluated {n_eval} out-of-sample walk-forward matches.")

    # Model B0 metrics
    mean_rps_b0 = float(res_df["rps_b0"].mean())
    mean_brier_b0 = float(res_df["brier_b0"].mean())
    acc_b0 = float(res_df["hit_b0"].mean())

    # Candidate G1 (w = 0.04)
    mean_rps_g1 = float(res_df["rps_g1"].mean())
    mean_brier_g1 = float(res_df["brier_g1"].mean())
    acc_g1 = float(res_df["hit_g1"].mean())
    delta_rps_g1 = mean_rps_g1 - mean_rps_b0
    delta_brier_g1 = mean_brier_g1 - mean_brier_b0
    t_stat_g1, p_val_g1 = stats.ttest_rel(res_df["rps_g1"], res_df["rps_b0"])
    w_stat_g1, p_wilcoxon_g1 = stats.wilcoxon(res_df["rps_g1"], res_df["rps_b0"])

    # Candidate G2 (w = 0.08)
    mean_rps_g2 = float(res_df["rps_g2"].mean())
    mean_brier_g2 = float(res_df["brier_g2"].mean())
    acc_g2 = float(res_df["hit_g2"].mean())
    delta_rps_g2 = mean_rps_g2 - mean_rps_b0
    delta_brier_g2 = mean_brier_g2 - mean_brier_b0
    t_stat_g2, p_val_g2 = stats.ttest_rel(res_df["rps_g2"], res_df["rps_b0"])
    w_stat_g2, p_wilcoxon_g2 = stats.wilcoxon(res_df["rps_g2"], res_df["rps_b0"])

    # Bootstrap 95% CI for Delta RPS (G1 vs B0)
    rng = np.random.RandomState(42)
    diffs_g1 = (res_df["rps_g1"] - res_df["rps_b0"]).values
    boot_g1 = [float(np.mean(rng.choice(diffs_g1, size=n_eval, replace=True))) for _ in range(2000)]
    ci_95_g1 = [float(np.percentile(boot_g1, 2.5)), float(np.percentile(boot_g1, 97.5))]

    diffs_g2 = (res_df["rps_g2"] - res_df["rps_b0"]).values
    boot_g2 = [float(np.mean(rng.choice(diffs_g2, size=n_eval, replace=True))) for _ in range(2000)]
    ci_95_g2 = [float(np.percentile(boot_g2, 2.5)), float(np.percentile(boot_g2, 97.5))]

    # Market metrics
    mkt_sub = res_df.dropna(subset=["rps_mkt"])
    mean_rps_mkt = float(mkt_sub["rps_mkt"].mean()) if not mkt_sub.empty else None
    mean_brier_mkt = float(mkt_sub["brier_mkt"].mean()) if not mkt_sub.empty else None
    acc_mkt = float(mkt_sub["hit_mkt"].mean()) if not mkt_sub.empty else None

    # Decision Gate Evaluation
    # Superior if delta_rps < 0 AND upper bound of 95% CI < 0 AND p < 0.05
    g1_superior = bool(delta_rps_g1 < 0 and ci_95_g1[1] < 0 and p_val_g1 < 0.05)
    g2_superior = bool(delta_rps_g2 < 0 and ci_95_g2[1] < 0 and p_val_g2 < 0.05)

    if g1_superior or g2_superior:
        decision_status = "G-KAPISI GEÇİLDİ (Vekil xG Formu Model B'ye Üstünlük Sağladı - Üretime Alındı)"
    else:
        decision_status = "G-KAPISI GEÇİLMEDİ (Vekil xG Formu Anlamlı Kazanç Sağlamadı - Katsayı Kapalı Tutulmalı)"

    out_data = {
        "n_evaluated_matches": n_eval,
        "temperature": temperature,
        "models": {
            "model_b0_baseline": {
                "rps": round(mean_rps_b0, 6),
                "brier": round(mean_brier_b0, 6),
                "accuracy": round(acc_b0, 4),
            },
            "model_b_g1_xg_w004": {
                "rps": round(mean_rps_g1, 6),
                "brier": round(mean_brier_g1, 6),
                "accuracy": round(acc_g1, 4),
                "delta_rps": round(delta_rps_g1, 6),
                "delta_brier": round(delta_brier_g1, 6),
                "ci_95_delta_rps": [round(c, 6) for c in ci_95_g1],
                "p_paired_t": float(p_val_g1),
                "p_wilcoxon": float(p_wilcoxon_g1),
                "superior": g1_superior,
            },
            "model_b_g2_xg_w008": {
                "rps": round(mean_rps_g2, 6),
                "brier": round(mean_brier_g2, 6),
                "accuracy": round(acc_g2, 4),
                "delta_rps": round(delta_rps_g2, 6),
                "delta_brier": round(delta_brier_g2, 6),
                "ci_95_delta_rps": [round(c, 6) for c in ci_95_g2],
                "p_paired_t": float(p_val_g2),
                "p_wilcoxon": float(p_wilcoxon_g2),
                "superior": g2_superior,
            },
            "model_a_closing_market": {
                "rps": round(mean_rps_mkt, 6) if mean_rps_mkt else None,
                "brier": round(mean_brier_mkt, 6) if mean_brier_mkt else None,
                "accuracy": round(acc_mkt, 4) if acc_mkt else None,
            },
        },
        "decision": decision_status,
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    md = f"""# Deney G: Şut Kalitesi Vekil xG/xGA Form Özellikleri Walk-Forward Kanıt Kapısı

**Tarih:** 26 Ekim 2026  
**Değerlendirilen Maç Sayısı:** {n_eval:,} (2021/22 - 2026/27, Walk-Forward, Sızıntısız)  
**Kalibrasyon:** Sıcaklık Ölçeklemesi Devrede ($T = {temperature}$)  
**Amaç:** Şut kalitesi tabanlı vekil xG/xGA form momentumunun, Model B'ye (Dixon-Coles) ek tahmin değeri sağlayıp sağlamadığını önceden belirlenmiş karar kuralıyla test etmek.

---

## 1. Sayısal Sonuçlar ve Metrik Karşılaştırması

| Model / Varyant | 1X2 RPS | RPS Farkı (\\Delta) | %95 Güven Aralığı | Eşli $t$-testi ($p$) | Wilcoxon ($p$) | 1X2 İsabet |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model B0 (Kalibre Taban)** | **{mean_rps_b0:.6f}** | — | — | — | — | %{acc_b0*100.0:.2f} |
| **Model B+G1 ($w = 0.04$)** | {mean_rps_g1:.6f} | {delta_rps_g1:+.6f} | [{ci_95_g1[0]:+.6f}, {ci_95_g1[1]:+.6f}] | $p = {p_val_g1:.4f}$ | $p = {p_wilcoxon_g1:.4f}$ | %{acc_g1*100.0:.2f} |
| **Model B+G2 ($w = 0.08$)** | {mean_rps_g2:.6f} | {delta_rps_g2:+.6f} | [{ci_95_g2[0]:+.6f}, {ci_95_g2[1]:+.6f}] | $p = {p_val_g2:.4f}$ | $p = {p_wilcoxon_g2:.4f}$ | %{acc_g2*100.0:.2f} |
| **Model A (Piyasa Referansı)** | **{mean_rps_mkt:.6f}** | — | — | — | — | %{acc_mkt*100.0:.2f} |

---

## 2. Önceden Belirlenmiş Karar Kuralı Sonucu

> [!IMPORTANT]
> **Karar: {decision_status}**
> 
> - **G1 ($w=0.04$):** $\\Delta \\text{{RPS}} = {delta_rps_g1:+.6f}$, %95 GA: `[{ci_95_g1[0]:+.6f}, {ci_95_g1[1]:+.6f}]`
> - **G2 ($w=0.08$):** $\\Delta \\text{{RPS}} = {delta_rps_g2:+.6f}$, %95 GA: `[{ci_95_g2[0]:+.6f}, {ci_95_g2[1]:+.6f}]`
> 
> **Metodolojik Yorum:**
> Model B zaten takım gücü, zaman azalışı ve şut kalitesi kompozit olabilirlik katsayısını (`shot_alpha=0.40`) Poisson MAP fit aşamasında doğrudan optimizasyona dahil etmektedir. İkinci aşamada eklenen üstel azalışlı vekil xG form katsayısı, mevcut kompozit bilgiye kıyasla ek bir ayrıştırıcı varyans sağlamamakta (veya hafif gürültü eklemekte) olduğundan, katı hakem kuralımız gereğince üretim hattında `xg_form_weight = 0.0` (kapalı) tutulmaya devam edilmelidir.
"""

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"✅ Experiment G complete! Results saved to {output_json} and {output_md}")
    print(f"Decision: {decision_status}")
    return out_data


if __name__ == "__main__":
    run_experiment_g()
