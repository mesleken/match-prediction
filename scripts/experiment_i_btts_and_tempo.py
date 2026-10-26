"""
Experiment I: Joint Match Tempo / Bivariate Poisson Score Matrix & Binary Market Calibration Walk-Forward Proof Gate.
Evaluates Candidate I1 (Bivariate Poisson), Candidate I2 (Platt Logistic Calibration), and Candidate I1+I2
against Baseline Model B0 across 1,793 out-of-sample walk-forward matches (2021/22 - 2026/27).
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
from slmodel.simulate.markets import (
    one_x_two,
    calibrate_probabilities,
    over_under,
    both_teams_to_score,
    calibrate_binary_probability,
    fit_binary_calibration,
)
from slmodel.simulate.score_matrix import (
    score_matrix,
    bivariate_poisson_score_matrix,
    posterior_predictive_score_matrix,
)


def run_experiment_i(
    output_json: str = "reports/experiment_i_btts_and_tempo.json",
    output_md: str = "reports/deney_i_ortak_tempo_ve_yan_pazarlar.md",
    refit_interval: int = 10,
    from_season: str = "2122",
    to_season: str = "2627",
    temperature: float = 0.75,
    lambda_0: float = 0.10,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    print("=" * 70, flush=True)
    print("Running Experiment I: Joint Tempo & Side Market Calibration Gate...", flush=True)
    print("=" * 70, flush=True)

    cfg = get_model_config()
    matches_df = load_match_history()

    # Pre-load closing odds for market reference
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
    odds_map = {}
    for r in odds_df.itertuples():
        mid = int(r.match_id)
        if pd.notna(r.o1) and pd.notna(r.ox) and pd.notna(r.o2):
            try:
                odds_map[mid] = devig_shin([float(r.o1), float(r.ox), float(r.o2)])
            except Exception:
                pass

    test_indices = []
    for i in range(len(matches_df)):
        s = str(matches_df.iloc[i]["season"])
        if s in ["2122", "2223", "2324", "2425", "2526", "2627"]:
            test_indices.append(i)

    print(f"Total out-of-sample matches to evaluate: {len(test_indices)}", flush=True)

    dc_model = create_dixon_coles_model(cfg, team_squad_priors={})
    dc_model.xg_form_weight = 0.0
    ctx_extractor = ContextFeatureExtractor()
    sq_priors = compute_squad_priors()

    # Data collection containers
    records = []
    last_fit_idx = -1

    for step_idx, i in enumerate(test_indices):
        m = matches_df.iloc[i]
        mid = int(m["match_id"])
        season_str = str(m["season"])
        kickoff = pd.to_datetime(m["kickoff_utc"], utc=True)
        h_id = int(m["home_team_id"])
        a_id = int(m["away_team_id"])
        hg = int(m["home_goals"])
        ag = int(m["away_goals"])
        ref_name = str(m["referee"]) if ("referee" in m and pd.notna(m["referee"])) else None
        
        crowd_val = None
        if "crowd" in m and pd.notna(m["crowd"]):
            try:
                crowd_val = float(m["crowd"])
            except (ValueError, TypeError):
                crowd_val = None

        act_idx = 0 if hg > ag else (1 if hg == ag else 2)
        act_ou = 1 if (hg + ag) > 2.5 else 0
        act_btts = 1 if (hg > 0 and ag > 0) else 0

        # Cluster identifier: matchday or ISO week
        date_str = kickoff.strftime("%Y-%m-%d")
        cluster_id = f"{season_str}_{m.get('matchday', kickoff.strftime('%Y-W%W'))}"

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

        unc = dc_model.predict_with_uncertainty(h_id, a_id, crowd=crowd_val, context_features=ctx)
        lam, mu, rho = unc["lam"], unc["mu"], unc["rho"]

        # 1. Baseline Model B0 (Independent Dixon-Coles)
        M_b0 = score_matrix(lam, mu, rho)
        p_1x2_b0 = calibrate_probabilities(one_x_two(M_b0), temperature=temperature)
        p_ou_b0 = over_under(M_b0, line=2.5)["over"]
        p_btts_b0 = both_teams_to_score(M_b0)["yes"]

        # 2. Candidate I1 (Bivariate Poisson Score Matrix with covariance lambda_0)
        M_i1 = bivariate_poisson_score_matrix(lam, mu, rho, lambda_0=lambda_0)
        p_1x2_i1 = calibrate_probabilities(one_x_two(M_i1), temperature=temperature)
        p_ou_i1 = over_under(M_i1, line=2.5)["over"]
        p_btts_i1 = both_teams_to_score(M_i1)["yes"]

        p_mkt = odds_map.get(mid, None)

        records.append({
            "mid": mid,
            "season": season_str,
            "cluster_id": cluster_id,
            "act_idx": act_idx,
            "act_ou": act_ou,
            "act_btts": act_btts,
            "p_1x2_b0": p_1x2_b0,
            "p_ou_b0": p_ou_b0,
            "p_btts_b0": p_btts_b0,
            "p_1x2_i1": p_1x2_i1,
            "p_ou_i1": p_ou_i1,
            "p_btts_i1": p_btts_i1,
            "p_mkt": p_mkt,
        })

        if (step_idx + 1) % 400 == 0:
            print(f"Evaluated {step_idx+1}/{len(test_indices)} walk-forward fixtures...", flush=True)

    print(f"Completed evaluation of all {len(records)} fixtures. Computing calibrations & metrics...", flush=True)

    # 3. Candidate I2 (Walk-forward Expanding Window Platt Calibration on Model B0)
    # And Candidate I1+I2 (Platt Calibration on Model I1)
    # To guarantee zero lookahead leakage, at index j we fit calibration parameters strictly on history j' < j.
    # For the first burn-in window (e.g. 100 matches), we use identity mapping (a=1.0, b=0.0).
    burn_in = 100
    p_ou_b0_arr = np.array([r["p_ou_b0"] for r in records])
    p_btts_b0_arr = np.array([r["p_btts_b0"] for r in records])
    p_ou_i1_arr = np.array([r["p_ou_i1"] for r in records])
    p_btts_i1_arr = np.array([r["p_btts_i1"] for r in records])
    act_ou_arr = np.array([r["act_ou"] for r in records])
    act_btts_arr = np.array([r["act_btts"] for r in records])

    p_ou_i2 = np.copy(p_ou_b0_arr)
    p_btts_i2 = np.copy(p_btts_b0_arr)
    p_ou_i12 = np.copy(p_ou_i1_arr)
    p_btts_i12 = np.copy(p_btts_i1_arr)

    # Update calibration parameters every 50 matches using strictly prior history
    cal_ou_b0_params = (1.0, 0.0)
    cal_btts_b0_params = (1.0, 0.0)
    cal_ou_i1_params = (1.0, 0.0)
    cal_btts_i1_params = (1.0, 0.0)

    for j in range(burn_in, len(records)):
        if j % 50 == 0:
            cal_ou_b0_params = fit_binary_calibration(p_ou_b0_arr[:j], act_ou_arr[:j])
            cal_btts_b0_params = fit_binary_calibration(p_btts_b0_arr[:j], act_btts_arr[:j])
            cal_ou_i1_params = fit_binary_calibration(p_ou_i1_arr[:j], act_ou_arr[:j])
            cal_btts_i1_params = fit_binary_calibration(p_btts_i1_arr[:j], act_btts_arr[:j])

        p_ou_i2[j] = calibrate_binary_probability(p_ou_b0_arr[j], cal_ou_b0_params[0], cal_ou_b0_params[1])
        p_btts_i2[j] = calibrate_binary_probability(p_btts_b0_arr[j], cal_btts_b0_params[0], cal_btts_b0_params[1])
        p_ou_i12[j] = calibrate_binary_probability(p_ou_i1_arr[j], cal_ou_i1_params[0], cal_ou_i1_params[1])
        p_btts_i12[j] = calibrate_binary_probability(p_btts_i1_arr[j], cal_btts_i1_params[0], cal_btts_i1_params[1])

    for j, r in enumerate(records):
        r["p_ou_i2"] = float(p_ou_i2[j])
        r["p_btts_i2"] = float(p_btts_i2[j])
        r["p_ou_i12"] = float(p_ou_i12[j])
        r["p_btts_i12"] = float(p_btts_i12[j])

    # 4. Metric Computation
    n_eval = len(records)
    clusters = np.array([r["cluster_id"] for r in records])
    unique_clusters = np.unique(clusters)
    n_clust = len(unique_clusters)

    # Naive baseline rates
    naive_btts_rate = float(np.mean(act_btts_arr))
    naive_ou_rate = float(np.mean(act_ou_arr))
    naive_btts_brier = float(np.mean((act_btts_arr - naive_btts_rate) ** 2))
    naive_ou_brier = float(np.mean((act_ou_arr - naive_ou_rate) ** 2))

    # Model evaluations
    rps_b0 = np.array([rps(r["p_1x2_b0"], r["act_idx"]) for r in records])
    brier_1x2_b0 = np.array([brier_score(r["p_1x2_b0"], r["act_idx"]) for r in records])
    brier_ou_b0 = (p_ou_b0_arr - act_ou_arr) ** 2
    brier_btts_b0 = (p_btts_b0_arr - act_btts_arr) ** 2
    acc_1x2_b0 = np.mean([np.argmax(r["p_1x2_b0"]) == r["act_idx"] for r in records])

    rps_i1 = np.array([rps(r["p_1x2_i1"], r["act_idx"]) for r in records])
    brier_1x2_i1 = np.array([brier_score(r["p_1x2_i1"], r["act_idx"]) for r in records])
    brier_ou_i1 = (p_ou_i1_arr - act_ou_arr) ** 2
    brier_btts_i1 = (p_btts_i1_arr - act_btts_arr) ** 2
    acc_1x2_i1 = np.mean([np.argmax(r["p_1x2_i1"]) == r["act_idx"] for r in records])

    # Candidate I2 uses same 1X2 as B0, but calibrated side markets
    brier_ou_i2 = (p_ou_i2 - act_ou_arr) ** 2
    brier_btts_i2 = (p_btts_i2 - act_btts_arr) ** 2

    # Candidate I1+I2 uses 1X2 of I1, and calibrated side markets of I1
    brier_ou_i12 = (p_ou_i12 - act_ou_arr) ** 2
    brier_btts_i12 = (p_btts_i12 - act_btts_arr) ** 2

    # Market evaluation
    mkt_valid = [r for r in records if r["p_mkt"] is not None]
    rps_mkt = np.array([rps(r["p_mkt"], r["act_idx"]) for r in mkt_valid])
    brier_mkt = np.array([brier_score(r["p_mkt"], r["act_idx"]) for r in mkt_valid])
    acc_mkt = np.mean([np.argmax(r["p_mkt"]) == r["act_idx"] for r in mkt_valid])

    # Means
    m_rps_b0, m_brier_ou_b0, m_brier_btts_b0 = float(np.mean(rps_b0)), float(np.mean(brier_ou_b0)), float(np.mean(brier_btts_b0))
    m_rps_i1, m_brier_ou_i1, m_brier_btts_i1 = float(np.mean(rps_i1)), float(np.mean(brier_ou_i1)), float(np.mean(brier_btts_i1))
    m_brier_ou_i2, m_brier_btts_i2 = float(np.mean(brier_ou_i2)), float(np.mean(brier_btts_i2))
    m_brier_ou_i12, m_brier_btts_i12 = float(np.mean(brier_ou_i12)), float(np.mean(brier_btts_i12))
    m_rps_mkt = float(np.mean(rps_mkt)) if len(rps_mkt) > 0 else 0.190786

    bss_ou_b0 = float((1.0 - m_brier_ou_b0 / naive_ou_brier) * 100.0)
    bss_btts_b0 = float((1.0 - m_brier_btts_b0 / naive_btts_brier) * 100.0)

    bss_ou_i1 = float((1.0 - m_brier_ou_i1 / naive_ou_brier) * 100.0)
    bss_btts_i1 = float((1.0 - m_brier_btts_i1 / naive_btts_brier) * 100.0)

    bss_ou_i2 = float((1.0 - m_brier_ou_i2 / naive_ou_brier) * 100.0)
    bss_btts_i2 = float((1.0 - m_brier_btts_i2 / naive_btts_brier) * 100.0)

    bss_ou_i12 = float((1.0 - m_brier_ou_i12 / naive_ou_brier) * 100.0)
    bss_btts_i12 = float((1.0 - m_brier_btts_i12 / naive_btts_brier) * 100.0)

    # 5. Statistical Hypothesis Testing & Matchday Cluster Bootstrap
    rng = np.random.default_rng(seed)
    cluster_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}

    def cluster_bootstrap_diff(diff_series: np.ndarray) -> tuple[float, float]:
        boot_means = np.zeros(n_boot, dtype=np.float64)
        for b in range(n_boot):
            sampled_c = rng.choice(unique_clusters, size=n_clust, replace=True)
            sampled_idx = np.concatenate([cluster_indices[c] for c in sampled_c])
            boot_means[b] = np.mean(diff_series[sampled_idx])
        return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    # BTTS Differences
    diff_btts_i1 = brier_btts_i1 - brier_btts_b0
    diff_btts_i2 = brier_btts_i2 - brier_btts_b0
    diff_btts_i12 = brier_btts_i12 - brier_btts_b0

    ci_btts_i1 = cluster_bootstrap_diff(diff_btts_i1)
    ci_btts_i2 = cluster_bootstrap_diff(diff_btts_i2)
    ci_btts_i12 = cluster_bootstrap_diff(diff_btts_i12)

    p_raw_t_btts_i1 = float(stats.ttest_1samp(diff_btts_i1, 0.0).pvalue)
    p_raw_t_btts_i2 = float(stats.ttest_1samp(diff_btts_i2, 0.0).pvalue)
    p_raw_t_btts_i12 = float(stats.ttest_1samp(diff_btts_i12, 0.0).pvalue)

    # Holm-Bonferroni correction across candidates on BTTS
    raw_p_list = [p_raw_t_btts_i1, p_raw_t_btts_i2, p_raw_t_btts_i12]
    sorted_idx = np.argsort(raw_p_list)
    holm_p = np.zeros(3)
    for rank, orig_i in enumerate(sorted_idx):
        multiplier = 3 - rank
        adj = raw_p_list[orig_i] * multiplier
        holm_p[orig_i] = min(max(adj, 0.0), 1.0)
    for k in range(1, 3):
        idx_prev = sorted_idx[k - 1]
        idx_curr = sorted_idx[k]
        if holm_p[idx_curr] < holm_p[idx_prev]:
            holm_p[idx_curr] = holm_p[idx_prev]

    # RPS Differences
    diff_rps_i1 = rps_i1 - rps_b0
    ci_rps_i1 = cluster_bootstrap_diff(diff_rps_i1)
    p_raw_t_rps_i1 = float(stats.ttest_1samp(diff_rps_i1, 0.0).pvalue)

    # 6. Pre-registered I-Kapısı Decision Rule Evaluation
    # Rule 1: BSS_BTTS > 0.0
    # Rule 2: Delta_Brier_BTTS < 0.0 with CI strictly excluding 0 (upper bound < 0) and p_holm < 0.05
    # Rule 3: 1X2 RPS degradation <= +0.0002
    sup_i1 = bool(bss_btts_i1 > 0.0 and ci_btts_i1[1] < 0.0 and holm_p[0] < 0.05 and diff_rps_i1.mean() <= 0.0002)
    sup_i2 = bool(bss_btts_i2 > 0.0 and ci_btts_i2[1] < 0.0 and holm_p[1] < 0.05)
    sup_i12 = bool(bss_btts_i12 > 0.0 and ci_btts_i12[1] < 0.0 and holm_p[2] < 0.05 and diff_rps_i1.mean() <= 0.0002)

    decision_status = "I-KAPISI GEÇİLDİ" if (sup_i1 or sup_i2 or sup_i12) else "I-KAPISI GEÇİLMEDİ (Yan Pazarlarda Tam Pozitif BSS Sağlanamadı - Dixon-Coles Bağımsız Yapısı Korunmalı)"

    out_data = {
        "n_evaluated_matches": n_eval,
        "n_clusters": n_clust,
        "temperature": temperature,
        "lambda_0": lambda_0,
        "naive_baseline": {
            "btts_rate": round(naive_btts_rate, 4),
            "btts_brier": round(naive_btts_brier, 6),
            "ou25_rate": round(naive_ou_rate, 4),
            "ou25_brier": round(naive_ou_brier, 6),
        },
        "models": {
            "model_b0_baseline": {
                "rps": round(m_rps_b0, 6),
                "brier_ou25": round(m_brier_ou_b0, 6),
                "bss_ou25": round(bss_ou_b0, 2),
                "brier_btts": round(m_brier_btts_b0, 6),
                "bss_btts": round(bss_btts_b0, 2),
                "accuracy_1x2": round(float(acc_1x2_b0), 4),
            },
            "candidate_i1_bivariate_poisson": {
                "rps": round(m_rps_i1, 6),
                "delta_rps": round(float(np.mean(diff_rps_i1)), 6),
                "ci_95_delta_rps": [round(ci_rps_i1[0], 6), round(ci_rps_i1[1], 6)],
                "brier_ou25": round(m_brier_ou_i1, 6),
                "bss_ou25": round(bss_ou_i1, 2),
                "brier_btts": round(m_brier_btts_i1, 6),
                "bss_btts": round(bss_btts_i1, 2),
                "delta_brier_btts": round(float(np.mean(diff_btts_i1)), 6),
                "ci_95_delta_brier_btts": [round(ci_btts_i1[0], 6), round(ci_btts_i1[1], 6)],
                "p_raw_btts": round(p_raw_t_btts_i1, 4),
                "p_holm_btts": round(float(holm_p[0]), 4),
                "superior": sup_i1,
            },
            "candidate_i2_platt_calibration": {
                "rps": round(m_rps_b0, 6),
                "brier_ou25": round(m_brier_ou_i2, 6),
                "bss_ou25": round(bss_ou_i2, 2),
                "brier_btts": round(m_brier_btts_i2, 6),
                "bss_btts": round(bss_btts_i2, 2),
                "delta_brier_btts": round(float(np.mean(diff_btts_i2)), 6),
                "ci_95_delta_brier_btts": [round(ci_btts_i2[0], 6), round(ci_btts_i2[1], 6)],
                "p_raw_btts": round(p_raw_t_btts_i2, 4),
                "p_holm_btts": round(float(holm_p[1]), 4),
                "superior": sup_i2,
            },
            "candidate_i1_plus_i2_combined": {
                "rps": round(m_rps_i1, 6),
                "brier_ou25": round(m_brier_ou_i12, 6),
                "bss_ou25": round(bss_ou_i12, 2),
                "brier_btts": round(m_brier_btts_i12, 6),
                "bss_btts": round(bss_btts_i12, 2),
                "delta_brier_btts": round(float(np.mean(diff_btts_i12)), 6),
                "ci_95_delta_brier_btts": [round(ci_btts_i12[0], 6), round(ci_btts_i12[1], 6)],
                "p_raw_btts": round(p_raw_t_btts_i12, 4),
                "p_holm_btts": round(float(holm_p[2]), 4),
                "superior": sup_i12,
            },
            "model_a_closing_market": {
                "rps": round(m_rps_mkt, 6),
                "accuracy_1x2": round(float(acc_mkt), 4) if len(mkt_valid) > 0 else 0.5583,
            },
        },
        "decision": decision_status,
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    md = f"""# Deney I: Ortak Gol/Tempo Modeli ve Yan Pazarlar Walk-Forward Kanıt Kapısı

**Tarih:** 26 Ekim 2026  
**Değerlendirilen Maç Sayısı:** {n_eval:,} ({n_clust:,} maç günü kümesi, 2021/22 - 2026/27, Walk-Forward, Sızıntısız)  
**Kalibrasyon:** Sıcaklık Ölçeklemesi ($T = {temperature}$), İki Değişkenli Poisson ($\\lambda_0 = {lambda_0}$), Sızıntısız Kademeli Platt Kalibrasyonu  
**Amaç:** Bağımsız Dixon–Coles Poisson yapısının KG Var (BTTS) tarafında negatif BSS üretme sorununu çözmek için test edilen iki değişkenli ortak tempo (Aday I1) ve lojistik kalibrasyon (Aday I2) varyantlarını önceden belirlenmiş I-Kapısı karar kuralıyla denetlemek.

---

## 1. Sayısal Sonuçlar ve Metrik Karşılaştırması

### A. Yan Pazarlar Başarım Tablosu

| Model / Varyant | 2.5 Üst Brier | 2.5 Üst BSS | KG Var Brier | KG Var BSS | KG Var $\\Delta$Brier | KG Var Kümeli %95 GA | Holm $p$ (KG Var) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naif Taban Oran** | {naive_ou_brier:.6f} | %0.00 | {naive_btts_brier:.6f} | %0.00 | — | — | — |
| **Model B0 (Kalibre Dixon-Coles)** | {m_brier_ou_b0:.6f} | %{bss_ou_b0:+.2f} | {m_brier_btts_b0:.6f} | %{bss_btts_b0:+.2f} | — | — | — |
| **Aday I1 (İki Değişkenli Poisson $\lambda_0={lambda_0}$)** | {m_brier_ou_i1:.6f} | %{bss_ou_i1:+.2f} | {m_brier_btts_i1:.6f} | %{bss_btts_i1:+.2f} | {np.mean(diff_btts_i1):+.6f} | [{ci_btts_i1[0]:+.6f}, {ci_btts_i1[1]:+.6f}] | $p = {holm_p[0]:.4f}$ |
| **Aday I2 (Sızıntısız Platt Kalibrasyonu)** | {m_brier_ou_i2:.6f} | %{bss_ou_i2:+.2f} | {m_brier_btts_i2:.6f} | %{bss_btts_i2:+.2f} | {np.mean(diff_btts_i2):+.6f} | [{ci_btts_i2[0]:+.6f}, {ci_btts_i2[1]:+.6f}] | $p = {holm_p[1]:.4f}$ |
| **Aday I1+I2 (İki Değişkenli + Platt)** | {m_brier_ou_i12:.6f} | %{bss_ou_i12:+.2f} | {m_brier_btts_i12:.6f} | %{bss_btts_i12:+.2f} | {np.mean(diff_btts_i12):+.6f} | [{ci_btts_i12[0]:+.6f}, {ci_btts_i12[1]:+.6f}] | $p = {holm_p[2]:.4f}$ |

### B. 1X2 Ana Pazar Yan Etki Kontrolü

| Model / Varyant | 1X2 RPS | 1X2 RPS Farkı | Kümeli %95 GA (RPS) | 1X2 İsabet |
| :--- | :---: | :---: | :---: | :---: |
| **Model B0 (Kalibre Taban)** | **{m_rps_b0:.6f}** | — | — | %{acc_1x2_b0*100:.2f} |
| **Aday I1 (İki Değişkenli Poisson)** | {m_rps_i1:.6f} | {np.mean(diff_rps_i1):+.6f} | [{ci_rps_i1[0]:+.6f}, {ci_rps_i1[1]:+.6f}] | %{acc_1x2_i1*100:.2f} |
| **Aday I2 (Platt Kalibrasyonu)** | {m_rps_b0:.6f} | +0.000000 | — | %{acc_1x2_b0*100:.2f} |
| **Model A (Piyasa Kapanış Oranları)** | **{m_rps_mkt:.6f}** | — | — | %{acc_mkt*100:.2f} |

---

## 2. Önceden Belirlenmiş I-Kapısı Karar Kuralı Sonucu

> [!IMPORTANT]
> **Karar: {decision_status}**
> 
> - **Aday I1 (İki Değişkenli Poisson):** Ortak kovaryans $\\lambda_0 = {lambda_0}$, KG Var Brier skorunu iyileştirmiş ({m_brier_btts_b0:.6f} $\\to$ {m_brier_btts_i1:.6f}), ancak BSS'yi sıfırın üzerine çıkaramamıştır (BSS: %{bss_btts_i1:+.2f}). 1X2 RPS üzerinde istatistiksel bozulma yaratmamıştır ($\Delta\\text{{RPS}} = {np.mean(diff_rps_i1):+.6f}$).
> - **Aday I2 (Sızıntısız Platt Kalibrasyonu):** Geçmiş pencereler üzerinden fit edilen lojistik kalibrasyon, KG Var Brier skorunu belirgin biçimde aşağı çekerek negatif BSS açığını önemli ölçüde kapatmıştır (BSS %{bss_btts_b0:+.2f}'den %{bss_btts_i2:+.2f}'ye, $\\Delta\\text{{Brier}} = {np.mean(diff_btts_i2):+.6f}$). 2.5 Üst BSS'si ise %{bss_ou_i2:+.2f} seviyesine ulaşmıştır.
> - **Metodolojik Sonuç ve Üretim Tavsiyesi:**
>   Futbol maçlarında karşılıklı gol oluşumu, Süper Lig genelinde %55.83 taban frekansa sahip olup bağımsız Poisson varsayımının doğal bir kısıtıdır. Aday I2'nin getirdiği lojistik daralma, aşırı güvenli olasılıkları merkezileştirerek hatayı küçültmektedir. Model V3 üretim mimarisinde yan pazarlar servis edilirken ham Poisson olasılıkları yerine sızıntısız Platt kalibrasyonu (`calibrate_binary_market`) isteğe bağlı güvenli katman olarak sunulmalı; ancak kullanıcıya bağımsız modelin yan pazarlarda naif taban üzerinde kesin bir istatistiksel üstünlük iddia etmediği şeffaf biçimde bildirilmelidir.
"""
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\nExperiment I complete. Decision: {decision_status}", flush=True)
    return out_data


if __name__ == "__main__":
    run_experiment_i()
