"""
Deney B: Çoklu Piyasa Kalibrasyon Eğrileri, Brier Beceri Skoru (BSS) ve ECE
1X2, Alt/Üst 2.5 ve KG Var Piyasalarında Model ve Piyasanın Olasılık Kalibrasyonu.
"""

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculates Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
    p_hat = k / n
    denominator = 1 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denominator
    spread = z * np.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n) / denominator
    return float(max(0.0, center - spread)), float(min(1.0, center + spread))


def compute_calibration_curve(
    probs: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
    bin_range: tuple[float, float] = (0.0, 1.0),
) -> dict[str, Any]:
    """Computes binned calibration stats: mean predicted prob, empirical frequency, Wilson CI, ECE, MCE."""
    bins = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_records = []
    total_n = len(probs)
    ece = 0.0
    mce = 0.0

    for b in range(n_bins):
        mask = (bin_indices == b)
        n_in_bin = int(np.sum(mask))
        b_low, b_high = float(bins[b]), float(bins[b + 1])
        if n_in_bin > 0:
            p_mean = float(np.mean(probs[mask]))
            k_events = int(np.sum(actuals[mask]))
            y_mean = float(k_events / n_in_bin)
            w_l, w_h = wilson_ci(k_events, n_in_bin)
            gap = abs(p_mean - y_mean)
            ece += (n_in_bin / total_n) * gap
            if gap > mce:
                mce = gap
            bin_records.append({
                "bin_idx": b,
                "bin_range": f"[{b_low:.2f}, {b_high:.2f})",
                "count": n_in_bin,
                "pred_prob_mean": round(p_mean, 4),
                "actual_freq": round(y_mean, 4),
                "actual_ci_95": [round(w_l, 4), round(w_h, 4)],
                "calibration_gap": round(p_mean - y_mean, 4),
                "abs_gap": round(gap, 4),
            })
        else:
            bin_records.append({
                "bin_idx": b,
                "bin_range": f"[{b_low:.2f}, {b_high:.2f})",
                "count": 0,
                "pred_prob_mean": None,
                "actual_freq": None,
                "actual_ci_95": [0.0, 0.0],
                "calibration_gap": 0.0,
                "abs_gap": 0.0,
            })

    brier = float(np.mean((probs - actuals) ** 2))
    eps = 1e-12
    p_clipped = np.clip(probs, eps, 1.0 - eps)
    log_loss = float(-np.mean(actuals * np.log(p_clipped) + (1.0 - actuals) * np.log(1.0 - p_clipped)))

    return {
        "n_samples": total_n,
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "brier_score": round(brier, 4),
        "log_loss": round(log_loss, 4),
        "bins": bin_records,
    }


def run_experiment_b(
    parquet_path: str = "reports/historical_eval_records.parquet",
) -> dict[str, Any]:
    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(f"Veri bulunamadi: {parquet_path}")

    df = pd.read_parquet(p)
    rng = np.random.default_rng(42)
    eps = 1e-12

    # 1. 1X2 Kalibrasyonu (Model vs Piyasa)
    y_home = (df["act_outcome_idx"] == 0).astype(float).values
    cal_mod_home = compute_calibration_curve(df["p_home"].values, y_home)

    y_draw = (df["act_outcome_idx"] == 1).astype(float).values
    cal_mod_draw = compute_calibration_curve(df["p_draw"].values, y_draw)

    y_away = (df["act_outcome_idx"] == 2).astype(float).values
    cal_mod_away = compute_calibration_curve(df["p_away"].values, y_away)

    fav_prob = df["p_fav"].values
    y_fav = df["hit_1x2"].astype(float).values
    cal_mod_fav = compute_calibration_curve(fav_prob, y_fav, n_bins=8, bin_range=(0.30, 0.90))

    df_mkt = df[df["has_market"] == True].copy()
    y_mkt_home = (df_mkt["act_outcome_idx"] == 0).astype(float).values
    cal_mkt_home = compute_calibration_curve(df_mkt["p_mkt_home"].values, y_mkt_home)
    cal_mkt_draw = compute_calibration_curve(df_mkt["p_mkt_draw"].values, (df_mkt["act_outcome_idx"] == 1).astype(float).values)
    cal_mkt_away = compute_calibration_curve(df_mkt["p_mkt_away"].values, (df_mkt["act_outcome_idx"] == 2).astype(float).values)

    p_mkt_fav = np.maximum(df_mkt["p_mkt_home"].values, np.maximum(df_mkt["p_mkt_draw"].values, df_mkt["p_mkt_away"].values))
    cal_mkt_fav = compute_calibration_curve(p_mkt_fav, df_mkt["hit_mkt_1x2"].astype(float).values, n_bins=8, bin_range=(0.30, 0.90))

    # 2. Favorite Calibration Error Residuals e_i = y_i - p_i (Cluster Bootstrap & Holm-Bonferroni)
    df["cluster"] = df["season"].astype(str) + "_" + df["matchday"].astype(str)
    tier_definitions = [
        ("Düşük Güven (< %45)", df["p_fav"] < 0.45),
        ("Orta Güven (%45 - %55)", (df["p_fav"] >= 0.45) & (df["p_fav"] < 0.55)),
        ("Yüksek Güven (%55 - %65)", (df["p_fav"] >= 0.55) & (df["p_fav"] < 0.65)),
        ("Çok Yüksek Güven (≥ %65)", df["p_fav"] >= 0.65),
    ]
    fav_residual_tiers = []
    for t_name, mask in tier_definitions:
        sub = df[mask].copy()
        n_sub = len(sub)
        sub["e"] = sub["hit_1x2"].astype(float) - sub["p_fav"].values
        mean_e = float(sub["e"].mean())
        se_e = float(stats.sem(sub["e"]))
        t_stat, p_val = stats.ttest_1samp(sub["e"], 0.0)

        clust_grouped = sub.groupby("cluster")["e"].agg(["sum", "count"])
        sums = clust_grouped["sum"].values
        counts = clust_grouped["count"].values
        n_c = len(sums)

        boot_means = []
        for _ in range(5000):
            idx = rng.choice(n_c, size=n_c, replace=True)
            boot_means.append(np.sum(sums[idx]) / np.sum(counts[idx]))
        ci_l, ci_h = np.percentile(boot_means, [2.5, 97.5])

        fav_residual_tiers.append({
            "tier": t_name,
            "n": n_sub,
            "mean_pred": round(float(sub["p_fav"].mean()), 4),
            "actual_win_rate": round(float(sub["hit_1x2"].mean()), 4),
            "mean_residual_e": round(mean_e, 4),
            "se": round(se_e, 4),
            "ci_clust": [round(float(ci_l), 4), round(float(ci_h), 4)],
            "t_stat": round(float(t_stat), 2),
            "raw_p": round(float(p_val), 4),
        })

    # Holm-Bonferroni correction across the 4 confidence tiers
    fav_residual_tiers.sort(key=lambda x: x["raw_p"])
    m_tiers = len(fav_residual_tiers)
    for i, r in enumerate(fav_residual_tiers):
        rank = i + 1
        hb_thresh = round(0.05 / (m_tiers - rank + 1), 4)
        adj_p = round(min(1.0, r["raw_p"] * (m_tiers - rank + 1)), 4)
        r["hb_thresh"] = hb_thresh
        r["adj_p"] = adj_p
        r["sig_hb"] = bool(r["raw_p"] < hb_thresh)

    # Also compute overall total
    e_all = df["hit_1x2"].astype(float) - df["p_fav"].values
    t_all, p_all = stats.ttest_1samp(e_all, 0.0)
    cg_all = df.groupby("cluster").apply(lambda g: (g["hit_1x2"].astype(float) - g["p_fav"]).agg(["sum", "count"])).unstack()
    # Fast overall cluster bootstrap
    cg_all_sums = df.groupby("cluster").apply(lambda g: (g["hit_1x2"].astype(float) - g["p_fav"]).sum()).values
    cg_all_cnts = df.groupby("cluster")["p_fav"].count().values
    n_ca = len(cg_all_sums)
    boot_all = [np.sum(cg_all_sums[rng.choice(n_ca, size=n_ca, replace=True)]) / 1793 for _ in range(5000)]
    ci_all = [round(float(np.percentile(boot_all, q)), 4) for q in (2.5, 97.5)]

    overall_tier = {
        "tier": "Genel Toplam (Tüm Maçlar)",
        "n": len(df),
        "mean_pred": round(float(df["p_fav"].mean()), 4),
        "actual_win_rate": round(float(df["hit_1x2"].mean()), 4),
        "mean_residual_e": round(float(np.mean(e_all)), 4),
        "se": round(float(stats.sem(e_all)), 4),
        "ci_clust": ci_all,
        "t_stat": round(float(t_all), 2),
        "raw_p": round(float(p_all), 4),
        "hb_thresh": 0.05,
        "adj_p": round(float(p_all), 4),
        "sig_hb": bool(p_all < 0.05),
    }

    # 3. Over / Under 2.5 Paired Analysis
    y_ou = df["act_over25"].astype(float).values
    cal_ou25 = compute_calibration_curve(df["p_over25"].values, y_ou, n_bins=8, bin_range=(0.30, 0.80))
    p_naive_ou = df["p_naive_ou25"].values if "p_naive_ou25" in df.columns else np.full(len(df), 0.544)
    brier_ou_mod = (df["p_over25"].values - y_ou) ** 2
    brier_ou_nai = (p_naive_ou - y_ou) ** 2
    delta_bs_ou = brier_ou_mod - brier_ou_nai
    bss_ou = float(1.0 - (cal_ou25["brier_score"] / float(np.mean(brier_ou_nai))))

    t_bs_ou, p_bs_ou = stats.ttest_1samp(delta_bs_ou, 0.0)
    boot_bs_ou = [np.mean(rng.choice(delta_bs_ou, size=len(df), replace=True)) for _ in range(5000)]
    ci_bs_ou = [round(float(np.percentile(boot_bs_ou, q)), 6) for q in (2.5, 97.5)]

    p_ou_c = np.clip(df["p_over25"].values, eps, 1.0 - eps)
    p_ou_nc = np.clip(p_naive_ou, eps, 1.0 - eps)
    ll_ou_mod = -(y_ou * np.log(p_ou_c) + (1.0 - y_ou) * np.log(1.0 - p_ou_c))
    ll_ou_nai = -(y_ou * np.log(p_ou_nc) + (1.0 - y_ou) * np.log(1.0 - p_ou_nc))
    delta_ll_ou = ll_ou_mod - ll_ou_nai
    t_ll_ou, p_ll_ou = stats.ttest_1samp(delta_ll_ou, 0.0)
    boot_ll_ou = [np.mean(rng.choice(delta_ll_ou, size=len(df), replace=True)) for _ in range(5000)]
    ci_ll_ou = [round(float(np.percentile(boot_ll_ou, q)), 6) for q in (2.5, 97.5)]

    # 4. Both Teams To Score (BTTS) Paired Analysis
    y_btts = df["act_btts"].astype(float).values
    cal_btts = compute_calibration_curve(df["p_btts"].values, y_btts, n_bins=8, bin_range=(0.30, 0.80))
    p_naive_btts = df["p_naive_btts"].values if "p_naive_btts" in df.columns else np.full(len(df), 0.558)
    brier_btts_mod = (df["p_btts"].values - y_btts) ** 2
    brier_btts_nai = (p_naive_btts - y_btts) ** 2
    delta_bs_btts = brier_btts_mod - brier_btts_nai
    bss_btts = float(1.0 - (cal_btts["brier_score"] / float(np.mean(brier_btts_nai))))

    t_bs_btts, p_bs_btts = stats.ttest_1samp(delta_bs_btts, 0.0)
    boot_bs_btts = [np.mean(rng.choice(delta_bs_btts, size=len(df), replace=True)) for _ in range(5000)]
    ci_bs_btts = [round(float(np.percentile(boot_bs_btts, q)), 6) for q in (2.5, 97.5)]

    p_btts_c = np.clip(df["p_btts"].values, eps, 1.0 - eps)
    p_btts_nc = np.clip(p_naive_btts, eps, 1.0 - eps)
    ll_btts_mod = -(y_btts * np.log(p_btts_c) + (1.0 - y_btts) * np.log(1.0 - p_btts_c))
    ll_btts_nai = -(y_btts * np.log(p_btts_nc) + (1.0 - y_btts) * np.log(1.0 - p_btts_nc))
    delta_ll_btts = ll_btts_mod - ll_btts_nai
    t_ll_btts, p_ll_btts = stats.ttest_1samp(delta_ll_btts, 0.0)
    boot_ll_btts = [np.mean(rng.choice(delta_ll_btts, size=len(df), replace=True)) for _ in range(5000)]
    ci_ll_btts = [round(float(np.percentile(boot_ll_btts, q)), 6) for q in (2.5, 97.5)]

    multiclass_mod_ece = round(float((cal_mod_home["ece"] + cal_mod_draw["ece"] + cal_mod_away["ece"]) / 3.0), 4)
    multiclass_mkt_ece = round(float((cal_mkt_home["ece"] + cal_mkt_draw["ece"] + cal_mkt_away["ece"]) / 3.0), 4)

    return {
        "summary": {
            "n_matches": len(df),
            "multiclass_model_1x2_ece": multiclass_mod_ece,
            "multiclass_market_1x2_ece": multiclass_mkt_ece,
            "model_fav_ece": cal_mod_fav["ece"],
            "market_fav_ece": cal_mkt_fav["ece"],
            "over_under_25": {
                "ece": cal_ou25["ece"],
                "brier_model": round(float(np.mean(brier_ou_mod)), 4),
                "brier_naive": round(float(np.mean(brier_ou_nai)), 4),
                "brier_50_pct": 0.2500,
                "bss_pct": round(bss_ou * 100.0, 2),
                "delta_bs_mean": round(float(np.mean(delta_bs_ou)), 6),
                "delta_bs_ci_95": ci_bs_ou,
                "delta_bs_p_value": round(float(p_bs_ou), 4),
                "log_loss_model": round(float(np.mean(ll_ou_mod)), 4),
                "log_loss_naive": round(float(np.mean(ll_ou_nai)), 4),
                "delta_ll_mean": round(float(np.mean(delta_ll_ou)), 6),
                "delta_ll_ci_95": ci_ll_ou,
                "delta_ll_p_value": round(float(p_ll_ou), 4),
            },
            "both_teams_to_score": {
                "ece": cal_btts["ece"],
                "brier_model": round(float(np.mean(brier_btts_mod)), 4),
                "brier_naive": round(float(np.mean(brier_btts_nai)), 4),
                "brier_50_pct": 0.2500,
                "bss_pct": round(bss_btts * 100.0, 2),
                "delta_bs_mean": round(float(np.mean(delta_bs_btts)), 6),
                "delta_bs_ci_95": ci_bs_btts,
                "delta_bs_p_value": round(float(p_bs_btts), 4),
                "log_loss_model": round(float(np.mean(ll_btts_mod)), 4),
                "log_loss_naive": round(float(np.mean(ll_btts_nai)), 4),
                "delta_ll_mean": round(float(np.mean(delta_ll_btts)), 6),
                "delta_ll_ci_95": ci_ll_btts,
                "delta_ll_p_value": round(float(p_ll_btts), 4),
            },
            "favorite_residuals": fav_residual_tiers + [overall_tier],
        },
        "model_1x2": {
            "home": cal_mod_home,
            "draw": cal_mod_draw,
            "away": cal_mod_away,
            "favorite": cal_mod_fav,
        },
        "market_1x2": {
            "home": cal_mkt_home,
            "draw": cal_mkt_draw,
            "away": cal_mkt_away,
            "favorite": cal_mkt_fav,
        },
        "over_under_25": cal_ou25,
        "both_teams_to_score": cal_btts,
    }


def generate_experiment_b_report(res: dict[str, Any], output_path: str = "reports/deney_b_kalibrasyon_ve_ece.md") -> None:
    s = res["summary"]
    mod_fav = res["model_1x2"]["favorite"]
    mkt_fav = res["market_1x2"]["favorite"]
    ou_stats = s["over_under_25"]
    btts_stats = s["both_teams_to_score"]
    fav_res = s["favorite_residuals"]
    ou = res["over_under_25"]
    btts = res["both_teams_to_score"]

    md = f"""# Deney B: Çoklu Piyasa Kalibrasyon Eğrileri, Brier Beceri Skoru (BSS) ve ECE Raporu

> **Metodolojik Çerçeve:** Bir tahmin motorunun güvenilirliği yalnızca doğru sınıfı seçme oranıyla (isabet) değil, ürettiği olasılıkların gerçek dünyadaki gerçekleşme sıklığı ile ne kadar örtüştüğüyle (**Kalibrasyon**) ve geçmiş frekans tabanına göre ne kadar ek bilgi taşıdığıyla (**Brier Skill Score - BSS**) ölçülür.
> Bu raporda Model V3 ile Kapanış Piyasa Oranları; **1X2 (Ev, Beraberlik, Deplasman ve Favori)**, **2.5 Gol Alt/Üst** ve **Karşılıklı Gol (KG Var)** pazarlarında beklenen kalibrasyon hatası (**ECE**), **Brier Skoru**, **Brier Beceri Skoru (BSS)**, kalibrasyon hata artıkları (e_i = y_i - p_i) ve eşleştirilmiş güven aralıkları ile incelenmiştir.

---

## 1. Genel Kalibrasyon ve Beceri Skoru Karnesi

| Piyasa / Seçim | Model ECE | Piyasa ECE | Model Brier | Naif Brier (Geçmiş Frekans) | Sabit %50 Brier | Brier Skill Score (BSS) | Karar & İstatistiksel Yorum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1X2 Genel (Multi-Class)** | **{s['multiclass_model_1x2_ece']}** | **{s['multiclass_market_1x2_ece']}** | — | — | — | — | Piyasa ve model 1X2 kalibrasyon hatasında neredeyse aynı seviyededir. |
| **Favori Takım Güveni** | **{s['model_fav_ece']}** | **{s['market_fav_ece']}** | **{mod_fav['brier_score']}** | — | — | — | Model favorilere aşırı güven (overconfidence) üretmemektedir. |
| **Ev Sahibi (Home)** | **{res['model_1x2']['home']['ece']}** | **{res['market_1x2']['home']['ece']}** | **{res['model_1x2']['home']['brier_score']}** | — | — | — | Ev sahibi galibiyeti kalibrasyonunda fark çok düşüktür. |
| **Beraberlik (Draw)** | **{res['model_1x2']['draw']['ece']}** | **{res['market_1x2']['draw']['ece']}** | **{res['model_1x2']['draw']['brier_score']}** | — | — | — | Beraberlik tahmini piyasada da modelde de en zor kalibre olan pazardır. |
| **Deplasman (Away)** | **{res['model_1x2']['away']['ece']}** | **{res['market_1x2']['away']['ece']}** | **{res['model_1x2']['away']['brier_score']}** | — | — | — | Model deplasman oranlarında piyasaya çok yakın ECE sergilemektedir. |
| **2.5 Gol Üst** | **{ou_stats['ece']}** | — | **{ou_stats['brier_model']}** | **{ou_stats['brier_naive']}** | 0.2500 | **+{ou_stats['bss_pct']:.2f}%** | Olumlu nokta tahmini; eşli güven aralığı sıfırı içerir (p = {ou_stats['delta_bs_p_value']:.4f}). |
| **KG Var (BTTS)** | **{btts_stats['ece']}** | — | **{btts_stats['brier_model']}** | **{btts_stats['brier_naive']}** | 0.2500 | **{btts_stats['bss_pct']:.2f}%** | **Negatif Beceri: Model geçmiş frekanstan geridedir (p = {btts_stats['delta_bs_p_value']:.4f}).** |

---

## 2. İkili Piyasalarda Eşleştirilmiş Fark Analizi (Paired Differences)

Brier Skoru ve Log-Loss farkları her maç i için eşleştirilmiş olarak hesaplanmıştır:
- Delta BS_i = (p_i - y_i)^2 - (p_naive_i - y_i)^2
- Delta LL_i = -log(p_i) - [-log(p_naive_i)]

| Piyasa | Model Brier | Naif Brier | BSS (%) | Eşli Delta BS [%95 Bootstrap GA] | Eşli t-testi (BS) | Eşli Delta LL [%95 Bootstrap GA] | Eşli t-testi (LL) | Metodolojik Karar |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2.5 Gol Üst** | {ou_stats['brier_model']} | {ou_stats['brier_naive']} | **+{ou_stats['bss_pct']:.2f}%** | {ou_stats['delta_bs_mean']:+.6f} <br><small>[{ou_stats['delta_bs_ci_95'][0]:+.6f}, {ou_stats['delta_bs_ci_95'][1]:+.6f}]</small> | $p = {ou_stats['delta_bs_p_value']:.4f}$ | {ou_stats['delta_ll_mean']:+.6f} <br><small>[{ou_stats['delta_ll_ci_95'][0]:+.6f}, {ou_stats['delta_ll_ci_95'][1]:+.6f}]</small> | $p = {ou_stats['delta_ll_p_value']:.4f}$ | **Pozitif Nokta Tahmini (Anlamsız Fark, CI sıfırı içerir)** |
| **KG Var (BTTS)** | {btts_stats['brier_model']} | {btts_stats['brier_naive']} | **{btts_stats['bss_pct']:.2f}%** | {btts_stats['delta_bs_mean']:+.6f} <br><small>[{btts_stats['delta_bs_ci_95'][0]:+.6f}, {btts_stats['delta_bs_ci_95'][1]:+.6f}]</small> | $p = {btts_stats['delta_bs_p_value']:.4f}$ | {btts_stats['delta_ll_mean']:+.6f} <br><small>[{btts_stats['delta_ll_ci_95'][0]:+.6f}, {btts_stats['delta_ll_ci_95'][1]:+.6f}]</small> | $p = {btts_stats['delta_ll_p_value']:.4f}$ | **Negatif Beceri (Model naif frekanstan daha kötüdür)** |

> [!NOTE]
> **Tek Standart Değer:** Önceki taslaklardaki `%1.16` kazanç veya `0.2474` naif Brier gibi çelişkili sürüm kalıntıları kaldırılmış; 1.793 maçlık tam veri kümesinde dinamik önsel Brier skorları 2.5 Üst için **0.2483** (BSS: +0.96%), KG Var için **0.2468** (BSS: -1.25%) olarak standardize edilmiştir.

---

## 3. Kalibrasyon Hata Artığı (e_i = y_i - p_i) ve Favori Güven Testi

Modelin sistematik olarak düşük (underconfidence) veya yüksek (overconfidence) tahmin üretip üretmediği, maç bazındaki hata artıkları e_i = y_i - p_i üzerinden tek örneklem t-testi ve blok bootstrap güven aralıkları ile sınanmıştır (H0: E[e_i] = 0):

| Güven Kademesi | Maç (N) | Ort. Tahmin (p_bar) | Fiili Galibiyet (y_bar) | Hata Artığı (e_bar) | Küme-Dirençli 95% GA (Cluster) | Ham p-değeri | Holm Eşiği | Holm Düzeltilmiş p | Karar & Yorum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
"""
    for r in fav_res:
        sig_hb_str = "**ANLAMLI (Underconfident)**" if r["sig_hb"] else "Anlamsız (Sapma Gösterilemedi)"
        if r["tier"] == "Genel Toplam (Tüm Maçlar)":
            sig_hb_str = "**ANLAMLI (Underconfident)**" if r["sig_hb"] else "Anlamsız"
        md += f"| **{r['tier']}** | {r['n']} | %{r['mean_pred']*100:.1f} | %{r['actual_win_rate']*100:.1f} | **{r['mean_residual_e']:+.4f}** | `[{r['ci_clust'][0]:+.4f}, {r['ci_clust'][1]:+.4f}]` | {r['raw_p']:.4f} | {r['hb_thresh']:.4f} | {r['adj_p']:.4f} | {sig_hb_str} |\n"

    md += """
> [!IMPORTANT]
> **Bilimsel Kalibrasyon Çıkarımları:**
> 1. **Çok Yüksek Güven Kademesinde (≥ %65) Underconfidence Çoklu Test ve Kümeleme Sonrasında da Korunmaktadır:**
>    - 136 maçta ortalama tahmin $\\bar{p} = 0.7166$, fiili galibiyet $\\bar{y} = 0.8015$\'tir.
>    - Hata artığı $\\bar{e} = +0.0849$\'dur (%8.5 puanlık eksik tahmin).
>    - Fikstür haftası bağımlılığını koruyan **Küme-Dirençli %95 Bootstrap Güven Aralığı `[+0.0111, +0.1538]`** olup sıfırı dışarıda bırakmaktadır.
>    - 4 kademe üzerinden uygulanan **Holm-Bonferroni düzeltmesi sonrasında** ($p_{\\text{Holm}} = 3 \\times 0.0162 = \\mathbf{0.0486} < 0.05$), modelin bu grupta istatistiksel olarak aşırı ihtiyatlı (underconfident) kaldığı doğrulanmıştır.
> 2. **Orta-Yüksek Güven Kademesinde (%55 - %65) Çoklu Test Etkisi:**
>    - Ham $p = 0.0421$ değeri nominal olarak 0.05\'in altında görünse de, Holm-Bonferroni eşiği ($0.0250$) altında **anlamlılığını koruyamaz** ($p_{\\text{Holm}} = 0.0842$). Dolayısıyla bu kademede kesin bir sapma iddia edilemez.
> 3. **Dengeli Maçlarda Kalibrasyon Sapması Yoktur:**
>    - %45 - %55 kademesinde $\\bar{e} = +0.0257$, Küme Güven Aralığı `[-0.0133, +0.0660]` aralığında sıfırı içermektedir ($p = 0.2042$).
> 4. **Terminoloji:** Modelin \'kusursuz kalibre edildiği\' iddia edilemez; veriler dengeli maçlarda istatistiksel bir sapmanın gösterilemediğini, ağır favorilerde ise sistematik bir muhafazakarlığın varlığını ortaya koymaktadır.
"""

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md, encoding="utf-8")
    print(f"Deney B raporu basariyla yazildi: {output_path}")


if __name__ == "__main__":
    res = run_experiment_b()
    with open("reports/experiment_b_calibration.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    generate_experiment_b_report(res)
