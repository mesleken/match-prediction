# Deney G: Şut Kalitesi Vekil xG/xGA Form Özellikleri Walk-Forward Kanıt Kapısı

**Tarih:** 26 Ekim 2026  
**Değerlendirilen Maç Sayısı:** 1,793 (194 maç günü kümesi, 2021/22 - 2026/27, Walk-Forward, Sızıntısız)  
**Asgari Saptanabilir Etki (MDE, Güç: %80, \alpha: 0.05):** \pm 0.000559 RPS  
**Kalibrasyon:** Sıcaklık Ölçeklemesi Devrede ($T = 0.75$)  
**Amaç:** Gerçek rakip gücü düzeltmeli vekil xG/xGA form momentumunun, Model B'ye (Dixon-Coles) ek tahmin değeri sağlayıp sağlamadığını önceden belirlenmiş karar kuralıyla test etmek.

---

## 1. Sayısal Sonuçlar ve Metrik Karşılaştırması

| Model / Varyant | 1X2 RPS | RPS Farkı (\Delta) | Kümeli %95 Güven Aralığı | Ham $p$ | Holm $p$ | 1X2 İsabet | 2.5 Üst Brier | KG Var Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model B0 (Kalibre Taban)** | **0.199752** | — | — | — | — | %53.60 | 0.245542 | 0.248893 |
| **Model B+G1 ($w = 0.04$)** | 0.199470 | -0.000282 | [-0.000719, +0.000118] | $p = 0.1576$ | $p = 0.3152$ | %53.49 | 0.246110 | 0.248952 |
| **Model B+G2 ($w = 0.08$)** | 0.199409 | -0.000344 | [-0.001220, +0.000458] | $p = 0.3883$ | $p = 0.3883$ | %53.49 | 0.246835 | 0.249091 |
| **Model A (Piyasa Referansı)** | **0.190786** | — | — | — | — | %55.83 | — | — |

---

## 2. Önceden Belirlenmiş Karar Kuralı Sonucu

> [!IMPORTANT]
> **Karar: G-KAPISI GEÇİLMEDİ (Vekil xG Formu Anlamlı Kazanç Sağlamadı - Katsayı Kapalı Tutulmalı)**
> 
> - **G1 ($w=0.04$):** $\Delta \text{RPS} = -0.000282$, Kümeli %95 GA: `[-0.000719, +0.000118]`, $p_{\text{Holm}} = 0.3152$
> - **G2 ($w=0.08$):** $\Delta \text{RPS} = -0.000344$, Kümeli %95 GA: `[-0.001220, +0.000458]`, $p_{\text{Holm}} = 0.3883$
> 
> **Metodolojik Değerlendirme:**
> Model B, şut kalitesi ve gol oranlarını Poisson MAP fit aşamasında `shot_alpha=0.40` ile doğrudan optimize etmektedir. Üzerine eklenen doğrusal form kayması, maç günü kümeli güven aralığının üst sınırında sıfırı dışlayacak düzeyde ek bir kestirimsel güç üretmemiştir. Bu nedenle aşırı öğrenmeyi önlemek adına `xg_form_weight = 0.0` (kapalı) tutulmaya devam edilmelidir.
