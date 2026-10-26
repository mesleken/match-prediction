# Deney G: Şut Kalitesi Vekil xG/xGA Form Özellikleri Walk-Forward Kanıt Kapısı

**Tarih:** 26 Ekim 2026  
**Değerlendirilen Maç Sayısı:** 1,793 (2021/22 - 2026/27, Walk-Forward, Sızıntısız)  
**Kalibrasyon:** Sıcaklık Ölçeklemesi Devrede ($T = 0.75$)  
**Amaç:** Şut kalitesi tabanlı vekil xG/xGA form momentumunun, Model B'ye (Dixon-Coles) ek tahmin değeri sağlayıp sağlamadığını önceden belirlenmiş karar kuralıyla test etmek.

---

## 1. Sayısal Sonuçlar ve Metrik Karşılaştırması

| Model / Varyant | 1X2 RPS | RPS Farkı (\Delta) | %95 Güven Aralığı | Eşli $t$-testi ($p$) | Wilcoxon ($p$) | 1X2 İsabet |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model B0 (Kalibre Taban)** | **0.200772** | — | — | — | — | %53.54 |
| **Model B+G1 ($w = 0.04$)** | 0.200381 | -0.000391 | [-0.000748, +0.000017] | $p = 0.0531$ | $p = 0.0400$ | %53.49 |
| **Model B+G2 ($w = 0.08$)** | 0.200212 | -0.000560 | [-0.001374, +0.000230] | $p = 0.1650$ | $p = 0.1023$ | %52.98 |
| **Model A (Piyasa Referansı)** | **0.190786** | — | — | — | — | %55.83 |

---

## 2. Önceden Belirlenmiş Karar Kuralı Sonucu

> [!IMPORTANT]
> **Karar: G-KAPISI GEÇİLMEDİ (Vekil xG Formu Anlamlı Kazanç Sağlamadı - Katsayı Kapalı Tutulmalı)**
> 
> - **G1 ($w=0.04$):** $\Delta \text{RPS} = -0.000391$, %95 GA: `[-0.000748, +0.000017]`
> - **G2 ($w=0.08$):** $\Delta \text{RPS} = -0.000560$, %95 GA: `[-0.001374, +0.000230]`
> 
> **Metodolojik Yorum:**
> Model B zaten takım gücü, zaman azalışı ve şut kalitesi kompozit olabilirlik katsayısını (`shot_alpha=0.40`) Poisson MAP fit aşamasında doğrudan optimizasyona dahil etmektedir. İkinci aşamada eklenen üstel azalışlı vekil xG form katsayısı, mevcut kompozit bilgiye kıyasla ek bir ayrıştırıcı varyans sağlamamakta (veya hafif gürültü eklemekte) olduğundan, katı hakem kuralımız gereğince üretim hattında `xg_form_weight = 0.0` (kapalı) tutulmaya devam edilmelidir.
