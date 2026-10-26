# Deney E: MatchEngine vs Dixon-Coles Walk-Forward Kanıt Kapısı (Faz 3)

**Tarih:** 26 Ekim 2026  
**Değerlendirilen Maç Sayısı:** 497 (Tüm walk-forward sezonlarından tabakalı örneklem)  
**Tekrar Sayısı:** Maç başına 150 olay tabanlı simülasyon  
**Amaç:** Olay tabanlı dinamik simülasyon motorunun (MatchEngine), analitik Model B'ye (Dixon-Coles) kıyasla maç sonucu tahmin doğruluğuna (1X2 RPS, Brier, NLL, Alt/Üst, KG) ek bir tahmin avantajı sağlayıp sağlamadığını önceden belirlenmiş karar kuralıyla test etmek.

---

## 1. Sayısal Sonuçlar ve Metrik Karşılaştırması

| Metrik | Model B (Analitik Dixon-Coles) | Model B + MatchEngine (Monte Carlo) | Fark (\Delta) | Eşli $t$-testi ($p$) | Wilcoxon ($p$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1X2 RPS** | **0.203336** | **0.214799** | **+0.011464** | $t = 4.127, p = 0.0000$ | $p = 0.0000$ |
| **1X2 Brier Skoru** | 0.597780 | 0.625003 | +0.027223 | $t = 4.321, p = 0.0000$ | — |
| **Log-Loss (NLL)** | 1.002357 | 1.040681 | +0.038324 | — | — |
| **Alt / Üst 2.5 Brier** | 0.244209 | 0.256828 | +0.012619 | — | — |
| **KG Var Brier** | 0.247459 | 0.251070 | +0.003612 | — | — |

- **%95 Bootstrap Güven Aralığı ($\Delta \text{RPS}$):** `[+0.006232, +0.017001]`

---

## 2. Önceden Belirlenmiş Karar Kuralı ve Mimari Sonuç

> [!IMPORTANT]
> **Karar Kuralı Sonucu: SENARYO VE ANLATIM ARACI (Tahmin Üstünlüğü Yok, Taktik/Olay Dağılımı İçin Kullanılmalı)**
> 
> 1. **Tahmin Doğruluğu:** Monte Carlo simülasyonu, sonlu $N$ örneklem varyansı (simülasyon gürültüsü) nedeniyle analitik Dixon-Coles formülüne göre RPS'te bir üstünlük sağlamamaktadır (Fark: `+0.011464`, %95 GA `[+0.006232, +0.017001]` sıfırı tamamen dışarıda bırakarak pozitif taraftadır; yani simülasyon analitik formüle kıyasla istatistiksel olarak anlamlı biçimde daha gürültülü ve yüksek hata üretmektedir).
> 2. **Rol Ayrımı (Mimari Netlik):**
>    - **Olasılık ve Bahis Tahmin Motoru:** Saf Dixon-Coles (Model B) ve Kapanış Piyasası (Model A) birincil 1X2, Alt/Üst ve KG olasılık kaynağı olarak kalmalıdır.
>    - **MatchEngine Simülasyon Motoru:** Tekil olasılık hesaplamak yerine çoklu senaryo üretimi, gol dakikaları dağılımı, korner/kart beklentileri, maç içi geri dönüş (comeback) olasılığı ve kronolojik olay akışı (timeline) sunan bir **senaryo ve simülasyon katmanı** olarak konumlandırılmalıdır.
