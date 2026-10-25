# 🧪 2026/27 Süper Lig Simülasyon Denetimi ve Model Geliştirme Raporu (V2.2)
**Denetim Tarihi:** 2026-10-25 18:03 UTC | **İncelenen Maç Sayısı:** 72 Maç (1–8. Hafta Tamamı)
**Veri Sızıntısı:** %0 (Strictly Out-of-Sample, her maç yalnızca maç saatinden önceki verilerle simüle edilmiştir)

## 1. Yönetici Özeti ve Model Karşılaştırma Karnesi

| Değerlendirme Metriği | Baz Model (V2.0) | Geliştirilmiş Model (V2.2) | Değişim / İyileşme |
| :--- | :---: | :---: | :---: |
| **RPS (Ranked Probability Score)** | 0.2211 | **0.2070** | 📉 -6.35% |
| **Log-Loss (Exact Score NLL)** | 3.0743 | **2.9795** | 📉 -3.08% |
| **1X2 İsabet Oranı (Hit Rate)** | %47.2 | **%52.8** | +5.6% |
| **Çifte Şans İsabet Oranı** | %73.6 | **%83.3** | +9.7% |
| **Alt/Üst 2.5 İsabet Oranı** | %62.5 | **%70.8** | +8.3% |
| **İlk 1 (Tam) Skor Yakalama** | %6.9 | **%8.3** | +1.4% |
| **İlk 3 Skor Yakalama Oranı** | %15.3 | **%26.4** | +11.1% |
| **İlk 5 Skor Yakalama Oranı** | %36.1 | **%43.1** | +6.9% |
| **xG Farkı Hatası (xG Diff MAE)** | 0.882 | **0.868** | -0.014 |
| **Korner Beklentisi MAE** | 2.54 | **2.50** | -0.04 |
| **Kart Beklentisi MAE** | 2.00 | **1.94** | -0.07 |

## 2. Hata Teşhisi: Model Hatası vs. Bitiricilik / Şans Ayrıştırması (Variance Decomposition)

Gerçekleşen gol farkı ile model beklentisi arasındaki sapmanın matematiksel varyans ayrıştırması:

- **Model Kurgu Hatası (Model Setup MSE — $(\lambda_H - \mu_A) - (xG_H - xG_A)$):** `1.195` (**%33.3**)
- **Bitiricilik & Şans Varyansı (Finishing Luck MSE — $(xG_H - xG_A) - (G_H - G_A)$):** `2.394` (**%66.7**)
- **Toplam Skor Farkı MSE:** `3.637`
> [!NOTE]
> Yanılgıların yaklaşık **%67'i**, modelin saha içi pozisyon kalitesini ($xG$) doğru tahmin etmesine rağmen oyuncuların olağan dışı şut bitiriciliği veya kaleci kurtarışları nedeniyle gerçekleşen şans faktöründen kaynaklanmaktadır.

### Teşhis Dağılımı (Diagnosis Breakdown)

- **✅ Baskın İsabet (Dominant Hit):** 25 maç (%34.7)
- **⚠️ Model / Taktiksel Sapma (Model Miss):** 17 maç (%23.6)
- **🎯 Tam İsabet (Exact Hit):** 13 maç (%18.1)
- **🍀 Şans / Bitiricilik Sapması (Finishing Luck):** 11 maç (%15.3)
- **⚖️ Kritik Denge / Beraberlik (Draw Variance):** 6 maç (%8.3)

## 3. Hafta Hafta Performans İlerlemesi (Round-by-Round Learning)

| Hafta | Maç Sayısı | RPS | 1X2 İsabet % | Çifte Şans % | İlk 3 Skor % | Korner MAE | Kart MAE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Hafta | 9 | 0.2203 | %44.4 | %77.8 | %44.4 | 2.09 | 1.60 |
| 2. Hafta | 9 | 0.2891 | %44.4 | %66.7 | %22.2 | 2.74 | 1.96 |
| 3. Hafta | 9 | 0.2290 | %44.4 | %88.9 | %33.3 | 2.76 | 1.83 |
| 4. Hafta | 9 | 0.2416 | %44.4 | %88.9 | %22.2 | 2.06 | 2.05 |
| 5. Hafta | 9 | 0.2035 | %55.6 | %77.8 | %22.2 | 3.27 | 1.33 |
| 6. Hafta | 9 | 0.2228 | %33.3 | %77.8 | %0.0 | 2.44 | 2.66 |
| 7. Hafta | 9 | 0.1328 | %66.7 | %88.9 | %11.1 | 1.96 | 2.42 |
| 8. Hafta | 9 | 0.1172 | %88.9 | %100.0 | %55.6 | 2.70 | 1.64 |

## 4. Takım Bazlı Sistematik Yanlılık Analizi (Team Bias Table)

Modelin hangi takımları daha kolay çözdüğü ve hangi takımlarda sistematik sapma gösterdiği tablosu:

| Takım | Maç | Gerçek Gol (A-Y) | Gerçek xG (A-Y) | Model Bekl. (A-Y) | Hücum Sapması (G) | Savunma Sapması (G) | 1X2 Doğruluk % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Eyupspor** | 8 | 2-22 | 6.7-17.4 | 6.5-15.0 | +0.56 | -0.88 | %87.5 |
| **Galatasaray** | 8 | 17-11 | 18.5-9.1 | 19.3-7.6 | +0.29 | -0.43 | %75.0 |
| **Basaksehir** | 8 | 12-12 | 12.8-12.9 | 13.0-10.9 | +0.12 | -0.13 | %75.0 |
| **Gaziantep FK** | 8 | 8-12 | 8.2-12.8 | 9.2-12.3 | +0.15 | +0.04 | %75.0 |
| **Fenerbahce** | 8 | 22-6 | 18.7-9.2 | 19.6-7.3 | -0.30 | +0.16 | %62.5 |
| **Besiktas** | 8 | 16-10 | 14.4-7.8 | 15.7-8.6 | -0.03 | -0.17 | %62.5 |
| **Kocaelispor** | 8 | 9-8 | 9.3-11.3 | 7.6-11.7 | -0.18 | +0.46 | %62.5 |
| **Trabzonspor** | 8 | 18-5 | 14.7-11.8 | 15.0-9.5 | -0.37 | +0.56 | %50.0 |
| **Rizespor** | 8 | 8-11 | 10.8-9.7 | 10.8-11.7 | +0.36 | +0.09 | %50.0 |
| **Samsunspor** | 8 | 7-19 | 9.6-13.8 | 9.8-13.2 | +0.34 | -0.73 | %50.0 |
| **Genclerbirligi** | 8 | 7-16 | 8.2-17.1 | 8.8-15.5 | +0.22 | -0.06 | %50.0 |
| **Erzurumspor FK** | 8 | 5-11 | 6.8-10.8 | 6.9-13.2 | +0.24 | +0.28 | %50.0 |
| **Corum FK** | 8 | 18-13 | 11.1-11.0 | 12.8-11.6 | -0.65 | -0.17 | %50.0 |
| **Kasimpasa** | 8 | 12-9 | 9.6-9.6 | 8.8-11.7 | -0.40 | +0.34 | %37.5 |
| **Amedspor** | 8 | 18-9 | 15.2-6.6 | 12.8-9.5 | -0.66 | +0.07 | %37.5 |
| **Konyaspor** | 8 | 4-9 | 7.9-9.7 | 8.8-13.0 | +0.60 | +0.51 | %25.0 |
| **Alanyaspor** | 8 | 8-9 | 9.3-13.2 | 9.2-11.3 | +0.14 | +0.29 | %25.0 |
| **Goztepe** | 8 | 17-16 | 16.1-14.0 | 11.9-12.7 | -0.64 | -0.41 | %25.0 |

## 5. Yüksek Güvenli Favori Sürprizleri Analizi (Upset Post-Mortem)

Modelin favoriye %52'den fazla kazanma şansı verdiği ancak favorinin puan kaybettiği maçlar:

| Hafta | Maç | Skor | Model Favorisi | Kazanma İhtimali | Gerçek xG Farkı | Sürprizin Temel Nedeni |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1. H | Genclerbirligi - Fenerbahce | **2-1** | Fenerbahce | %74.8 | -1.71 | Bitiricilik Şanssızlığı (Favori xG'yi domine etti ama puan kaybetti) |
| 1. H | Galatasaray - Corum FK | **2-2** | Galatasaray | %71.2 | +2.29 | Bitiricilik Şanssızlığı (Favori xG'yi domine etti ama puan kaybetti) |
| 3. H | Genclerbirligi - Erzurumspor FK | **1-1** | Genclerbirligi | %67.4 | -0.10 | Dengeli Maç / Yüksek Varyans |
| 7. H | Genclerbirligi - Amedspor | **2-2** | Amedspor | %67.4 | -1.66 | Bitiricilik Şanssızlığı (Favori xG'yi domine etti ama puan kaybetti) |
| 5. H | Gaziantep FK - Fenerbahce | **0-0** | Fenerbahce | %65.1 | -0.84 | Bitiricilik Şanssızlığı (Favori xG'yi domine etti ama puan kaybetti) |
| 5. H | Konyaspor - Trabzonspor | **1-0** | Trabzonspor | %62.5 | +0.13 | Dengeli Maç / Yüksek Varyans |
| 3. H | Basaksehir - Kasimpasa | **1-1** | Basaksehir | %60.0 | -0.81 | Taktiksel Yanılma (Rakip sahada xG üstünlüğünü kurdu) |
| 6. H | Corum FK - Alanyaspor | **1-2** | Corum FK | %59.2 | -0.02 | Dengeli Maç / Yüksek Varyans |
| 7. H | Alanyaspor - Erzurumspor FK | **0-0** | Alanyaspor | %59.2 | -0.71 | Taktiksel Yanılma (Rakip sahada xG üstünlüğünü kurdu) |
| 1. H | Kasimpasa - Trabzonspor | **1-1** | Trabzonspor | %58.6 | +1.00 | Taktiksel Yanılma (Rakip sahada xG üstünlüğünü kurdu) |
| 2. H | Corum FK - Kasimpasa | **0-1** | Corum FK | %58.5 | +0.79 | Bitiricilik Şanssızlığı (Favori xG'yi domine etti ama puan kaybetti) |
| 2. H | Alanyaspor - Besiktas | **1-0** | Besiktas | %55.5 | -0.20 | Dengeli Maç / Yüksek Varyans |
| 2. H | Goztepe - Genclerbirligi | **0-1** | Goztepe | %53.8 | +0.36 | Dengeli Maç / Yüksek Varyans |
| 2. H | Rizespor - Samsunspor | **0-2** | Rizespor | %53.4 | +1.53 | Bitiricilik Şanssızlığı (Favori xG'yi domine etti ama puan kaybetti) |
| 4. H | Goztepe - Gaziantep FK | **2-4** | Goztepe | %53.4 | +1.82 | Bitiricilik Şanssızlığı (Favori xG'yi domine etti ama puan kaybetti) |
| 3. H | Amedspor - Trabzonspor | **2-1** | Trabzonspor | %53.2 | +1.55 | Taktiksel Yanılma (Rakip sahada xG üstünlüğünü kurdu) |

## 6. Hakem Etkisi, Kart Puanı ve Oyun Temposu Karnesi (Referee Officiating Impact)

> [!NOTE]
> **Hakem Veri Dürüstlüğü:** 72 maçın **38** adedi Mackolik resmi maç detaylarından doğrulanmış gerçek hakemlerdir. Hakemi henüz sisteme işlenmemiş veya bilinmeyen **34** maçta modele sıfır yanlılık (lig ortalaması) atanmıştır; sahte rotasyon fallback'i tamamen kaldırılmıştır.

| Hakem Adı | Maç | Gerçekleşen Kart Pts | Model Kart Pts | Kart Pts MAE | Gerçek Kart | Model Kart | Gerçek xG | Model xG |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bilinmiyor (Lig Ortalaması)** | 34 | 47.2 pts | 44.3 pts | 21.5 pts | 4.7 | 4.1 | 2.98 | 2.96 |
| **Batuhan Kolak** | 3 | 31.7 pts | 42.8 pts | 11.1 pts | 3.0 | 3.9 | 2.91 | 2.84 |
| **Gürcan Hasova** | 3 | 38.3 pts | 41.1 pts | 17.7 pts | 3.7 | 3.8 | 2.38 | 2.70 |
| **Yasin Kol** | 3 | 40.0 pts | 42.6 pts | 17.8 pts | 4.0 | 4.0 | 2.32 | 2.82 |
| **Halil Umut Meler** | 3 | 40.0 pts | 44.6 pts | 21.1 pts | 4.0 | 4.2 | 2.15 | 2.81 |
| **Çağdaş Altay** | 3 | 33.3 pts | 37.3 pts | 13.8 pts | 3.3 | 3.5 | 3.69 | 2.92 |
| **Ümit Öztürk** | 3 | 51.7 pts | 41.8 pts | 24.8 pts | 5.0 | 3.9 | 2.60 | 2.43 |
| **Cihan Aydın** | 2 | 57.5 pts | 47.6 pts | 9.9 pts | 5.5 | 4.3 | 2.13 | 2.29 |
| **Oğuzhan Aksu** | 2 | 32.5 pts | 45.8 pts | 22.5 pts | 3.0 | 4.2 | 3.78 | 2.74 |
| **Mehmet Türkmen** | 2 | 40.0 pts | 42.8 pts | 2.8 pts | 4.0 | 4.0 | 2.97 | 3.33 |
| **Oğuzhan Çakır** | 2 | 72.5 pts | 49.2 pts | 23.3 pts | 7.0 | 4.5 | 1.76 | 2.91 |
| **Kadir Sağlam** | 2 | 57.5 pts | 45.7 pts | 19.3 pts | 5.5 | 4.2 | 4.37 | 3.14 |
| **Atilla Karaoğlan** | 2 | 55.0 pts | 42.8 pts | 28.1 pts | 5.5 | 4.0 | 3.87 | 3.22 |
| **Reşat O. Coşkunses** | 2 | 22.5 pts | 45.7 pts | 23.2 pts | 2.0 | 4.2 | 2.49 | 2.33 |
| **Fatih Tokail** | 1 | 40.0 pts | 44.1 pts | 4.1 pts | 4.0 | 4.1 | 1.41 | 2.80 |
| **Ömer F. Turtay** | 1 | 20.0 pts | 45.6 pts | 25.6 pts | 2.0 | 4.2 | 2.17 | 3.11 |
| **Ali Şansalan** | 1 | 70.0 pts | 43.1 pts | 26.9 pts | 7.0 | 4.0 | 4.22 | 2.96 |
| **Adnan D. Kayatepe** | 1 | 0.0 pts | 45.3 pts | 45.3 pts | 0.0 | 4.2 | 1.28 | 2.42 |
| **Davut D. Çelik** | 1 | 30.0 pts | 43.6 pts | 13.6 pts | 3.0 | 4.0 | 3.30 | 2.75 |
| **Abdullah B. Taşkınsoy** | 1 | 20.0 pts | 44.6 pts | 24.6 pts | 2.0 | 4.1 | 3.35 | 2.33 |

## 7. Oynanmış 72 Maçın Hafta Hafta Detaylı Simülasyon Tablosu (1–8. Hafta)

### 1. Hafta Karşılaşmaları

| Maç | Gerçek Skor | Gerçek xG | Tahmin (λ - μ) | Model 1X2 % | İlk 3 Skor Tahmini | Korner (T/G) | Kart (T/G) | Teşhis |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Galatasaray - Corum FK** | **2-2** | 2.90-0.61 | 2.36 - 0.87 | %71 / %17 / %12 | 2-1 (%10), 2-0 (%11), 1-0 (%10) | 9.8 / 9 | 3.8 / 3 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Kasimpasa - Trabzonspor** | **1-1** | 1.86-0.86 | 0.94 - 1.81 | %20 / %22 / %59 | 0-1 (%12), 0-2 (%11), 1-1 (%10) | 9.4 / 13 | 4.0 / 2 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Konyaspor - Rizespor** | **0-1** | 0.61-1.52 | 1.43 - 1.18 | %44 / %25 / %32 | 1-0 (%11), 1-1 (%12), 0-1 (%9) | 9.2 / 8 | 4.5 / 5 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Gaziantep FK - Alanyaspor** | **1-1** | 1.28-1.25 | 1.23 - 1.17 | %34 / %34 / %32 | 1-1 (%16), 0-0 (%11), 1-0 (%10) | 8.9 / 4 | 4.4 / 6 | 🎯 Tam İsabet (Exact Hit) |
| **Genclerbirligi - Fenerbahce** | **2-1** | 1.19-2.90 | 0.78 - 2.46 | %10 / %15 / %75 | 0-2 (%12), 0-1 (%10), 0-3 (%10) | 10.4 / 13 | 4.2 / 1 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Basaksehir - Kocaelispor** | **2-0** | 1.96-0.44 | 1.70 - 0.78 | %60 / %23 / %17 | 1-0 (%15), 2-0 (%12), 1-1 (%11) | 8.7 / 10 | 4.2 / 4 | 🎯 Tam İsabet (Exact Hit) |
| **Amedspor - Erzurumspor FK** | **3-0** | 1.51-0.24 | 1.58 - 1.02 | %51 / %24 / %25 | 1-0 (%12), 1-1 (%11), 2-1 (%9) | 9.4 / 8 | 4.2 / 1 | ✅ Baskın İsabet (Dominant Hit) |
| **Besiktas - Eyupspor** | **1-0** | 1.29-0.09 | 2.95 - 0.68 | %83 / %11 / %6 | 2-0 (%12), 3-0 (%11), 4-0 (%8) | 10.4 / 10 | 4.4 / 7 | ✅ Baskın İsabet (Dominant Hit) |
| **Samsunspor - Goztepe** | **3-3** | 2.74-1.89 | 1.52 - 1.32 | %43 / %24 / %33 | 2-1 (%9), 1-1 (%11), 1-0 (%9) | 9.4 / 12 | 4.3 / 4 | ⚖️ Kritik Denge / Beraberlik (Draw Variance) |

### 2. Hafta Karşılaşmaları

| Maç | Gerçek Skor | Gerçek xG | Tahmin (λ - μ) | Model 1X2 % | İlk 3 Skor Tahmini | Korner (T/G) | Kart (T/G) | Teşhis |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Erzurumspor FK - Galatasaray** | **0-4** | 0.48-3.08 | 0.69 - 2.47 | %9 / %15 / %77 | 0-2 (%13), 0-1 (%11), 0-3 (%11) | 9.6 / 4 | 3.5 / 1 | ✅ Baskın İsabet (Dominant Hit) |
| **Corum FK - Kasimpasa** | **0-1** | 1.10-0.31 | 1.84 - 0.96 | %58 / %22 / %20 | 1-0 (%12), 2-0 (%10), 1-1 (%10) | 9.1 / 7 | 4.1 / 4 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Rizespor - Samsunspor** | **0-2** | 1.85-0.32 | 1.89 - 1.22 | %53 / %21 / %25 | 2-1 (%10), 1-1 (%10), 1-0 (%9) | 9.7 / 11 | 4.2 / 2 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Fenerbahce - Konyaspor** | **4-2** | 2.36-2.15 | 2.92 - 0.60 | %84 / %10 / %5 | 2-0 (%13), 3-0 (%12), 4-0 (%9) | 10.9 / 10 | 4.3 / 3 | ✅ Baskın İsabet (Dominant Hit) |
| **Eyupspor - Gaziantep FK** | **0-1** | 0.49-1.16 | 1.10 - 1.32 | %32 / %26 / %42 | 0-1 (%12), 1-1 (%12), 1-0 (%10) | 9.1 / 6 | 4.6 / 3 | 🎯 Tam İsabet (Exact Hit) |
| **Trabzonspor - Basaksehir** | **2-1** | 2.27-1.95 | 1.77 - 1.20 | %51 / %22 / %26 | 2-1 (%10), 1-1 (%10), 1-0 (%9) | 9.8 / 6 | 4.0 / 7 | 🎯 Tam İsabet (Exact Hit) |
| **Alanyaspor - Besiktas** | **1-0** | 0.54-0.74 | 0.84 - 1.58 | %20 / %24 / %55 | 0-1 (%15), 1-1 (%11), 0-2 (%11) | 8.9 / 14 | 4.2 / 0 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Goztepe - Genclerbirligi** | **0-1** | 1.92-1.56 | 1.79 - 1.12 | %54 / %22 / %24 | 2-1 (%10), 1-1 (%10), 1-0 (%10) | 9.9 / 11 | 3.8 / 3 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Kocaelispor - Amedspor** | **2-0** | 1.30-0.59 | 0.82 - 1.16 | %26 / %29 / %44 | 0-1 (%17), 0-0 (%13), 1-1 (%13) | 8.3 / 10 | 3.9 / 2 | ⚠️ Model / Taktiksel Sapma (Model Miss) |

### 3. Hafta Karşılaşmaları

| Maç | Gerçek Skor | Gerçek xG | Tahmin (λ - μ) | Model 1X2 % | İlk 3 Skor Tahmini | Korner (T/G) | Kart (T/G) | Teşhis |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Genclerbirligi - Erzurumspor FK** | **1-1** | 0.76-0.86 | 1.86 - 0.64 | %67 / %20 / %12 | 1-0 (%16), 2-0 (%14), 1-1 (%9) | 8.3 / 8 | 3.5 / 6 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Konyaspor - Kocaelispor** | **1-2** | 0.86-0.77 | 1.17 - 0.90 | %42 / %29 / %29 | 1-0 (%15), 1-1 (%13), 0-0 (%12) | 8.6 / 5 | 3.9 / 6 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Galatasaray - Goztepe** | **3-2** | 2.92-0.63 | 3.30 - 0.87 | %84 / %10 / %6 | 3-1 (%8), 3-0 (%9), 2-0 (%8) | 10.3 / 8 | 3.9 / 4 | ✅ Baskın İsabet (Dominant Hit) |
| **Gaziantep FK - Rizespor** | **1-2** | 0.80-0.93 | 1.15 - 1.25 | %32 / %31 / %37 | 0-1 (%11), 1-1 (%15), 1-0 (%10) | 9.0 / 8 | 4.1 / 3 | ✅ Baskın İsabet (Dominant Hit) |
| **Eyupspor - Alanyaspor** | **2-1** | 1.47-0.66 | 0.70 - 1.26 | %21 / %28 / %51 | 0-1 (%18), 0-0 (%13), 1-1 (%12) | 8.5 / 3 | 4.2 / 6 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Basaksehir - Kasimpasa** | **1-1** | 0.77-1.58 | 1.79 - 0.86 | %60 / %22 / %18 | 1-0 (%13), 2-0 (%11), 1-1 (%10) | 9.1 / 8 | 4.2 / 4 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Samsunspor - Fenerbahce** | **0-2** | 0.46-1.56 | 1.21 - 2.54 | %17 / %17 / %67 | 1-2 (%9), 1-3 (%8), 0-2 (%8) | 10.7 / 7 | 3.9 / 1 | 🎯 Tam İsabet (Exact Hit) |
| **Amedspor - Trabzonspor** | **2-1** | 2.39-0.84 | 1.13 - 1.78 | %25 / %22 / %53 | 1-2 (%10), 1-1 (%10), 0-1 (%10) | 9.5 / 7 | 3.8 / 8 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Besiktas - Corum FK** | **6-2** | 2.86-0.60 | 1.81 - 0.80 | %62 / %22 / %16 | 1-0 (%14), 2-0 (%12), 1-1 (%10) | 9.3 / 14 | 3.5 / 5 | ✅ Baskın İsabet (Dominant Hit) |

### 4. Hafta Karşılaşmaları

| Maç | Gerçek Skor | Gerçek xG | Tahmin (λ - μ) | Model 1X2 % | İlk 3 Skor Tahmini | Korner (T/G) | Kart (T/G) | Teşhis |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Basaksehir - Galatasaray** | **2-3** | 1.98-2.12 | 1.25 - 2.18 | %22 / %19 / %59 | 1-2 (%10), 1-1 (%8), 0-2 (%8) | 9.4 / 11 | 4.0 / 7 | ✅ Baskın İsabet (Dominant Hit) |
| **Erzurumspor FK - Konyaspor** | **1-0** | 0.98-1.44 | 1.00 - 1.30 | %29 / %27 / %44 | 0-1 (%14), 1-1 (%12), 1-0 (%10) | 8.2 / 7 | 3.8 / 5 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Fenerbahce - Besiktas** | **1-2** | 0.59-1.75 | 1.89 - 1.32 | %51 / %22 / %27 | 2-1 (%10), 1-1 (%10), 1-0 (%8) | 10.6 / 9 | 4.2 / 7 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Corum FK - Eyupspor** | **3-0** | 2.58-0.72 | 1.79 - 0.95 | %58 / %22 / %20 | 1-0 (%12), 1-1 (%10), 2-0 (%10) | 9.1 / 11 | 4.0 / 3 | ✅ Baskın İsabet (Dominant Hit) |
| **Kasimpasa - Amedspor** | **2-2** | 1.03-1.11 | 1.06 - 1.14 | %31 / %34 / %35 | 0-1 (%12), 1-1 (%16), 0-0 (%13) | 8.7 / 11 | 4.6 / 7 | ⚖️ Kritik Denge / Beraberlik (Draw Variance) |
| **Kocaelispor - Samsunspor** | **1-0** | 1.59-1.74 | 1.27 - 0.97 | %44 / %27 / %29 | 1-0 (%14), 1-1 (%13), 0-1 (%11) | 8.7 / 8 | 3.9 / 1 | 🎯 Tam İsabet (Exact Hit) |
| **Trabzonspor - Genclerbirligi** | **5-0** | 3.20-0.86 | 2.14 - 0.84 | %68 / %18 / %14 | 2-0 (%12), 1-0 (%11), 2-1 (%10) | 10.1 / 9 | 3.7 / 4 | ✅ Baskın İsabet (Dominant Hit) |
| **Goztepe - Gaziantep FK** | **2-4** | 3.05-1.23 | 1.75 - 1.09 | %53 / %23 / %24 | 2-1 (%10), 1-0 (%11), 1-1 (%11) | 9.3 / 11 | 4.3 / 7 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Rizespor - Alanyaspor** | **0-1** | 1.29-2.06 | 1.43 - 0.90 | %50 / %26 / %24 | 1-0 (%14), 1-1 (%12), 2-0 (%10) | 8.5 / 15 | 4.1 / 2 | ⚠️ Model / Taktiksel Sapma (Model Miss) |

### 5. Hafta Karşılaşmaları

| Maç | Gerçek Skor | Gerçek xG | Tahmin (λ - μ) | Model 1X2 % | İlk 3 Skor Tahmini | Korner (T/G) | Kart (T/G) | Teşhis |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Besiktas - Erzurumspor FK** | **3-0** | 2.29-0.49 | 2.83 - 0.44 | %86 / %10 / %4 | 2-0 (%15), 3-0 (%14), 1-0 (%11) | 9.2 / 11 | 3.4 / 3 | 🎯 Tam İsabet (Exact Hit) |
| **Eyupspor - Rizespor** | **0-2** | 1.43-2.05 | 0.89 - 1.34 | %26 / %27 / %48 | 0-1 (%15), 1-1 (%12), 0-0 (%10) | 8.9 / 10 | 4.2 / 5 | ✅ Baskın İsabet (Dominant Hit) |
| **Samsunspor - Corum FK** | **1-5** | 1.24-1.73 | 1.65 - 1.61 | %36 / %30 / %34 | 2-1 (%8), 1-1 (%13), 2-2 (%9) | 9.9 / 10 | 3.6 / 3 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Alanyaspor - Goztepe** | **2-2** | 1.73-1.79 | 1.63 - 1.17 | %49 / %23 / %28 | 2-1 (%9), 1-1 (%11), 1-0 (%10) | 9.2 / 12 | 4.2 / 4 | ⚖️ Kritik Denge / Beraberlik (Draw Variance) |
| **Konyaspor - Trabzonspor** | **1-0** | 1.05-0.92 | 1.07 - 2.14 | %18 / %19 / %62 | 1-2 (%10), 0-2 (%9), 0-1 (%9) | 9.6 / 10 | 4.1 / 6 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Genclerbirligi - Kasimpasa** | **1-2** | 1.15-1.31 | 1.07 - 1.26 | %32 / %27 / %41 | 0-1 (%13), 1-1 (%13), 1-0 (%11) | 9.1 / 17 | 4.2 / 7 | ✅ Baskın İsabet (Dominant Hit) |
| **Amedspor - Basaksehir** | **5-0** | 4.25-0.40 | 1.73 - 1.44 | %45 / %22 / %33 | 2-1 (%9), 1-1 (%10), 1-0 (%8) | 9.6 / 6 | 4.3 / 3 | ✅ Baskın İsabet (Dominant Hit) |
| **Galatasaray - Kocaelispor** | **1-0** | 2.26-0.82 | 2.37 - 0.66 | %76 / %15 / %9 | 2-0 (%14), 1-0 (%12), 3-0 (%11) | 9.2 / 19 | 3.8 / 5 | 🎯 Tam İsabet (Exact Hit) |
| **Gaziantep FK - Fenerbahce** | **0-0** | 0.49-1.33 | 1.05 - 2.25 | %17 / %18 / %65 | 1-2 (%10), 0-2 (%9), 0-1 (%9) | 10.2 / 12 | 4.2 / 7 | ⚠️ Model / Taktiksel Sapma (Model Miss) |

### 6. Hafta Karşılaşmaları

| Maç | Gerçek Skor | Gerçek xG | Tahmin (λ - μ) | Model 1X2 % | İlk 3 Skor Tahmini | Korner (T/G) | Kart (T/G) | Teşhis |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Kasimpasa - Konyaspor** | **0-0** | 0.56-1.15 | 1.45 - 0.93 | %50 / %25 / %25 | 1-0 (%14), 1-1 (%12), 2-0 (%10) | 9.2 / 10 | 4.5 / 7 | ⚖️ Kritik Denge / Beraberlik (Draw Variance) |
| **Corum FK - Alanyaspor** | **1-2** | 1.14-1.16 | 2.09 - 1.16 | %59 / %20 / %21 | 2-1 (%10), 1-1 (%9), 2-0 (%9) | 9.5 / 6 | 3.8 / 5 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Kocaelispor - Gaziantep FK** | **2-0** | 1.16-0.88 | 1.14 - 0.87 | %43 / %29 / %28 | 1-0 (%16), 0-0 (%13), 1-1 (%13) | 8.8 / 3 | 4.2 / 3 | ✅ Baskın İsabet (Dominant Hit) |
| **Basaksehir - Genclerbirligi** | **4-0** | 2.78-0.46 | 2.22 - 1.15 | %62 / %19 / %19 | 2-1 (%10), 2-0 (%8), 1-1 (%8) | 10.1 / 10 | 4.1 / 3 | ✅ Baskın İsabet (Dominant Hit) |
| **Trabzonspor - Galatasaray** | **4-0** | 2.28-0.86 | 1.50 - 1.78 | %33 / %22 / %45 | 1-2 (%9), 1-1 (%10), 2-1 (%8) | 9.7 / 6 | 4.5 / 7 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Erzurumspor FK - Samsunspor** | **1-0** | 0.72-0.36 | 1.05 - 1.16 | %31 / %32 / %36 | 0-1 (%12), 1-1 (%15), 0-0 (%12) | 8.5 / 8 | 3.3 / 5 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Fenerbahce - Eyupspor** | **8-0** | 5.14-0.72 | 2.78 - 0.49 | %85 / %11 / %4 | 2-0 (%15), 3-0 (%14), 1-0 (%11) | 10.5 / 14 | 4.1 / 2 | ✅ Baskın İsabet (Dominant Hit) |
| **Amedspor - Besiktas** | **3-2** | 1.54-2.04 | 1.55 - 1.57 | %34 / %31 / %35 | 1-2 (%8), 1-1 (%14), 2-2 (%9) | 9.8 / 11 | 4.0 / 14 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Goztepe - Rizespor** | **2-2** | 2.78-0.78 | 1.65 - 1.48 | %42 / %23 / %35 | 2-1 (%9), 1-1 (%10), 1-2 (%8) | 9.8 / 7 | 4.3 / 6 | ⚖️ Kritik Denge / Beraberlik (Draw Variance) |

### 7. Hafta Karşılaşmaları

| Maç | Gerçek Skor | Gerçek xG | Tahmin (λ - μ) | Model 1X2 % | İlk 3 Skor Tahmini | Korner (T/G) | Kart (T/G) | Teşhis |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Galatasaray - Kasimpasa** | **3-1** | 2.40-0.51 | 2.18 - 0.83 | %69 / %18 / %13 | 2-0 (%12), 1-0 (%11), 2-1 (%10) | 10.2 / 9 | 4.2 / 8 | ✅ Baskın İsabet (Dominant Hit) |
| **Genclerbirligi - Amedspor** | **2-2** | 0.50-2.16 | 1.01 - 2.33 | %15 / %17 / %67 | 1-2 (%10), 0-2 (%10), 0-1 (%9) | 10.1 / 9 | 4.4 / 8 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Alanyaspor - Erzurumspor FK** | **0-0** | 0.98-1.69 | 1.45 - 0.58 | %59 / %26 / %15 | 1-0 (%19), 2-0 (%14), 0-0 (%13) | 8.3 / 8 | 3.7 / 6 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Samsunspor - Trabzonspor** | **0-3** | 1.66-2.02 | 1.05 - 2.03 | %19 / %20 / %61 | 1-2 (%10), 0-1 (%10), 0-2 (%10) | 9.6 / 14 | 3.8 / 4 | ✅ Baskın İsabet (Dominant Hit) |
| **Rizespor - Fenerbahce** | **0-3** | 1.52-0.96 | 1.08 - 2.20 | %18 / %19 / %63 | 1-2 (%10), 0-2 (%9), 0-1 (%9) | 10.3 / 7 | 4.1 / 0 | ✅ Baskın İsabet (Dominant Hit) |
| **Konyaspor - Basaksehir** | **0-0** | 0.41-0.97 | 1.46 - 1.32 | %40 / %25 / %34 | 2-1 (%9), 1-1 (%12), 1-0 (%9) | 9.1 / 8 | 4.4 / 0 | ⚖️ Kritik Denge / Beraberlik (Draw Variance) |
| **Gaziantep FK - Corum FK** | **0-3** | 0.99-2.67 | 1.50 - 1.60 | %34 / %27 / %38 | 1-2 (%8), 1-1 (%12), 2-2 (%8) | 9.3 / 14 | 4.3 / 5 | ✅ Baskın İsabet (Dominant Hit) |
| **Besiktas - Kocaelispor** | **2-1** | 2.21-1.65 | 2.18 - 0.56 | %75 / %16 / %8 | 2-0 (%15), 1-0 (%14), 3-0 (%11) | 10.0 / 11 | 3.9 / 3 | ✅ Baskın İsabet (Dominant Hit) |
| **Eyupspor - Goztepe** | **0-4** | 1.39-3.21 | 1.13 - 2.02 | %21 / %20 / %59 | 1-2 (%10), 1-1 (%9), 0-1 (%9) | 9.6 / 10 | 4.3 / 6 | ✅ Baskın İsabet (Dominant Hit) |

### 8. Hafta Karşılaşmaları

| Maç | Gerçek Skor | Gerçek xG | Tahmin (λ - μ) | Model 1X2 % | İlk 3 Skor Tahmini | Korner (T/G) | Kart (T/G) | Teşhis |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Corum FK - Rizespor** | **2-1** | 0.71-0.84 | 2.26 - 1.22 | %61 / %19 / %20 | 2-1 (%10), 1-1 (%8), 2-0 (%8) | 9.9 / 10 | 4.1 / 4 | 🎯 Tam İsabet (Exact Hit) |
| **Erzurumspor FK - Eyupspor** | **2-0** | 1.29-0.43 | 1.52 - 0.56 | %61 / %25 / %14 | 1-0 (%19), 2-0 (%15), 0-0 (%12) | 8.3 / 5 | 3.9 / 1 | 🎯 Tam İsabet (Exact Hit) |
| **Amedspor - Konyaspor** | **1-0** | 1.65-0.25 | 2.13 - 0.84 | %68 / %18 / %14 | 2-0 (%12), 1-0 (%11), 2-1 (%10) | 9.3 / 9 | 4.6 / 4 | 🎯 Tam İsabet (Exact Hit) |
| **Genclerbirligi - Galatasaray** | **0-1** | 1.74-1.92 | 0.92 - 2.66 | %11 / %14 / %75 | 1-2 (%9), 0-2 (%10), 0-3 (%9) | 10.6 / 19 | 4.0 / 1 | ✅ Baskın İsabet (Dominant Hit) |
| **Fenerbahce - Alanyaspor** | **3-0** | 3.84-0.96 | 2.53 - 0.73 | %77 / %14 / %9 | 2-0 (%12), 3-0 (%10), 1-0 (%10) | 9.8 / 7 | 3.8 / 1 | 🎯 Tam İsabet (Exact Hit) |
| **Kasimpasa - Samsunspor** | **4-1** | 2.41-1.07 | 1.45 - 0.96 | %49 / %25 / %25 | 1-0 (%13), 1-1 (%12), 2-0 (%9) | 9.4 / 15 | 4.1 / 3 | ✅ Baskın İsabet (Dominant Hit) |
| **Kocaelispor - Goztepe** | **1-2** | 1.61-0.83 | 1.45 - 1.29 | %42 / %24 / %34 | 2-1 (%9), 1-1 (%12), 1-0 (%10) | 9.2 / 6 | 4.1 / 4 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Basaksehir - Gaziantep FK** | **2-1** | 1.95-1.42 | 2.06 - 1.00 | %62 / %20 / %18 | 2-1 (%10), 1-0 (%10), 2-0 (%10) | 9.3 / 9 | 4.2 / 8 | 🎯 Tam İsabet (Exact Hit) |
| **Trabzonspor - Besiktas** | **2-0** | 2.35-1.19 | 1.86 - 1.50 | %47 / %22 / %32 | 2-1 (%9), 1-1 (%9), 1-2 (%7) | 10.3 / 10 | 4.6 / 5 | ✅ Baskın İsabet (Dominant Hit) |
