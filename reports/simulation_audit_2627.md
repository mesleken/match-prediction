# 🧪 2026/27 Süper Lig Simülasyon Denetimi ve Model Geliştirme Raporu (V2.2)
**Denetim Tarihi:** 2026-10-24 14:26 UTC | **İncelenen Maç Sayısı:** 72 Maç (1–8. Hafta Tamamı)
**Veri Sızıntısı:** %0 (Strictly Out-of-Sample, her maç yalnızca maç saatinden önceki verilerle simüle edilmiştir)

## 1. Yönetici Özeti ve Model Karşılaştırma Karnesi

| Değerlendirme Metriği | Baz Model (V2.0) | Geliştirilmiş Model (V2.2) | Değişim / İyileşme |
| :--- | :---: | :---: | :---: |
| **RPS (Ranked Probability Score)** | 0.2197 | **0.2067** | 📉 -5.91% |
| **Log-Loss (Exact Score NLL)** | 3.0772 | **2.9782** | 📉 -3.22% |
| **1X2 İsabet Oranı (Hit Rate)** | %43.1 | **%55.6** | +12.5% |
| **Çifte Şans İsabet Oranı** | %76.4 | **%83.3** | +6.9% |
| **Alt/Üst 2.5 İsabet Oranı** | %63.9 | **%70.8** | +6.9% |
| **İlk 1 (Tam) Skor Yakalama** | %5.6 | **%8.3** | +2.8% |
| **İlk 3 Skor Yakalama Oranı** | %18.1 | **%26.4** | +8.3% |
| **İlk 5 Skor Yakalama Oranı** | %41.7 | **%43.1** | +1.4% |
| **xG Farkı Hatası (xG Diff MAE)** | 0.882 | **0.872** | -0.010 |
| **Korner Beklentisi MAE** | 2.54 | **2.51** | -0.03 |
| **Kart Beklentisi MAE** | 2.00 | **1.95** | -0.05 |

## 2. Hata Teşhisi: Model Hatası vs. Bitiricilik / Şans Ayrıştırması (Variance Decomposition)

Gerçekleşen gol farkı ile model beklentisi arasındaki sapmanın matematiksel varyans ayrıştırması:

- **Model Kurgu Hatası (Model Setup MSE — $(\lambda_H - \mu_A) - (xG_H - xG_A)$):** `1.205` (**%33.5**)
- **Bitiricilik & Şans Varyansı (Finishing Luck MSE — $(xG_H - xG_A) - (G_H - G_A)$):** `2.394` (**%66.5**)
- **Toplam Skor Farkı MSE:** `3.628`
> [!NOTE]
> Yanılgıların yaklaşık **%67'i**, modelin saha içi pozisyon kalitesini ($xG$) doğru tahmin etmesine rağmen oyuncuların olağan dışı şut bitiriciliği veya kaleci kurtarışları nedeniyle gerçekleşen şans faktöründen kaynaklanmaktadır.

### Teşhis Dağılımı (Diagnosis Breakdown)

- **✅ Baskın İsabet (Dominant Hit):** 27 maç (%37.5)
- **⚠️ Model / Taktiksel Sapma (Model Miss):** 16 maç (%22.2)
- **🎯 Tam İsabet (Exact Hit):** 13 maç (%18.1)
- **🍀 Şans / Bitiricilik Sapması (Finishing Luck):** 11 maç (%15.3)
- **⚖️ Kritik Denge / Beraberlik (Draw Variance):** 5 maç (%6.9)

## 3. Hafta Hafta Performans İlerlemesi (Round-by-Round Learning)

| Hafta | Maç Sayısı | RPS | 1X2 İsabet % | Çifte Şans % | İlk 3 Skor % | Korner MAE | Kart MAE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Hafta | 9 | 0.2185 | %44.4 | %77.8 | %44.4 | 2.13 | 1.69 |
| 2. Hafta | 9 | 0.2907 | %44.4 | %66.7 | %22.2 | 2.73 | 2.02 |
| 3. Hafta | 9 | 0.2303 | %44.4 | %88.9 | %33.3 | 2.80 | 1.84 |
| 4. Hafta | 9 | 0.2399 | %55.6 | %88.9 | %22.2 | 2.05 | 2.00 |
| 5. Hafta | 9 | 0.2038 | %66.7 | %77.8 | %22.2 | 3.26 | 1.34 |
| 6. Hafta | 9 | 0.2223 | %33.3 | %77.8 | %0.0 | 2.45 | 2.66 |
| 7. Hafta | 9 | 0.1322 | %66.7 | %88.9 | %11.1 | 1.95 | 2.42 |
| 8. Hafta | 9 | 0.1158 | %88.9 | %100.0 | %55.6 | 2.70 | 1.64 |

## 4. Takım Bazlı Sistematik Yanlılık Analizi (Team Bias Table)

Modelin hangi takımları daha kolay çözdüğü ve hangi takımlarda sistematik sapma gösterdiği tablosu:

| Takım | Maç | Gerçek Gol (A-Y) | Gerçek xG (A-Y) | Model Bekl. (A-Y) | Hücum Sapması (G) | Savunma Sapması (G) | 1X2 Doğruluk % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Eyupspor** | 8 | 2-22 | 6.7-17.4 | 6.4-15.2 | +0.55 | -0.85 | %87.5 |
| **Galatasaray** | 8 | 17-11 | 18.5-9.1 | 19.3-7.6 | +0.29 | -0.42 | %75.0 |
| **Basaksehir** | 8 | 12-12 | 12.8-12.9 | 13.1-10.9 | +0.13 | -0.14 | %75.0 |
| **Gaziantep FK** | 8 | 8-12 | 8.2-12.8 | 9.1-12.3 | +0.14 | +0.04 | %75.0 |
| **Fenerbahce** | 8 | 22-6 | 18.7-9.2 | 19.7-7.2 | -0.29 | +0.16 | %62.5 |
| **Besiktas** | 8 | 16-10 | 14.4-7.8 | 15.9-8.6 | -0.01 | -0.18 | %62.5 |
| **Samsunspor** | 8 | 7-19 | 9.6-13.8 | 9.7-13.4 | +0.34 | -0.70 | %62.5 |
| **Kocaelispor** | 8 | 9-8 | 9.3-11.3 | 7.7-11.8 | -0.17 | +0.47 | %62.5 |
| **Corum FK** | 8 | 18-13 | 11.1-11.0 | 12.9-11.5 | -0.63 | -0.19 | %62.5 |
| **Trabzonspor** | 8 | 18-5 | 14.7-11.8 | 15.2-9.4 | -0.35 | +0.55 | %50.0 |
| **Kasimpasa** | 8 | 12-9 | 9.6-9.6 | 8.8-11.6 | -0.40 | +0.33 | %50.0 |
| **Rizespor** | 8 | 8-11 | 10.8-9.7 | 10.9-11.7 | +0.36 | +0.09 | %50.0 |
| **Genclerbirligi** | 8 | 7-16 | 8.2-17.1 | 8.7-15.6 | +0.21 | -0.04 | %50.0 |
| **Erzurumspor FK** | 8 | 5-11 | 6.8-10.8 | 6.9-13.3 | +0.23 | +0.28 | %50.0 |
| **Amedspor** | 8 | 18-9 | 15.2-6.6 | 12.8-9.5 | -0.65 | +0.06 | %50.0 |
| **Konyaspor** | 8 | 4-9 | 7.9-9.7 | 8.8-13.1 | +0.59 | +0.52 | %25.0 |
| **Alanyaspor** | 8 | 8-9 | 9.3-13.2 | 9.2-11.3 | +0.15 | +0.29 | %25.0 |
| **Goztepe** | 8 | 17-16 | 16.1-14.0 | 11.9-12.9 | -0.64 | -0.39 | %25.0 |

## 5. Yüksek Güvenli Favori Sürprizleri Analizi (Upset Post-Mortem)

Modelin favoriye %52'den fazla kazanma şansı verdiği ancak favorinin puan kaybettiği maçlar:

| Hafta | Maç | Skor | Model Favorisi | Kazanma İhtimali | Gerçek xG Farkı | Sürprizin Temel Nedeni |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1. H | Genclerbirligi - Fenerbahce | **2-1** | Fenerbahce | %75.8 | -1.71 | Bitiricilik Şanssızlığı (Favori xG'yi domine etti ama puan kaybetti) |
| 1. H | Galatasaray - Corum FK | **2-2** | Galatasaray | %70.1 | +2.29 | Bitiricilik Şanssızlığı (Favori xG'yi domine etti ama puan kaybetti) |
| 7. H | Genclerbirligi - Amedspor | **2-2** | Amedspor | %67.7 | -1.66 | Bitiricilik Şanssızlığı (Favori xG'yi domine etti ama puan kaybetti) |
| 3. H | Genclerbirligi - Erzurumspor FK | **1-1** | Genclerbirligi | %66.9 | -0.10 | Dengeli Maç / Yüksek Varyans |
| 5. H | Gaziantep FK - Fenerbahce | **0-0** | Fenerbahce | %65.0 | -0.84 | Bitiricilik Şanssızlığı (Favori xG'yi domine etti ama puan kaybetti) |
| 5. H | Konyaspor - Trabzonspor | **1-0** | Trabzonspor | %63.8 | +0.13 | Dengeli Maç / Yüksek Varyans |
| 3. H | Basaksehir - Kasimpasa | **1-1** | Basaksehir | %60.2 | -0.81 | Taktiksel Yanılma (Rakip sahada xG üstünlüğünü kurdu) |
| 1. H | Kasimpasa - Trabzonspor | **1-1** | Trabzonspor | %59.7 | +1.00 | Taktiksel Yanılma (Rakip sahada xG üstünlüğünü kurdu) |
| 6. H | Corum FK - Alanyaspor | **1-2** | Corum FK | %59.5 | -0.02 | Dengeli Maç / Yüksek Varyans |
| 7. H | Alanyaspor - Erzurumspor FK | **0-0** | Alanyaspor | %59.3 | -0.71 | Taktiksel Yanılma (Rakip sahada xG üstünlüğünü kurdu) |
| 2. H | Corum FK - Kasimpasa | **0-1** | Corum FK | %58.9 | +0.79 | Bitiricilik Şanssızlığı (Favori xG'yi domine etti ama puan kaybetti) |
| 2. H | Alanyaspor - Besiktas | **1-0** | Besiktas | %56.0 | -0.20 | Dengeli Maç / Yüksek Varyans |
| 2. H | Rizespor - Samsunspor | **0-2** | Rizespor | %54.5 | +1.53 | Bitiricilik Şanssızlığı (Favori xG'yi domine etti ama puan kaybetti) |
| 3. H | Amedspor - Trabzonspor | **2-1** | Trabzonspor | %53.9 | +1.55 | Taktiksel Yanılma (Rakip sahada xG üstünlüğünü kurdu) |
| 2. H | Goztepe - Genclerbirligi | **0-1** | Goztepe | %53.7 | +0.36 | Dengeli Maç / Yüksek Varyans |
| 4. H | Goztepe - Gaziantep FK | **2-4** | Goztepe | %53.5 | +1.82 | Bitiricilik Şanssızlığı (Favori xG'yi domine etti ama puan kaybetti) |

## 6. Hakem Etkisi, Kart Puanı ve Oyun Temposu Karnesi (Referee Officiating Impact)

2026/27 sezonunda görev alan hakemlerin maç başına beklenen vs. gerçekleşen kart puanları ve tempo/xG akışı:

| Hakem Adı | Maç | Gerçekleşen Kart Pts | Model Kart Pts | Kart Pts MAE | Gerçek Kart | Model Kart | Gerçek xG | Model xG |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bilinmiyor** | 72 | 44.0 pts | 44.3 pts | 20.4 pts | 4.3 | 4.1 | 2.89 | 2.87 |

## 7. Oynanmış 72 Maçın Hafta Hafta Detaylı Simülasyon Tablosu (1–8. Hafta)

### 1. Hafta Karşılaşmaları

| Maç | Gerçek Skor | Gerçek xG | Tahmin (λ - μ) | Model 1X2 % | İlk 3 Skor Tahmini | Korner (T/G) | Kart (T/G) | Teşhis |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Galatasaray - Corum FK** | **2-2** | 2.90-0.61 | 2.34 - 0.89 | %70 / %17 / %13 | 2-1 (%10), 2-0 (%11), 1-0 (%10) | 9.7 / 9 | 4.2 / 3 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Kasimpasa - Trabzonspor** | **1-1** | 1.86-0.86 | 0.91 - 1.83 | %19 / %22 / %60 | 0-1 (%12), 0-2 (%11), 1-1 (%10) | 9.4 / 13 | 4.2 / 2 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Konyaspor - Rizespor** | **0-1** | 0.61-1.52 | 1.44 - 1.18 | %43 / %25 / %32 | 1-0 (%11), 1-1 (%12), 0-1 (%9) | 9.3 / 8 | 4.5 / 5 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Gaziantep FK - Alanyaspor** | **1-1** | 1.28-1.25 | 1.23 - 1.19 | %34 / %35 / %32 | 1-1 (%16), 0-0 (%11), 1-0 (%10) | 9.1 / 4 | 4.5 / 6 | 🎯 Tam İsabet (Exact Hit) |
| **Genclerbirligi - Fenerbahce** | **2-1** | 1.19-2.90 | 0.76 - 2.51 | %9 / %15 / %76 | 0-2 (%12), 0-3 (%10), 0-1 (%10) | 10.4 / 13 | 4.2 / 1 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Basaksehir - Kocaelispor** | **2-0** | 1.96-0.44 | 1.72 - 0.78 | %60 / %23 / %17 | 1-0 (%15), 2-0 (%12), 1-1 (%11) | 8.6 / 10 | 4.2 / 4 | 🎯 Tam İsabet (Exact Hit) |
| **Amedspor - Erzurumspor FK** | **3-0** | 1.51-0.24 | 1.64 - 0.96 | %54 / %24 / %22 | 1-0 (%13), 1-1 (%11), 2-0 (%10) | 9.4 / 8 | 4.3 / 1 | ✅ Baskın İsabet (Dominant Hit) |
| **Besiktas - Eyupspor** | **1-0** | 1.29-0.09 | 3.08 - 0.66 | %85 / %10 / %5 | 3-0 (%12), 2-0 (%11), 4-0 (%9) | 10.5 / 10 | 4.2 / 7 | ✅ Baskın İsabet (Dominant Hit) |
| **Samsunspor - Goztepe** | **3-3** | 2.74-1.89 | 1.54 - 1.33 | %43 / %24 / %33 | 2-1 (%9), 1-1 (%11), 1-0 (%9) | 9.5 / 12 | 4.3 / 4 | ⚖️ Kritik Denge / Beraberlik (Draw Variance) |

### 2. Hafta Karşılaşmaları

| Maç | Gerçek Skor | Gerçek xG | Tahmin (λ - μ) | Model 1X2 % | İlk 3 Skor Tahmini | Korner (T/G) | Kart (T/G) | Teşhis |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Erzurumspor FK - Galatasaray** | **0-4** | 0.48-3.08 | 0.67 - 2.47 | %8 / %14 / %77 | 0-2 (%13), 0-1 (%11), 0-3 (%11) | 9.5 / 4 | 3.7 / 1 | ✅ Baskın İsabet (Dominant Hit) |
| **Corum FK - Kasimpasa** | **0-1** | 1.10-0.31 | 1.84 - 0.94 | %59 / %22 / %19 | 1-0 (%12), 2-0 (%10), 1-1 (%10) | 9.1 / 7 | 4.1 / 4 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Rizespor - Samsunspor** | **0-2** | 1.85-0.32 | 1.93 - 1.22 | %54 / %21 / %24 | 2-1 (%10), 1-1 (%10), 1-0 (%9) | 9.7 / 11 | 4.2 / 2 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Fenerbahce - Konyaspor** | **4-2** | 2.36-2.15 | 2.93 - 0.59 | %85 / %10 / %5 | 2-0 (%13), 3-0 (%12), 4-0 (%9) | 10.9 / 10 | 4.2 / 3 | ✅ Baskın İsabet (Dominant Hit) |
| **Eyupspor - Gaziantep FK** | **0-1** | 0.49-1.16 | 1.07 - 1.32 | %31 / %26 / %43 | 0-1 (%12), 1-1 (%12), 1-0 (%10) | 9.0 / 6 | 4.6 / 3 | 🎯 Tam İsabet (Exact Hit) |
| **Trabzonspor - Basaksehir** | **2-1** | 2.27-1.95 | 1.77 - 1.20 | %51 / %22 / %26 | 2-1 (%10), 1-1 (%10), 1-0 (%9) | 9.8 / 6 | 3.9 / 7 | 🎯 Tam İsabet (Exact Hit) |
| **Alanyaspor - Besiktas** | **1-0** | 0.54-0.74 | 0.83 - 1.59 | %20 / %24 / %56 | 0-1 (%15), 0-2 (%11), 1-1 (%11) | 8.9 / 14 | 4.2 / 0 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Goztepe - Genclerbirligi** | **0-1** | 1.92-1.56 | 1.80 - 1.12 | %54 / %22 / %24 | 2-1 (%10), 1-1 (%10), 1-0 (%10) | 9.9 / 11 | 4.2 / 3 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Kocaelispor - Amedspor** | **2-0** | 1.30-0.59 | 0.83 - 1.17 | %27 / %29 / %44 | 0-1 (%16), 0-0 (%13), 1-1 (%13) | 8.3 / 10 | 3.9 / 2 | ⚠️ Model / Taktiksel Sapma (Model Miss) |

### 3. Hafta Karşılaşmaları

| Maç | Gerçek Skor | Gerçek xG | Tahmin (λ - μ) | Model 1X2 % | İlk 3 Skor Tahmini | Korner (T/G) | Kart (T/G) | Teşhis |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Genclerbirligi - Erzurumspor FK** | **1-1** | 0.76-0.86 | 1.84 - 0.63 | %67 / %21 / %12 | 1-0 (%16), 2-0 (%14), 1-1 (%9) | 8.3 / 8 | 3.5 / 6 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Konyaspor - Kocaelispor** | **1-2** | 0.86-0.77 | 1.16 - 0.92 | %42 / %29 / %29 | 1-0 (%15), 1-1 (%13), 0-0 (%12) | 8.6 / 5 | 4.2 / 6 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Galatasaray - Goztepe** | **3-2** | 2.92-0.63 | 3.35 - 0.86 | %84 / %10 / %6 | 3-1 (%8), 3-0 (%9), 2-0 (%8) | 10.3 / 8 | 4.0 / 4 | ✅ Baskın İsabet (Dominant Hit) |
| **Gaziantep FK - Rizespor** | **1-2** | 0.80-0.93 | 1.13 - 1.23 | %32 / %31 / %37 | 0-1 (%11), 1-1 (%15), 1-0 (%10) | 9.0 / 8 | 4.4 / 3 | ✅ Baskın İsabet (Dominant Hit) |
| **Eyupspor - Alanyaspor** | **2-1** | 1.47-0.66 | 0.69 - 1.29 | %20 / %28 / %52 | 0-1 (%18), 0-0 (%13), 1-1 (%12) | 8.6 / 3 | 4.2 / 6 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Basaksehir - Kasimpasa** | **1-1** | 0.77-1.58 | 1.78 - 0.84 | %60 / %22 / %18 | 1-0 (%13), 2-0 (%12), 1-1 (%10) | 9.2 / 8 | 4.3 / 4 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Samsunspor - Fenerbahce** | **0-2** | 0.46-1.56 | 1.24 - 2.61 | %16 / %16 / %67 | 1-2 (%9), 1-3 (%8), 0-2 (%7) | 10.8 / 7 | 3.9 / 1 | 🎯 Tam İsabet (Exact Hit) |
| **Amedspor - Trabzonspor** | **2-1** | 2.39-0.84 | 1.12 - 1.81 | %24 / %22 / %54 | 1-2 (%10), 1-1 (%10), 0-1 (%10) | 9.5 / 7 | 3.7 / 8 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Besiktas - Corum FK** | **6-2** | 2.86-0.60 | 1.78 - 0.80 | %61 / %22 / %17 | 1-0 (%14), 2-0 (%12), 1-1 (%10) | 9.2 / 14 | 3.6 / 5 | ✅ Baskın İsabet (Dominant Hit) |

### 4. Hafta Karşılaşmaları

| Maç | Gerçek Skor | Gerçek xG | Tahmin (λ - μ) | Model 1X2 % | İlk 3 Skor Tahmini | Korner (T/G) | Kart (T/G) | Teşhis |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Basaksehir - Galatasaray** | **2-3** | 1.98-2.12 | 1.26 - 2.17 | %22 / %19 / %59 | 1-2 (%10), 1-1 (%8), 0-2 (%8) | 9.4 / 11 | 4.0 / 7 | ✅ Baskın İsabet (Dominant Hit) |
| **Erzurumspor FK - Konyaspor** | **1-0** | 0.98-1.44 | 1.00 - 1.30 | %29 / %27 / %44 | 0-1 (%14), 1-1 (%12), 1-0 (%10) | 8.2 / 7 | 3.9 / 5 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Fenerbahce - Besiktas** | **1-2** | 0.59-1.75 | 1.88 - 1.33 | %51 / %22 / %28 | 2-1 (%10), 1-1 (%10), 1-0 (%8) | 10.6 / 9 | 4.5 / 7 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Corum FK - Eyupspor** | **3-0** | 2.58-0.72 | 1.82 - 0.94 | %59 / %22 / %20 | 1-0 (%12), 2-0 (%11), 1-1 (%10) | 9.2 / 11 | 4.0 / 3 | ✅ Baskın İsabet (Dominant Hit) |
| **Kasimpasa - Amedspor** | **2-2** | 1.03-1.11 | 1.05 - 1.12 | %31 / %35 / %34 | 1-1 (%16), 0-0 (%14), 0-1 (%12) | 8.7 / 11 | 4.4 / 7 | ✅ Baskın İsabet (Dominant Hit) |
| **Kocaelispor - Samsunspor** | **1-0** | 1.59-1.74 | 1.29 - 0.97 | %44 / %27 / %28 | 1-0 (%14), 1-1 (%12), 0-1 (%11) | 8.7 / 8 | 3.9 / 1 | 🎯 Tam İsabet (Exact Hit) |
| **Trabzonspor - Genclerbirligi** | **5-0** | 3.20-0.86 | 2.16 - 0.83 | %68 / %18 / %14 | 2-0 (%12), 1-0 (%11), 2-1 (%10) | 10.1 / 9 | 3.8 / 4 | ✅ Baskın İsabet (Dominant Hit) |
| **Goztepe - Gaziantep FK** | **2-4** | 3.05-1.23 | 1.75 - 1.09 | %54 / %23 / %24 | 2-1 (%10), 1-1 (%11), 1-0 (%11) | 9.3 / 11 | 4.3 / 7 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Rizespor - Alanyaspor** | **0-1** | 1.29-2.06 | 1.42 - 0.89 | %50 / %26 / %24 | 1-0 (%14), 1-1 (%12), 2-0 (%10) | 8.5 / 15 | 4.1 / 2 | ⚠️ Model / Taktiksel Sapma (Model Miss) |

### 5. Hafta Karşılaşmaları

| Maç | Gerçek Skor | Gerçek xG | Tahmin (λ - μ) | Model 1X2 % | İlk 3 Skor Tahmini | Korner (T/G) | Kart (T/G) | Teşhis |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Besiktas - Erzurumspor FK** | **3-0** | 2.29-0.49 | 2.87 - 0.44 | %87 / %9 / %4 | 2-0 (%15), 3-0 (%14), 1-0 (%11) | 9.2 / 11 | 3.6 / 3 | 🎯 Tam İsabet (Exact Hit) |
| **Eyupspor - Rizespor** | **0-2** | 1.43-2.05 | 0.88 - 1.33 | %25 / %27 / %48 | 0-1 (%15), 1-1 (%12), 0-0 (%10) | 8.9 / 10 | 4.2 / 5 | ✅ Baskın İsabet (Dominant Hit) |
| **Samsunspor - Corum FK** | **1-5** | 1.24-1.73 | 1.63 - 1.64 | %35 / %31 / %35 | 1-2 (%7), 1-1 (%13), 2-2 (%9) | 9.9 / 10 | 3.6 / 3 | ✅ Baskın İsabet (Dominant Hit) |
| **Alanyaspor - Goztepe** | **2-2** | 1.73-1.79 | 1.65 - 1.17 | %49 / %23 / %28 | 2-1 (%10), 1-1 (%11), 1-0 (%10) | 9.2 / 12 | 4.2 / 4 | ⚖️ Kritik Denge / Beraberlik (Draw Variance) |
| **Konyaspor - Trabzonspor** | **1-0** | 1.05-0.92 | 1.06 - 2.20 | %17 / %19 / %64 | 1-2 (%10), 0-2 (%9), 0-1 (%9) | 9.6 / 10 | 4.1 / 6 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Genclerbirligi - Kasimpasa** | **1-2** | 1.15-1.31 | 1.05 - 1.27 | %31 / %27 / %42 | 0-1 (%13), 1-1 (%13), 1-0 (%11) | 9.1 / 17 | 4.2 / 7 | ✅ Baskın İsabet (Dominant Hit) |
| **Amedspor - Basaksehir** | **5-0** | 4.25-0.40 | 1.74 - 1.45 | %45 / %22 / %33 | 2-1 (%9), 1-1 (%10), 1-2 (%8) | 9.6 / 6 | 4.3 / 3 | ✅ Baskın İsabet (Dominant Hit) |
| **Galatasaray - Kocaelispor** | **1-0** | 2.26-0.82 | 2.37 - 0.67 | %76 / %15 / %9 | 2-0 (%14), 1-0 (%12), 3-0 (%11) | 9.2 / 19 | 3.8 / 5 | 🎯 Tam İsabet (Exact Hit) |
| **Gaziantep FK - Fenerbahce** | **0-0** | 0.49-1.33 | 1.05 - 2.25 | %17 / %18 / %65 | 1-2 (%10), 0-2 (%9), 0-1 (%9) | 10.2 / 12 | 4.2 / 7 | ⚠️ Model / Taktiksel Sapma (Model Miss) |

### 6. Hafta Karşılaşmaları

| Maç | Gerçek Skor | Gerçek xG | Tahmin (λ - μ) | Model 1X2 % | İlk 3 Skor Tahmini | Korner (T/G) | Kart (T/G) | Teşhis |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Kasimpasa - Konyaspor** | **0-0** | 0.56-1.15 | 1.46 - 0.92 | %50 / %25 / %24 | 1-0 (%14), 1-1 (%12), 2-0 (%10) | 9.2 / 10 | 4.5 / 7 | ⚖️ Kritik Denge / Beraberlik (Draw Variance) |
| **Corum FK - Alanyaspor** | **1-2** | 1.14-1.16 | 2.10 - 1.15 | %60 / %20 / %21 | 2-1 (%10), 1-1 (%9), 2-0 (%9) | 9.5 / 6 | 3.8 / 5 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Kocaelispor - Gaziantep FK** | **2-0** | 1.16-0.88 | 1.14 - 0.86 | %43 / %29 / %28 | 1-0 (%16), 0-0 (%13), 1-1 (%13) | 8.7 / 3 | 4.2 / 3 | ✅ Baskın İsabet (Dominant Hit) |
| **Basaksehir - Genclerbirligi** | **4-0** | 2.78-0.46 | 2.25 - 1.16 | %63 / %19 / %19 | 2-1 (%10), 2-0 (%8), 1-1 (%8) | 10.2 / 10 | 4.1 / 3 | ✅ Baskın İsabet (Dominant Hit) |
| **Trabzonspor - Galatasaray** | **4-0** | 2.28-0.86 | 1.51 - 1.76 | %34 / %22 / %44 | 1-2 (%9), 1-1 (%10), 2-1 (%8) | 9.7 / 6 | 4.5 / 7 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Erzurumspor FK - Samsunspor** | **1-0** | 0.72-0.36 | 1.07 - 1.15 | %31 / %34 / %35 | 0-1 (%12), 1-1 (%16), 0-0 (%13) | 8.5 / 8 | 3.3 / 5 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Fenerbahce - Eyupspor** | **8-0** | 5.14-0.72 | 2.77 - 0.48 | %85 / %11 / %4 | 2-0 (%15), 3-0 (%14), 1-0 (%11) | 10.5 / 14 | 4.1 / 2 | ✅ Baskın İsabet (Dominant Hit) |
| **Amedspor - Besiktas** | **3-2** | 1.54-2.04 | 1.53 - 1.58 | %34 / %31 / %36 | 1-2 (%8), 1-1 (%14), 2-2 (%9) | 9.8 / 11 | 4.0 / 14 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Goztepe - Rizespor** | **2-2** | 2.78-0.78 | 1.66 - 1.50 | %42 / %23 / %35 | 2-1 (%9), 1-1 (%10), 1-2 (%8) | 9.9 / 7 | 4.3 / 6 | ⚖️ Kritik Denge / Beraberlik (Draw Variance) |

### 7. Hafta Karşılaşmaları

| Maç | Gerçek Skor | Gerçek xG | Tahmin (λ - μ) | Model 1X2 % | İlk 3 Skor Tahmini | Korner (T/G) | Kart (T/G) | Teşhis |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Galatasaray - Kasimpasa** | **3-1** | 2.40-0.51 | 2.15 - 0.83 | %68 / %18 / %13 | 2-0 (%12), 1-0 (%11), 2-1 (%10) | 10.2 / 9 | 4.2 / 8 | ✅ Baskın İsabet (Dominant Hit) |
| **Genclerbirligi - Amedspor** | **2-2** | 0.50-2.16 | 1.00 - 2.35 | %15 / %17 / %68 | 1-2 (%10), 0-2 (%10), 0-1 (%9) | 10.1 / 9 | 4.4 / 8 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Alanyaspor - Erzurumspor FK** | **0-0** | 0.98-1.69 | 1.46 - 0.58 | %59 / %26 / %15 | 1-0 (%19), 2-0 (%14), 0-0 (%13) | 8.3 / 8 | 3.7 / 6 | ⚠️ Model / Taktiksel Sapma (Model Miss) |
| **Samsunspor - Trabzonspor** | **0-3** | 1.66-2.02 | 1.03 - 2.07 | %18 / %20 / %62 | 1-2 (%10), 0-2 (%10), 0-1 (%10) | 9.6 / 14 | 3.8 / 4 | ✅ Baskın İsabet (Dominant Hit) |
| **Rizespor - Fenerbahce** | **0-3** | 1.52-0.96 | 1.07 - 2.20 | %18 / %19 / %64 | 1-2 (%10), 0-2 (%9), 0-1 (%9) | 10.3 / 7 | 4.1 / 0 | ✅ Baskın İsabet (Dominant Hit) |
| **Konyaspor - Basaksehir** | **0-0** | 0.41-0.97 | 1.46 - 1.32 | %41 / %25 / %34 | 2-1 (%9), 1-1 (%12), 1-0 (%9) | 9.1 / 8 | 4.4 / 0 | ⚖️ Kritik Denge / Beraberlik (Draw Variance) |
| **Gaziantep FK - Corum FK** | **0-3** | 0.99-2.67 | 1.49 - 1.58 | %34 / %28 / %38 | 1-2 (%8), 1-1 (%13), 2-1 (%8) | 9.3 / 14 | 4.3 / 5 | ✅ Baskın İsabet (Dominant Hit) |
| **Besiktas - Kocaelispor** | **2-1** | 2.21-1.65 | 2.20 - 0.56 | %76 / %16 / %8 | 2-0 (%15), 1-0 (%14), 3-0 (%11) | 10.0 / 11 | 3.9 / 3 | ✅ Baskın İsabet (Dominant Hit) |
| **Eyupspor - Goztepe** | **0-4** | 1.39-3.21 | 1.13 - 2.05 | %21 / %20 / %59 | 1-2 (%10), 1-1 (%9), 0-1 (%9) | 9.6 / 10 | 4.3 / 6 | ✅ Baskın İsabet (Dominant Hit) |

### 8. Hafta Karşılaşmaları

| Maç | Gerçek Skor | Gerçek xG | Tahmin (λ - μ) | Model 1X2 % | İlk 3 Skor Tahmini | Korner (T/G) | Kart (T/G) | Teşhis |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Corum FK - Rizespor** | **2-1** | 0.71-0.84 | 2.27 - 1.21 | %62 / %19 / %20 | 2-1 (%10), 1-1 (%8), 2-0 (%8) | 9.9 / 10 | 4.1 / 4 | 🎯 Tam İsabet (Exact Hit) |
| **Erzurumspor FK - Eyupspor** | **2-0** | 1.29-0.43 | 1.54 - 0.55 | %62 / %24 / %13 | 1-0 (%20), 2-0 (%15), 0-0 (%12) | 8.3 / 5 | 3.9 / 1 | 🎯 Tam İsabet (Exact Hit) |
| **Amedspor - Konyaspor** | **1-0** | 1.65-0.25 | 2.11 - 0.83 | %68 / %19 / %14 | 2-0 (%12), 1-0 (%11), 2-1 (%10) | 9.3 / 9 | 4.6 / 4 | 🎯 Tam İsabet (Exact Hit) |
| **Genclerbirligi - Galatasaray** | **0-1** | 1.74-1.92 | 0.92 - 2.68 | %11 / %14 / %75 | 1-2 (%9), 0-2 (%10), 0-3 (%9) | 10.6 / 19 | 4.0 / 1 | ✅ Baskın İsabet (Dominant Hit) |
| **Fenerbahce - Alanyaspor** | **3-0** | 3.84-0.96 | 2.54 - 0.72 | %77 / %14 / %9 | 2-0 (%12), 3-0 (%10), 1-0 (%10) | 9.8 / 7 | 3.8 / 1 | 🎯 Tam İsabet (Exact Hit) |
| **Kasimpasa - Samsunspor** | **4-1** | 2.41-1.07 | 1.47 - 0.95 | %50 / %25 / %25 | 1-0 (%14), 1-1 (%12), 2-0 (%10) | 9.4 / 15 | 4.1 / 3 | ✅ Baskın İsabet (Dominant Hit) |
| **Kocaelispor - Goztepe** | **1-2** | 1.61-0.83 | 1.49 - 1.29 | %42 / %24 / %33 | 2-1 (%9), 1-1 (%11), 1-0 (%10) | 9.2 / 6 | 4.1 / 4 | 🍀 Şans / Bitiricilik Sapması (Finishing Luck) |
| **Basaksehir - Gaziantep FK** | **2-1** | 1.95-1.42 | 2.07 - 0.98 | %63 / %20 / %17 | 2-1 (%10), 2-0 (%10), 1-0 (%10) | 9.3 / 9 | 4.2 / 8 | 🎯 Tam İsabet (Exact Hit) |
| **Trabzonspor - Besiktas** | **2-0** | 2.35-1.19 | 1.87 - 1.49 | %47 / %21 / %31 | 2-1 (%9), 1-1 (%9), 1-2 (%7) | 10.3 / 10 | 4.6 / 5 | ✅ Baskın İsabet (Dominant Hit) |
