# Deney C: 4 Farklı Tahmin Modelinin Out-of-Sample Karşılaştırma Raporu

> **Metodolojik Çerçeve:** Bu deneyde piyasa oranları ile istatistiksel modelin en verimli nasıl birleştirilebileceğini belirlemek amacıyla 4 bağımsız mimari tamamen out-of-sample (geçmiş pencereler üzerinden katı zamansal sırayla) test edilmiştir:
> 1. **Model A (Saf Kapanış Piyasası):** Shin algoritmasıyla vigorish'ten arındırılmış saf piyasa olasılıkları.
> 2. **Model B (Saf Dixon-Coles):** Hiçbir piyasa verisi almayan saf matematiksel maç motoru.
> 3. **Model C (Optimal Dinamik Harman):** $P_{\text{blend}} = w \cdot P_{\text{mod}} + (1-w) \cdot P_{\text{mkt}}$. $w$ parametresi geçmiş 200 maçta RPS minimize edilerek katı walk-forward ile öğrenilmiştir (Ortalama $w = 0.056$).
> 4. **Model D (Piyasa + Bağlamsal Artık Düzeltmesi):** Piyasa baz alınarak modelin yüksek güven sapmalarıyla düzeltilen artık (residual) mimari.

---

## 1. Genel Model Karşılaştırma Karnesi (1793 Maç)

| Model Mimarisi | RPS (Yuvarlanmış) | RPS (Yuvarlanmamış) | 1X2 İsabeti | Brier Skoru | Piyasaya Göre RPS $\Delta$ | Durum / Karar |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model A: Saf Kapanış Piyasası** | **0.1908** | **0.190786** | **%55.8** | **0.565695** | Baz | **ANA REFERANS (BENCHMARK)** |
| **Model B: Saf Dixon-Coles** | **0.2022** | **0.20217** | **%53.4** | **0.590566** | +0.011384 | Bağımsız Temel Model |
| **Model C: Dinamik Optimal Harman** | **0.1908** | **0.190838** | **%55.9** | **0.565845** | +0.000052 | **DENEYSEL ALTERNATİF (Üstünlük Kanıtlanmadı)** |
| **Model D: Artık (Residual) Düzeltme** | **0.1916** | **0.19157** | **%55.8** | **0.56741** | +0.000784 | **REDDEDİLDİ (Piyasadan Kötü)** |

---

## 2. Model C (Harman) vs Model A (Piyasa) Eşleştirilmiş Test Detayı

| Metrik | Değer | Açıklama |
| :--- | :---: | :--- |
| **Yuvarlanmamış Ortalama Kayıp Farkı ($\\Delta \\text{RPS} = C - A$)** | **+0.000052** | Model C saf piyasadan 0.000052 puan daha yüksek (daha kötü) RPS üretmektedir ($0.190838 - 0.190786$). |
| **%95 Bootstrap Güven Aralığı** | **`[-0.000255, +0.000354]`** | **Sıfırı içermektedir.** Üstünlük veya gerilik istatistiksel olarak kanıtlanamaz. |
| **Eşleştirilmiş Paired t-Testi** | $t = 0.3343, p = 0.7382$ | **$p = 0.7382$ (Anlamsız)**. İki tahminin performansı istatistiksel olarak farksızdır. |
| **1X2 İsabet Farkı** | +%0.1 (55.8% -> 55.9%) | Yalnızca 1-2 maçlık örneklem gürültüsüdür; olasılık kaybını iyileştirmemektedir. |
ightarrow$ 55.9%) | Yalnızca 1-2 maçlık örneklem gürültüsüdür; olasılık kaybını iyileştirmemektedir. |

---

## 3. Sezon Bazında RPS Karşılaştırması

| Sezon | Maç | Model A (Piyasa) | Model B (Dixon-Coles) | Model C (Harman) | Model D (Artık) | En İyi Model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2122** | 380 | 0.2045 | 0.2088 | 0.2047 | 0.2048 | **A (Piyasa)** |
| **2223** | 313 | 0.1973 | 0.2087 | 0.1973 | 0.1981 | **A (Piyasa)** |
| **2324** | 380 | 0.1838 | 0.1972 | 0.1838 | 0.1847 | **A (Piyasa)** |
| **2425** | 342 | 0.1803 | 0.1970 | 0.1803 | 0.1814 | **A (Piyasa)** |
| **2526** | 306 | 0.1849 | 0.1945 | 0.1849 | 0.1855 | **A (Piyasa)** |
| **2627** | 72 | 0.2025 | 0.2229 | 0.2025 | 0.2041 | **A (Piyasa)** |

---

## 4. Teknik Hakem Kararı ve Mimari Sonuç

2. **Model C Deneysel Alternatif Olarak Tutulmalıdır:** Model C'nin ortalama ağırlığı $w \\approx 0.056$ çıkmıştır (%5.6 model, %94.4 piyasa). Yuvarlanmamış RPS (0.190838) ve Brier (0.565845) sonuçları Model A'dan daha iyi değildir ve paired $t$-testi ($p = 0.7382$) farkın anlamsız olduğunu kanıtlamaktadır. Daha fazla doğrulama yapılana kadar üretime alınmamalıdır.
2. **Model C Deneysel Alternatif Olarak Tutulmalıdır:** Model C'nin ortalama ağırlığı $w \approx 0.056$ çıkmıştır (%5.6 model, %94.4 piyasa). Yuvarlanmamış RPS ve Brier sonuçları Model A'dan daha iyi değildir ve paired $t$-testi ($p = 0.4997$) farkın anlamsız olduğunu kanıtlamaktadır. Daha fazla doğrulama yapılana kadar üretime alınmamalıdır.
3. **Model D Kesin Olarak Reddedilmiştir:** Raporlanan RPS (0.19157) ve Brier (0.56741) saf piyasadan açıkça daha kötüdür.
