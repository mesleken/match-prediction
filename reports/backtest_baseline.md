# Faz 1 & 2 — Geliştirilmiş Matematiksel Model ve Walk-Forward Backtest Raporu

**Değerlendirilen Sezonlar:** 1819 – 2627 (2026/27 canlı sezon dahil)  
**Toplam Maç Sayısı:** 2825  

## Referans Merdiveni (Benchmark Comparison)

| Model Seviyesi | RPS (Düşük = İyi) | Fark vs Naif | Skor Log-Kaybı | Not |
|---|---|---|---|---|
| **B0: Naif Baz (Tarihsel Frekans)** | 0.2272 | 0.0000 | - | Geçmiş maç dağılımı |
| **B3: Geliştirilmiş Dixon-Coles (Katman A+E)** | 0.2056 | -0.0216 | 2.9511 | Şut & xG Hibrit + Laplace + EB Önsel |
| **B4: Piyasa Kapanış (Shin Devig)** | 0.1966 | - | - | Bahis bürosu kapanış konsensüsü |
| **Katman F: Çoklu-Piyasa Harmanı (1X2 + O/U)** | 0.1971 | - | 2.9101 | İki ağırlıklı zımni harman (u, v) |

## İstatistiksel Güç ve Kalibrasyon Metrikleri

- **Kapanış Oranlı Maç Sayısı (N):** 2825
- **Brier Skoru (DC):** 0.6024
- **Beklenen Kalibrasyon Hatası (ECE):** 0.0332
- **Eşleştirilmiş Fark Standart Sapması ($s_d$):** 0.0574
- **Saptanabilir Minimum Etki (MDE, %80 güç):** 0.0030
- **Harman vs Piyasa Farkı:** +0.00053

## Sezon Bazlı Performans Dağılımı (2018/19 – 2026/27)

| Sezon | Maç Sayısı | Naif RPS | Saf Model RPS | Saf Skor Log-Loss | Piyasa Kapanış RPS | Harmanlanmış RPS | Harman Skor Log-Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| **2018/19 (`1819`)** | 306 | 0.2214 | **0.2076** | 2.8887 | 0.2043 | **0.2043** | 2.8678 |
| **2019/20 (`1920`)** | 306 | 0.2278 | **0.2139** | 2.9665 | 0.2070 | **0.2074** | 2.9243 |
| **2020/21 (`2021`)** | 420 | 0.2289 | **0.2111** | 2.9494 | 0.2082 | **0.2083** | 2.9284 |
| **2021/22 (`2122`)** | 380 | 0.2278 | **0.2124** | 2.9517 | 0.2047 | **0.2050** | 2.9078 |
| **2022/23 (`2223`)** | 313 | 0.2324 | **0.2067** | 3.0754 | 0.1972 | **0.1979** | 3.0338 |
| **2023/24 (`2324`)** | 380 | 0.2253 | **0.1963** | 2.9303 | 0.1837 | **0.1847** | 2.8778 |
| **2024/25 (`2425`)** | 342 | 0.2265 | **0.1980** | 2.9867 | 0.1803 | **0.1812** | 2.9291 |
| **2025/26 (`2526`)** | 306 | 0.2250 | **0.1941** | 2.8254 | 0.1850 | **0.1853** | 2.7843 |
| **2026/27 (`2627`)** | 72 | 0.2347 | **0.2214** | 3.0921 | 0.2029 | **0.2049** | 3.0109 |