# 2022/23 Sezonu Deprem Kaynaklı Hükmen (Forfeit) Maçlar Denetim Listesi
## (Veri Kaynağı: `data/warehouse.duckdb`, Toplam 29 Karşılaşma)

6 Şubat 2023 Kahramanmaraş depremleri sonrasında Atakaş Hatayspor ve Gaziantep FK ligden çekilmiştir.
Türkiye Futbol Federasyonu (TFF) Yönetim Kurulu kararıyla, bu iki takımın 23. haftadan itibaren fikstürde yer alan ve oynanmayan tüm karşılaşmaları 3-0 / 0-3 hükmen tescil edilmiştir.

Aşağıdaki tablo, `data/warehouse.duckdb` ambarındaki `matches` tablosunda `season = '2223'` ve `status = 'forfeit'` olarak filtrelenen **tam 29 resmi karşılaşmanın** dökümüdür:

| No | Maç ID | Hafta (Matchday) | Tarih (UTC) | Ev Sahibi | Deplasman | Tescil Skoru | Durum | Gerekçe & Detay |
| :-: | :-: | :-: | :-: | :--- | :--- | :-: | :-: | :--- |
| 1 | `1922` | 23. Hafta | 2023-02-25 | **Umraniyespor** | **Gaziantep FK** | 3 - 0 | `forfeit` | TFF Hükmen Tescil |
| 2 | `1931` | 24. Hafta | 2023-03-05 | **Adana Demirspor** | **Hatayspor** | 3 - 0 | `forfeit` | TFF Hükmen Tescil |
| 3 | `1932` | 24. Hafta | 2023-03-05 | **Gaziantep FK** | **Galatasaray** | 0 - 3 | `forfeit` | TFF Hükmen Tescil |
| 4 | `1940` | 25. Hafta | 2023-03-12 | **Hatayspor** | **Konyaspor** | 0 - 3 | `forfeit` | TFF Hükmen Tescil |
| 5 | `1951` | 26. Hafta | 2023-03-19 | **Gaziantep FK** | **Trabzonspor** | 0 - 3 | `forfeit` | TFF Hükmen Tescil |
| 6 | `1952` | 26. Hafta | 2023-03-19 | **Kayserispor** | **Hatayspor** | 3 - 0 | `forfeit` | TFF Hükmen Tescil |
| 7 | `1958` | 27. Hafta | 2023-04-02 | **Gaziantep FK** | **Kasimpasa** | 0 - 3 | `forfeit` | TFF Hükmen Tescil |
| 8 | `1959` | 27. Hafta | 2023-04-02 | **Hatayspor** | **Sivasspor** | 0 - 3 | `forfeit` | TFF Hükmen Tescil |
| 9 | `1966` | 28. Hafta | 2023-04-09 | **Adana Demirspor** | **Gaziantep FK** | 3 - 0 | `forfeit` | TFF Hükmen Tescil |
| 10 | `1967` | 28. Hafta | 2023-04-09 | **Alanyaspor** | **Hatayspor** | 3 - 0 | `forfeit` | TFF Hükmen Tescil |
| 11 | `1977` | 29. Hafta | 2023-04-16 | **Gaziantep FK** | **Konyaspor** | 0 - 3 | `forfeit` | TFF Hükmen Tescil |
| 12 | `1978` | 29. Hafta | 2023-04-16 | **Hatayspor** | **Fatih Karagumruk** | 0 - 3 | `forfeit` | TFF Hükmen Tescil |
| 13 | `1984` | 30. Hafta | 2023-04-19 | **Besiktas** | **Hatayspor** | 3 - 0 | `forfeit` | TFF Hükmen Tescil |
| 14 | `1993` | 31. Hafta | 2023-04-23 | **Gaziantep FK** | **Alanyaspor** | 0 - 3 | `forfeit` | TFF Hükmen Tescil |
| 15 | `1994` | 31. Hafta | 2023-04-23 | **Hatayspor** | **Ankaragucu** | 0 - 3 | `forfeit` | TFF Hükmen Tescil |
| 16 | `2004` | 32. Hafta | 2023-04-30 | **Basaksehir** | **Hatayspor** | 3 - 0 | `forfeit` | TFF Hükmen Tescil |
| 17 | `2005` | 32. Hafta | 2023-04-30 | **Kayserispor** | **Gaziantep FK** | 3 - 0 | `forfeit` | TFF Hükmen Tescil |
| 18 | `2012` | 33. Hafta | 2023-05-07 | **Gaziantep FK** | **Fatih Karagumruk** | 0 - 3 | `forfeit` | TFF Hükmen Tescil |
| 19 | `2013` | 33. Hafta | 2023-05-07 | **Hatayspor** | **Istanbulspor** | 0 - 3 | `forfeit` | TFF Hükmen Tescil |
| 20 | `2021` | 34. Hafta | 2023-05-17 | **Besiktas** | **Gaziantep FK** | 3 - 0 | `forfeit` | TFF Hükmen Tescil |
| 21 | `2022` | 34. Hafta | 2023-05-17 | **Giresunspor** | **Hatayspor** | 3 - 0 | `forfeit` | TFF Hükmen Tescil |
| 22 | `2031` | 35. Hafta | 2023-05-21 | **Gaziantep FK** | **Giresunspor** | 0 - 3 | `forfeit` | TFF Hükmen Tescil |
| 23 | `2032` | 35. Hafta | 2023-05-21 | **Hatayspor** | **Fenerbahce** | 0 - 3 | `forfeit` | TFF Hükmen Tescil |
| 24 | `2036` | 36. Hafta | 2023-05-30 | **Basaksehir** | **Gaziantep FK** | 3 - 0 | `forfeit` | TFF Hükmen Tescil |
| 25 | `2039` | 36. Hafta | 2023-05-30 | **Hatayspor** | **Umraniyespor** | 0 - 3 | `forfeit` | TFF Hükmen Tescil |
| 26 | `2045` | 37. Hafta | 2023-06-03 | **Antalyaspor** | **Hatayspor** | 3 - 0 | `forfeit` | TFF Hükmen Tescil |
| 27 | `2046` | 37. Hafta | 2023-06-03 | **Gaziantep FK** | **Istanbulspor** | 0 - 3 | `forfeit` | TFF Hükmen Tescil |
| 28 | `2057` | 38. Hafta | 2023-06-07 | **Fenerbahce** | **Gaziantep FK** | 3 - 0 | `forfeit` | TFF Hükmen Tescil |
| 29 | `2059` | 38. Hafta | 2023-06-07 | **Hatayspor** | **Galatasaray** | 0 - 3 | `forfeit` | TFF Hükmen Tescil |

---

## Sezon Fikstür ve Takım Dağılımı Özeti:
- **Resmi Takım Sayısı:** 19 Kulüp (2022/23 Süper Lig sezonu 19 takımla oynanmıştır)
- **Takım Başına Planlanan Maç:** 36 Karşılaşma (18 İç Saha + 18 Deplasman)
- **Fikstürdeki Toplam Planlanan Maç Sayısı:** $19 \times 18 = \mathbf{342 \text{ Maç}}$
- **Hükmen Tescil Edilen Maç Sayısı:** $\mathbf{29 \text{ Maç}}$
  - **Hatayspor:** 15 Maç (Tüm 15 maç diğer 17 rakiple oynanmamış fikstürdür)
  - **Gaziantep FK:** 14 Maç (Tüm 14 maç diğer 17 rakiple oynanmamış fikstürdür)
  - **Toplam:** $15 + 14 = \mathbf{29 \text{ Maç}}$
- **Hatayspor ile Gaziantep FK Karşılıklı Maç Durumu:**
  - İki kulüp arasındaki 2 lig maçı da depremden (6 Şubat 2023) önce **sahada fiilen oynanmış ve tamamlanmıştır**:
    1. **3. Hafta (2022-08-20):** Hatayspor 1 - 2 Gaziantep FK (Maç ID: `1739`, `played`)
    2. **22. Hafta (2023-02-01):** Gaziantep FK 4 - 1 Hatayspor (Maç ID: `1909`, `played`)
  - Dolayısıyla 23. haftadan itibaren Hatayspor ile Gaziantep FK arasında oynanmamış karşılıklı randevu bulunmamaktadır. 19 takımlı ligde bay (bye) haftaları nedeniyle Hatayspor'un 15, Gaziantep FK'nın 14 maçı kalmış ve $15 + 14 = 29$ hükmen maç oluşmuştur.
- **Fiilen Sahada Oynanmış Resmi Maç Sayısı:** $342 - 29 = \mathbf{313 \text{ Maç}}$
- **Dışarıda Bırakılan Başka Karşılaşma Var mıdır?:** Hayır. Oynanmış 313 maçın %100'ü modelin out-of-sample test havuzunda yer almaktadır.
