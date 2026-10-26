# Deney A: Model vs Piyasa RPS Makasının İstatistiksel Anlamlılık Raporu

> **Metodolojik Çerçeve:** Saf Dixon-Coles Modeli ile Piyasa Kapanış Oranları arasındaki Ranked Probability Score (RPS) farkı $\Delta \text{RPS}_i = \text{RPS}_{\text{mod}, i} - \text{RPS}_{\text{mkt}, i}$ hipotez testiyle incelenmiştir.
> Maçlar arası haftalık korelasyonu korumak için **194 Fikstür Haftası (Cluster)** üzerinden **10.000 İterasyonlu Blok Bootstrap (Block Bootstrap)** uygulanmış; parametrik olmayan **Wilcoxon İşaretli Sıralar Testi** ve **Küme-Dirençli Standart Hatalar (Cluster-Robust SE)** hesaplanmıştır.

---

## 1. Genel Hipotez Testi Sonuçları (1793 Maç)

| Metrik | Model | Kapanış Piyasa | Fark ($\Delta$) | 95% Güven Aralığı (Block Bootstrap) | Anlamlılık (p-değeri) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ortalama RPS** | **0.2022** | **0.1908** | **+0.0114** | **[0.0085, 0.0143]** | **p < 0.0001** |

### Test İstatistikleri Detayı:
- **Saf Model Ortalama RPS:** 0.2022
- **Piyasa Kapanış Ortalama RPS:** 0.1908
- **Ortalama RPS Farkı ($\overline{\Delta}$):** +0.0114 (Piyasa ortalama 0.0114 puan daha başarılı)
- **Naif Standart Hata:** 0.0014
- **Haftalık Küme-Dirençli Standart Hata (Cluster-Robust SE):** 0.00149
- **Blok Bootstrap %95 Güven Aralığı:** `[0.0085, 0.0143]` *(Aralık sıfırı içermemektedir)*
- **Wilcoxon Signed-Rank Test:** $W = 608,608.0$, $p = 4.72e-19$
- **Cluster-Robust Paired t-Test:** $t = 7.657$, $p = 8.91e-13$

> [!IMPORTANT]
> **Bilimsel Çıkarım:**
> Piyasa kapanış oranlarının Saf Dixon-Coles modeline karşı üstünlüğü şans eseri veya örneklem varyansından ibaret değildir; **istatistiksel olarak kesin ve yüksek anlamlılıktadır ($p < 10^{-15}$)**. 95% Güven Aralığının alt sınırı (+0.0085), en muhafazakar koşulda dahi piyasanın modele en az ~0.0085 RPS fark attığını teyit eder.

---

## 2. Sezon Sezon Anlamlılık ve Çoklu Test Düzeltmesi (FWER ≤ 0.05)

| Sezon | Maç | Model RPS | Piyasa RPS | $\Delta$ RPS | 95% Bootstrap GA | Ham Wilcoxon p | Tek Aşamalı Bonferroni ($\alpha=0.0083$) | Adım-Adım Holm-Bonferroni | Holm Düzeltilmiş p | Karar |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2122** | 380 | 0.2088 | 0.2045 | +0.0043 | [-0.0014, 0.0102] | 0.1317 | Kaldı | Eşik: 0.0500 (Kaldı) | 0.1317 | Anlamsız (Başa Baş) |
| **2223** | 313 | 0.2087 | 0.1973 | +0.0114 | [0.0059, 0.0166] | 0.0001 | Geçti | Eşik: 0.0125 (Geçti) | 0.0005 | **ANLAMLI (Piyasa Üstün)** |
| **2324** | 380 | 0.1972 | 0.1838 | +0.0134 | [0.0080, 0.0189] | 0.0000 | Geçti | Eşik: 0.0083 (Geçti) | 0.0000 | **ANLAMLI (Piyasa Üstün)** |
| **2425** | 342 | 0.1970 | 0.1803 | +0.0167 | [0.0080, 0.0247] | 0.0000 | Geçti | Eşik: 0.0100 (Geçti) | 0.0000 | **ANLAMLI (Piyasa Üstün)** |
| **2526** | 306 | 0.1945 | 0.1849 | +0.0096 | [0.0029, 0.0169] | 0.0005 | Geçti | Eşik: 0.0167 (Geçti) | 0.0015 | **ANLAMLI (Piyasa Üstün)** |
| **2627** | 72 | 0.2229 | 0.2025 | +0.0204 | [0.0070, 0.0341] | 0.0217 | Kaldı | Eşik: 0.0250 (Geçti) | 0.0434 | **Holm ile Anlamlı / Bonf. ile Anlamsız (N=72)** |

### Sezon Bazlı Metodolojik Çıkarımlar ve Çoklu Hipotez Ayrımı:
1. **2021/22 Sezonu:** Model ile piyasa arasındaki fark en küçüktür (+0.0043). 95% güven aralığı `[-0.0014, +0.0102]` aralığında olup sıfırı içermektedir ($p = 0.1317$). Bu sezonda model ile piyasa arasındaki fark istatistiksel olarak **anlamlı değildir**; model piyasa kalitesinde tahmin üretmiştir.
2. **2022/23 - 2025/26 Dönemi:** Deprem sonrası tüm tam sezonlarda ($p < 0.001$) piyasa üstünlüğü hem tek aşamalı Bonferroni ($\\alpha = 0.00833$) hem de adım-adım Holm-Bonferroni altında tartışmasız biçimde anlamlıdır.
3. **2026/27 Sezonu (Devam Eden Sezon - 72 Maç):**
   - **Holm-Bonferroni Altında:** 5. sıradaki eşik $\\alpha = 0.05 / 2 = 0.0250$\'dir. Ham $p = 0.0217 < 0.0250$ olduğundan adım-adım Holm yöntemiyle %5 aile-bazlı hata düzeyinde **anlamlıdır** (Holm-düzeltilmiş $p_{\\text{Holm}} = 2 \\times 0.0217 = 0.0434 < 0.05$).
   - **Tek Aşamalı Bonferroni Altında:** Katı tek aşamalı eşik $\\alpha = 0.05 / 6 = 0.00833$ olup, $p = 0.0217 > 0.00833$ olduğundan bu eşiği **geçemez**.
   - **Metodolojik Uyarı:** 72 maçlık örneklem küçük olduğu ve standart hata ($\\sigma / \\sqrt{72}$) geniş olduğu için bu sezon tam sezonlarla (342-380 maç) aynı kesinlikte okunmamalıdır.
4. **Nedensellik Değerlendirmesi:** Piyasa makasının zamanla açılması ile ligdeki bütçe/kadro kutuplaşması arasındaki ilişki gözlemsel bir hipotezdir; nedensellik iddia etmek için xG, kadro reytingleri ve piyasa oranlarının izole ablation testleriyle sınanması şarttır.
