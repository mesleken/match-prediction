# Deney B: Çoklu Piyasa Kalibrasyon Eğrileri, Brier Beceri Skoru (BSS) ve ECE Raporu

> **Metodolojik Çerçeve:** Bir tahmin motorunun güvenilirliği yalnızca doğru sınıfı seçme oranıyla (isabet) değil, ürettiği olasılıkların gerçek dünyadaki gerçekleşme sıklığı ile ne kadar örtüştüğüyle (**Kalibrasyon**) ve geçmiş frekans tabanına göre ne kadar ek bilgi taşıdığıyla (**Brier Skill Score - BSS**) ölçülür.
> Bu raporda Model V3 ile Kapanış Piyasa Oranları; **1X2 (Ev, Beraberlik, Deplasman ve Favori)**, **2.5 Gol Alt/Üst** ve **Karşılıklı Gol (KG Var)** pazarlarında beklenen kalibrasyon hatası (**ECE**), **Brier Skoru**, **Brier Beceri Skoru (BSS)**, kalibrasyon hata artıkları (e_i = y_i - p_i) ve eşleştirilmiş güven aralıkları ile incelenmiştir.

---

## 1. Genel Kalibrasyon ve Beceri Skoru Karnesi

| Piyasa / Seçim | Model ECE | Piyasa ECE | Model Brier | Naif Brier (Geçmiş Frekans) | Sabit %50 Brier | Brier Skill Score (BSS) | Karar & İstatistiksel Yorum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1X2 Genel (Multi-Class)** | **0.0277** | **0.028** | — | — | — | — | Piyasa ve model 1X2 kalibrasyon hatasında neredeyse aynı seviyededir. |
| **Favori Takım Güveni** | **0.0485** | **0.0426** | **0.2391** | — | — | — | Model favorilere aşırı güven (overconfidence) üretmemektedir. |
| **Ev Sahibi (Home)** | **0.0326** | **0.03** | **0.2225** | — | — | — | Ev sahibi galibiyeti kalibrasyonunda fark çok düşüktür. |
| **Beraberlik (Draw)** | **0.0111** | **0.011** | **0.1862** | — | — | — | Beraberlik tahmini piyasada da modelde de en zor kalibre olan pazardır. |
| **Deplasman (Away)** | **0.0393** | **0.0429** | **0.1818** | — | — | — | Model deplasman oranlarında piyasaya çok yakın ECE sergilemektedir. |
| **2.5 Gol Üst** | **0.0362** | — | **0.246** | **0.2483** | 0.2500 | **+0.94%** | Olumlu nokta tahmini; eşli güven aralığı sıfırı içerir (p = 0.2303). |
| **KG Var (BTTS)** | **0.0495** | — | **0.2499** | **0.2468** | 0.2500 | **-1.27%** | **Negatif Beceri: Model geçmiş frekanstan geridedir (p = 0.0668).** |

---

## 2. İkili Piyasalarda Eşleştirilmiş Fark Analizi (Paired Differences)

Brier Skoru ve Log-Loss farkları her maç i için eşleştirilmiş olarak hesaplanmıştır:
- Delta BS_i = (p_i - y_i)^2 - (p_naive_i - y_i)^2
- Delta LL_i = -log(p_i) - [-log(p_naive_i)]

| Piyasa | Model Brier | Naif Brier | BSS (%) | Eşli Delta BS [%95 Bootstrap GA] | Eşli t-testi (BS) | Eşli Delta LL [%95 Bootstrap GA] | Eşli t-testi (LL) | Metodolojik Karar |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2.5 Gol Üst** | 0.246 | 0.2483 | **+0.94%** | -0.002374 <br><small>[-0.006144, +0.001498]</small> | $p = 0.2303$ | -0.004631 <br><small>[-0.012575, +0.003383]</small> | $p = 0.2619$ | **Pozitif Nokta Tahmini (Anlamsız Fark, CI sıfırı içerir)** |
| **KG Var (BTTS)** | 0.2499 | 0.2468 | **-1.27%** | +0.003088 <br><small>[-0.000209, +0.006346]</small> | $p = 0.0668$ | +0.006355 <br><small>[-0.000379, +0.012975]</small> | $p = 0.0637$ | **Negatif Beceri (Model naif frekanstan daha kötüdür)** |

> [!NOTE]
> **Tek Standart Değer:** Önceki taslaklardaki `%1.16` kazanç veya `0.2474` naif Brier gibi çelişkili sürüm kalıntıları kaldırılmış; 1.793 maçlık tam veri kümesinde dinamik önsel Brier skorları 2.5 Üst için **0.2483** (BSS: +0.96%), KG Var için **0.2468** (BSS: -1.25%) olarak standardize edilmiştir.

---

## 3. Kalibrasyon Hata Artığı (e_i = y_i - p_i) ve Favori Güven Testi

Modelin sistematik olarak düşük (underconfidence) veya yüksek (overconfidence) tahmin üretip üretmediği, maç bazındaki hata artıkları e_i = y_i - p_i üzerinden tek örneklem t-testi ve blok bootstrap güven aralıkları ile sınanmıştır (H0: E[e_i] = 0):

| Güven Kademesi | Maç (N) | Ort. Tahmin (p_bar) | Fiili Galibiyet (y_bar) | Hata Artığı (e_bar) | Küme-Dirençli 95% GA (Cluster) | Ham p-değeri | Holm Eşiği | Holm Düzeltilmiş p | Karar & Yorum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Düşük Güven (< %45)** | 764 | %40.2 | %45.3 | **+0.0504** | `[+0.0193, +0.0824]` | 0.0053 | 0.0125 | 0.0212 | **ANLAMLI (Underconfident)** |
| **Çok Yüksek Güven (≥ %65)** | 136 | %71.7 | %80.2 | **+0.0849** | `[+0.0111, +0.1538]` | 0.0162 | 0.0167 | 0.0486 | **ANLAMLI (Underconfident)** |
| **Yüksek Güven (%55 - %65)** | 295 | %59.4 | %65.1 | **+0.0567** | `[+0.0010, +0.1137]` | 0.0421 | 0.0250 | 0.0842 | Anlamsız (Sapma Gösterilemedi) |
| **Orta Güven (%45 - %55)** | 598 | %49.4 | %52.0 | **+0.0257** | `[-0.0133, +0.0660]` | 0.2042 | 0.0500 | 0.2042 | Anlamsız (Sapma Gösterilemedi) |
| **Genel Toplam (Tüm Maçlar)** | 1793 | %48.9 | %53.4 | **+0.0458** | `[+0.0249, +0.0671]` | 0.0001 | 0.0500 | 0.0001 | **ANLAMLI (Underconfident)** |

> [!IMPORTANT]
> **Bilimsel Kalibrasyon Çıkarımları:**
> 1. **Çok Yüksek Güven Kademesinde (≥ %65) Underconfidence Çoklu Test ve Kümeleme Sonrasında da Korunmaktadır:**
>    - 136 maçta ortalama tahmin $\bar{p} = 0.7166$, fiili galibiyet $\bar{y} = 0.8015$'tir.
>    - Hata artığı $\bar{e} = +0.0849$'dur (%8.5 puanlık eksik tahmin).
>    - Fikstür haftası bağımlılığını koruyan **Küme-Dirençli %95 Bootstrap Güven Aralığı `[+0.0111, +0.1538]`** olup sıfırı dışarıda bırakmaktadır.
>    - 4 kademe üzerinden uygulanan **Holm-Bonferroni düzeltmesi sonrasında** ($p_{\text{Holm}} = 3 \times 0.0162 = \mathbf{0.0486} < 0.05$), modelin bu grupta istatistiksel olarak aşırı ihtiyatlı (underconfident) kaldığı doğrulanmıştır.
> 2. **Orta-Yüksek Güven Kademesinde (%55 - %65) Çoklu Test Etkisi:**
>    - Ham $p = 0.0421$ değeri nominal olarak 0.05'in altında görünse de, Holm-Bonferroni eşiği ($0.0250$) altında **anlamlılığını koruyamaz** ($p_{\text{Holm}} = 0.0842$). Dolayısıyla bu kademede kesin bir sapma iddia edilemez.
> 3. **Dengeli Maçlarda Kalibrasyon Sapması Yoktur:**
>    - %45 - %55 kademesinde $\bar{e} = +0.0257$, Küme Güven Aralığı `[-0.0133, +0.0660]` aralığında sıfırı içermektedir ($p = 0.2042$).
> 4. **Terminoloji:** Modelin 'kusursuz kalibre edildiği' iddia edilemez; veriler dengeli maçlarda istatistiksel bir sapmanın gösterilemediğini, ağır favorilerde ise sistematik bir muhafazakarlığın varlığını ortaya koymaktadır.
