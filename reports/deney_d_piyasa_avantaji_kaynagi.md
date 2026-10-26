# Deney D: Piyasa Bilgi Avantajının Zamansal Kaynağı Analiz Raporu

> **Kritik Araştırma Sorusu:** Piyasa modelden daha başarılıysa, bu üstünlük **ne zaman** ve **nasıl** oluşmaktadır?
> Bahis şirketlerinin ilk açtığı oranlar (**Açılış Oranları - Opening Odds**, maçtan 5-7 gün önce) saf istatistiksel modeller ve erken beklentilerle belirlenir.
> Maç saatine yaklaşıldıkça (**Kapanış Oranları - Closing Odds**) ilk 11 kadroları, taktiksel sızıntılar, hava durumu ve profesyonel fonların (sharp money) likiditesi piyasaya girer.
> Bu raporda **1544 maç** üzerinde hem açılış hem kapanış oranları izole edilerek model ile yan yana test edilmiştir.

---

## 1. Açılış vs Kapanış vs Model Genel Karnesi (1544 Maç)

| Zaman Dilimi / Model | Ortalama RPS | 1X2 İsabeti | Brier Skoru | Açıklama |
| :--- | :---: | :---: | :---: | :--- |
| **Saf Dixon-Coles Modeli** | **0.2011** | **%53.8** | **0.5884** | Temel takım dinamiklerine dayalı istatistiksel tahmin. |
| **Piyasa Açılış Oranları (Opening)** | **0.1917** | **%55.1** | **0.5678** | Maçtan günler önce piyasaya sunulan ham piyasa tahmini. |
| **Piyasa Kapanış Oranları (Closing)** | **0.1901** | **%55.9** | **0.564** | Tüm son dakika bilgilerini absorbe etmiş nihai piyasa. |

---

## 2. Açılış $\rightarrow$ Kapanış Eşleştirilmiş Kayma (Drift) İstatistiği

| Metrik | Değer | İstatistiksel Değerlendirme |
| :--- | :---: | :--- |
| **Ortalama Kayma (Drift = Open - Close)** | **++0.001620 RPS** | Kapanış oranları açılışa göre ortalama 0.001620 RPS daha keskindir. |
| **%95 Bootstrap Güven Aralığı** | **`[+0.000175, +0.003104]`** | Sıfırı içermemektedir (Pozitif yönde kesin kayma). |
| **Paired t-Testi** | $t = 2.1755, p = 0.0297$ | **$p = 0.0297$ (Anlamlı)**. |
| **Wilcoxon Signed-Rank Testi** | $W = 542654.0, p = 0.0025$ | **$p = 0.0025$ (Yüksek Anlamlılık)**. |

---

## 3. Sezon Sezon Açılış vs Kapanış vs Model Dağılımı

| Sezon | Maç | Model RPS | Açılış RPS | Kapanış RPS | Model vs Açılış Farkı | Açılış $\rightarrow$ Kapanış İlerlemesi |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2122** | 380 | 0.2088 | 0.2059 | 0.2045 | +0.0029 | -0.0015 |
| **2223** | 313 | 0.2087 | 0.1993 | 0.1973 | +0.0093 | -0.0020 |
| **2324** | 379 | 0.1973 | 0.1858 | 0.1840 | +0.0115 | -0.0018 |
| **2425** | 340 | 0.1966 | 0.1808 | 0.1798 | +0.0158 | -0.0010 |
| **2526** | 132 | 0.1837 | 0.1773 | 0.1753 | +0.0064 | -0.0020 |

---

## 4. Stratejik ve Bilimsel Çıkarımlar

1. **Piyasanın Zaman İçinde Keskinleşmesi Doğrulanmıştır ($p = 0.0025$):**
   - Açılıştan kapanışa yaşanan ++0.001620 RPS'lik iyileşme eşleştirilmiş Wilcoxon testinde $p = 0.0025$ ile istatistiksel olarak anlamlıdır.
2. **Nedensellik Sınırı ve Hipotez Niteliği:**
   - Kapanış oranlarının açılıştan daha iyi olması verilerle kanıtlanmıştır; ancak bunun tam olarak kadro haberlerinden mi, taktik sızıntılardan mı yoksa genel piyasa likiditesi ve profesyonel arbitrajdan mı kaynaklandığı bu veri kesitinden doğrudan çıkarılamaz. Bu bir çalışma hipotezidir.
   - Kesin nedensellik tespiti için maç saatine 24 saat ve 1 saat kala resmi kadro anonsu anındaki timestamped oran hareketlerinin izole test edilmesi gerekir.
3. **Model V3 İçin Çıkarım:**
   - Model ile piyasanın açılış oranları arasındaki makas (+0.0094) kapanışa göre daha dardır. Maç günü ilk 11 ve sakatlık girdilerini dinamik olarak beslemek, bu bilgi kaybını en aza indirmenin en rasyonel yoludur.
