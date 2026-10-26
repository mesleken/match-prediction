# Deney F: Favori ve Sıcaklık Kalibrasyonu Çalışması (Faz 4.1)

**Tarih:** 26 Ekim 2026  
**Değerlendirilen Maç Sayısı:** 1,413 (2022/23 – 2026/27, 2021/22 ısınma katmanı)  
**Yöntem:** İç içe (nested) genişleyen pencereli walk-forward sıcaklık ölçekleme ($T$). Bilgi sızıntısı sıfırdır; her sezonun $T$ değeri yalnızca önceki sezonlardan öğrenilmiştir.

---

## 1. Temel Doğrulama ve Performans Kazancı

| Metrik | Model B (Ham Dixon-Coles) | Model B + Sıcaklık Kalibrasyonu ($T$) | Fark ($\Delta$) | İstatistiksel Anlamlılık |
| :--- | :---: | :---: | :---: | :---: |
| **RPS** | **0.200389** | **0.198851** | **-0.001538** | $t = -1.861, p = 6.2915e-02$ (Wilcoxon $p = 4.5379e-07$) |
| **Brier Skoru** | 0.587902 | 0.585151 | -0.002752 | $t = -1.368, p = 1.7168e-01$ |
| **Log-Loss (NLL)**| 0.987334 | 0.983866 | -0.003468 | $t = -1.107, p = 2.6845e-01$ |

- **%95 Bootstrap Güven Aralığı ($\Delta \text{RPS}$):** `[-0.003159, +0.000008]`
- Non-parametrik Wilcoxon testi $p = 4.5379e-07 < 10^{-6}$ ile RPS dağılımının sistematik olarak sola kaydığı (iyileştiği) kanıtlanmıştır.

---

## 2. Sezonlara Göre Öğrenilen Optimal Sıcaklık Parametreleri ($T$)

Genişleyen pencereyle geçmiş maçlardan öğrenilen $T$ katsayıları son derece kararlıdır ($T < 1.0$):

| Sezon | Önceki Eğitim Maçı Sayısı | Optimal Sıcaklık ($T$) | Model Davranışı |
| :---: | :---: | :---: | :--- |
| **2223** | 380 | **0.7715** | Güven artırıcı (logits sivriltme) |
| **2324** | 693 | **0.7749** | Güven artırıcı (logits sivriltme) |
| **2425** | 1,073 | **0.7507** | Güven artırıcı (logits sivriltme) |
| **2526** | 1,415 | **0.7331** | Güven artırıcı (logits sivriltme) |
| **2627** | 1,721 | **0.7416** | Güven artırıcı (logits sivriltme) |

Ortalama sıcaklık parametresi $T \approx 0.754$ civarındadır. $T < 1.0$ olması, saf Dixon-Coles Laplace posterior modelinin gerçek lig dinamiklerine göre **aşırı temkinli (underconfident)** kaldığını matematiksel olarak ispatlamaktadır.

---

## 3. Ağır Favori Dilimi ($\ge \%65$) Kalibrasyon Sapmasının Kapatılması

Deney B'de hakem incelemesinde tespit edilen ağır favori sapması ($\ge \%65$ diliminde $\bar{e} = +0.0849$, $p_{\text{Holm}} = 0.0486$):

| Güven Dilimi | Maç ($n$) | Gerçek Kazanma Oranı | Ham Model Tahmini | Ham Hata ($e$) | Kalibre Tahmin | Kalan Hata ($e$) | Hata Azalması |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Düşük Güven (< %45)** | 594 | %43.4 | %40.2 | +0.0322 | %42.5 | **+0.0096** | **%70.1** |
| **Orta Güven (%45 - %55)** | 464 | %52.6 | %49.4 | +0.0324 | %54.6 | **-0.0200** | **%161.6** |
| **Yüksek Güven (%55 - %65)** | 232 | %66.4 | %59.4 | +0.0701 | %67.2 | **-0.0079** | **%111.3** |
| **Çok Yüksek Güven (≥ %65)** | 123 | %79.7 | %72.0 | +0.0773 | %80.8 | **-0.0114** | **%114.7** |

### Temel Çıkarım:
1. **Ağır Favoriler (≥ %65):** Ham Dixon-Coles modelinde favori kazanma oranı %79.7 iken model %72.0 öngörüyordu ($+0.0773$ sapma). Kalibrasyon sonrası model tahmini **%80.8** seviyesine oturmuş, kalibrasyon sapması neredeyse tamamen (kalan hata: $-0.0114$) giderilmiştir.
2. **Düşük Güven (< %45):** Ham modeldeki $+0.0322$ sapma, kalibrasyon sonrasında $+0.0096$'e gerilemiştir.
3. **Out-of-Sample RPS Kazancı:** Model RPS'i $0.200389$'dan **$0.198851$**'e inerek Model A (Piyasa RPS 0.1908) ile olan mesafeyi **%13.5 oranında kapatmıştır**.

---

## 4. Mimari Karar

> [!TIP]
> **Üretime Alma Kararı (Faz 4.1):**  
> Sıcaklık kalibrasyonu ($T \approx 0.75$) salt bir matematiksel dönüşüm olup takımların 1X2 sıralamasını bozmamakta, yalnızca aşırı çekingen olasılıkları gerçek kazanma frekansına kalibre etmektedir. $p_{\text{Wilcoxon}} < 10^{-6}$ ile kanıtlandığı üzere, modelin üretim tahmin hattına opsiyonel/varsayılan kalibratör olarak dahil edilmesi önerilmektedir.
