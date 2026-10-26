# Deney I: Ortak Gol/Tempo Modeli ve Yan Pazarlar Walk-Forward Kanıt Kapısı

**Tarih:** 26 Ekim 2026  
**Değerlendirilen Maç Sayısı:** 1,793 (194 maç günü kümesi, 2021/22 - 2026/27, Walk-Forward, Sızıntısız)  
**Kalibrasyon:** Sıcaklık Ölçeklemesi ($T = 0.75$), İki Değişkenli Poisson ($\lambda_0 = 0.1$), Sızıntısız Kademeli Platt Kalibrasyonu  
**Amaç:** Bağımsız Dixon–Coles Poisson yapısının KG Var (BTTS) tarafında negatif BSS üretme sorununu çözmek için test edilen iki değişkenli ortak tempo (Aday I1) ve lojistik kalibrasyon (Aday I2) varyantlarını önceden belirlenmiş I-Kapısı karar kuralıyla denetlemek.

---

## 1. Sayısal Sonuçlar ve Metrik Karşılaştırması

### A. Yan Pazarlar Başarım Tablosu

| Model / Varyant | 2.5 Üst Brier | 2.5 Üst BSS | KG Var Brier | KG Var BSS | KG Var $\Delta$Brier | KG Var Kümeli %95 GA | Holm $p$ (KG Var) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naif Taban Oran** | 0.248034 | %0.00 | 0.246603 | %0.00 | — | — | — |
| **Model B0 (Kalibre Dixon-Coles)** | 0.245542 | %+1.00 | 0.248893 | %-0.93 | — | — | — |
| **Aday I1 (İki Değişkenli Poisson $\lambda_0=0.1$)** | 0.245708 | %+0.94 | 0.248615 | %-0.82 | -0.000278 | [-0.000598, +0.000028] | $p = 0.2570$ |
| **Aday I2 (Sızıntısız Platt Kalibrasyonu)** | 0.245708 | %+0.94 | 0.247172 | %-0.23 | -0.001721 | [-0.003971, +0.000858] | $p = 0.2570$ |
| **Aday I1+I2 (İki Değişkenli + Platt)** | 0.245710 | %+0.94 | 0.247165 | %-0.23 | -0.001728 | [-0.004201, +0.000723] | $p = 0.2570$ |

### B. 1X2 Ana Pazar Yan Etki Kontrolü

| Model / Varyant | 1X2 RPS | 1X2 RPS Farkı | Kümeli %95 GA (RPS) | 1X2 İsabet |
| :--- | :---: | :---: | :---: | :---: |
| **Model B0 (Kalibre Taban)** | **0.199752** | — | — | %53.60 |
| **Aday I1 (İki Değişkenli Poisson)** | 0.199758 | +0.000005 | [-0.000154, +0.000183] | %53.54 |
| **Aday I2 (Platt Kalibrasyonu)** | 0.199752 | +0.000000 | — | %53.60 |
| **Model A (Piyasa Kapanış Oranları)** | **0.190786** | — | — | %55.83 |

---

## 2. Önceden Belirlenmiş I-Kapısı Karar Kuralı Sonucu

> [!IMPORTANT]
> **Karar: I-KAPISI GEÇİLMEDİ (Yan Pazarlarda Tam Pozitif BSS Sağlanamadı - Dixon-Coles Bağımsız Yapısı Korunmalı)**
> 
> - **Aday I1 (İki Değişkenli Poisson):** Ortak kovaryans $\lambda_0 = 0.1$, KG Var Brier skorunu iyileştirmiş (0.248893 $\to$ 0.248615), ancak BSS'yi sıfırın üzerine çıkaramamıştır (BSS: %-0.82). 1X2 RPS üzerinde istatistiksel bozulma yaratmamıştır ($\Delta\text{RPS} = +0.000005$).
> - **Aday I2 (Sızıntısız Platt Kalibrasyonu):** Geçmiş pencereler üzerinden fit edilen lojistik kalibrasyon, KG Var Brier skorunu belirgin biçimde aşağı çekerek negatif BSS açığını önemli ölçüde kapatmıştır (BSS %-0.93'den %-0.23'ye, $\Delta\text{Brier} = -0.001721$). 2.5 Üst BSS'si ise %+0.94 seviyesine ulaşmıştır.
> - **Metodolojik Sonuç ve Üretim Tavsiyesi:**
>   Futbol maçlarında karşılıklı gol oluşumu, Süper Lig genelinde %55.83 taban frekansa sahip olup bağımsız Poisson varsayımının doğal bir kısıtıdır. Aday I2'nin getirdiği lojistik daralma, aşırı güvenli olasılıkları merkezileştirerek hatayı küçültmektedir. Model V3 üretim mimarisinde yan pazarlar servis edilirken ham Poisson olasılıkları yerine sızıntısız Platt kalibrasyonu (`calibrate_binary_market`) isteğe bağlı güvenli katman olarak sunulmalı; ancak kullanıcıya bağımsız modelin yan pazarlarda naif taban üzerinde kesin bir istatistiksel üstünlük iddia etmediği şeffaf biçimde bildirilmelidir.
