# Skor Modeli Derin Analiz Raporu: İlk 3 Skor İsabeti ve Entropi İncelemesi

> **Özet:** Saf modelin favoriye verdiği galibiyet olasılığı %40.3'ten %71.7'ye yükseldiğinde, gerçekleşen 1X2 maç sonucu isabeti **%45.3'ten %80.1'e (+34.8 puan)** sıçramaktadır.
> Buna karşın aynı maçlarda modelin öngördüğü en olası "İlk 3 Skor" gerçekleşme oranı **%29.5 ile %34.3 arasında (~%31) neredeyse düz bir çizgi** izlemektedir.
> Bu rapor; bu olgunun Poisson dağılımı, skor entropisi, empirik kalibrasyon ve naif önsel kıyaslamasıyla titiz bir matematiksel analizini sunar.

---

## 1. Güven Kademelerine Göre Skor ve 1X2 Kalibrasyonu (1793 Maç)

Aşağıdaki tablo, modelin güven kademelerine göre 1X2 başarısını, modelin ilk 3 skor için atadığı teorik olasılık kütlesini, gerçekleşen ilk 3 skor isabetini (95% Wilson Güven Aralığı ile) ve ligin en sık görülen 3 skorunu (1-1, 2-1, 1-0) sabit tahmin eden naif taban çizgisini göstermektedir:

| Güven Kademesi | Maç Sayısı | Ort. Fav Şansı | 1X2 İsabeti [%95 GA] | Model Öngörülen Top-3 Kütlesi | Model Top-3 İsabeti [%95 GA] | Naif Top-3 İsabeti [%95 GA] | Model Top-5 İsabeti |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Düşük Güven (< %45)** | 764 | %40.3 | **%45.3** <br><small>[%41.8, %48.8]</small> | %32.1 | **%29.7** <br><small>[%26.6, %33.0]</small> | %31.5 <br><small>[%28.3, %34.9]</small> | %45.9 |
| **Orta Güven (%45 - %55)** | 598 | %49.4 | **%52.0** <br><small>[%48.0, %56.0]</small> | %31.4 | **%34.3** <br><small>[%30.6, %38.2]</small> | %31.6 <br><small>[%28.0, %35.4]</small> | %51.7 |
| **Yüksek Güven (%55 - %65)** | 295 | %59.4 | **%65.1** <br><small>[%59.5, %70.3]</small> | %31.4 | **%29.5** <br><small>[%24.6, %34.9]</small> | %27.8 <br><small>[%23.0, %33.2]</small> | %48.5 |
| **Çok Yüksek Güven (≥ %65)** | 136 | %71.7 | **%80.1** <br><small>[%72.7, %86.0]</small> | %29.5 | **%30.1** <br><small>[%23.1, %38.3]</small> | %30.1 <br><small>[%23.1, %38.3]</small> | %44.9 |

---

## 2. Süper Lig'in En Sık Biten 10 Skoru (Tarihsel Önsel)

| Sıra | Skor | Maç Sayısı | Lig İçi Görülme Sıklığı | Kümülatif Sıklık |
| :---: | :---: | :---: | :---: | :---: |
| 1 | **1-1** | 215 | %11.99 | %12.0 |
| 2 | **2-1** | 177 | %9.87 | %21.9 |
| 3 | **1-0** | 161 | %8.98 | %30.8 |
| 4 | **2-0** | 126 | %7.03 | %37.9 |
| 5 | **0-0** | 124 | %6.92 | %44.8 |
| 6 | **1-2** | 123 | %6.86 | %51.6 |
| 7 | **0-1** | 114 | %6.36 | %58.0 |
| 8 | **2-2** | 83 | %4.63 | %62.6 |
| 9 | **3-1** | 81 | %4.52 | %67.2 |
| 10 | **0-2** | 77 | %4.29 | %71.5 |

---

## 3. Matematiksel Analiz: Süper Lig Skor Entropisi ve Poisson Dinamiği

### A. Yanlış Bir Varsayımın Düzeltilmesi (Evrensel Üst Sınır Yanılgısı)
Önceki raporda *"Poisson dağılımının matematiksel yapısı gereği ilk 3 skor olasılığı evrensel olarak %25-%30'a mahkumdur"* şeklinde ifade edilen iddia **matematiksel olarak hatalıdır**. 

*Karşı Örnek:* Düşük gollü liglerde veya defansif maçlarda $\lambda = 0.50$ ve $\mu = 0.40$ olduğunda:
$$P(0-0) = e^{-0.5} \times e^{-0.4} = e^{-0.9} \approx 0.4066 \quad (\%40.7)$$
Yalnızca `0-0` skoru tek başına %40'ı aşmakta; `0-0`, `1-0` (%20.3) ve `0-1` (%16.3) skorlarının toplam olasılık kütlesi **%77.3'e** ulaşmaktadır. Dolayısıyla Poisson dağılımı ilk 3 skoru evrensel olarak %30 ile sınırlamaz.

### B. Analitik Poisson ve Süper Lig Skor Rejimi Karşılaştırması

Hakem incelemesinde yerinde vurgulandığı üzere, teorik Poisson formülleri ile empirik model çıktıları birbirinden açıkça ayrılmalıdır:

1. **Teorik Düşük Deplasman Beklentisi ($\lambda = 2.65, \mu = 0.65$):**
   Bağımsız Poisson analitik formülüyle ($P(X=x, Y=y) = \frac{\lambda^x e^{-\lambda}}{x!} \frac{\mu^y e^{-\mu}}{y!}$):
   - $P(2-0) = \frac{2.65^2 e^{-2.65}}{2!} \times e^{-0.65} = 0.2481 \times 0.5220 = \mathbf{12.95\%}$
   - $P(3-0) = \frac{2.65^3 e^{-2.65}}{3!} \times e^{-0.65} = 0.2191 \times 0.5220 = \mathbf{11.44\%}$
   - $P(1-0) = 2.65 e^{-2.65} \times e^{-0.65} = 0.1872 \times 0.5220 = \mathbf{9.77\%}$
   - $P(2-1) = 0.2481 \times (0.65 e^{-0.65}) = 0.2481 \times 0.3393 = \mathbf{8.42\%}$
   - $P(4-0) = \mathbf{7.58\%}, \quad P(3-1) = \mathbf{7.44\%}, \quad P(1-1) = \mathbf{6.35\%}, \quad P(4-1) = \mathbf{4.93\%}$
   - Bu teorik parametre setinde en olası ilk 3 skor `2-0`, `3-0` ve `1-0` olup toplam olasılık kütlesi **%34.16**'dır.

2. **Gerçekçi Süper Lig Ağır Favori Parametresi ($\lambda = 2.50, \mu = 0.90$):**
   Süper Lig'de zirve takımlarının iç saha maçlarında dahi rakipler ortalama $\mu \approx 0.90$ gol beklentisine sahiptir. Clean sheet (gol yememe) olasılığı $e^{-0.90} \approx 40.7\%$ seviyesine indiğinde skorlar hızla dağılır:
   - `2-0`: %10.43, `2-1`: %9.39, `3-0`: %8.69, `1-0`: %8.34, `3-1`: %7.82, `1-1`: %7.51, `4-0`: %5.43, `4-1`: %4.89
   - İlk 3 skorun (`2-0`, `2-1`, `3-0`) toplam olasılık kütlesi **%28.51**'e iner.

3. **Empirik Model Çıktıları (136 Maçlık Çok Yüksek Güven Dilimi):**
   - Dixon-Coles modelimiz, parametre belirsizliğini (Laplace kovaryansı üzerinden Gauss-Hermite kuadratürü ile) entegre ettiği için marjinal olasılık kütlesi aşırı yayılımlı (overdispersed) bir Poisson karışımı oluşturur.
   - 136 maçta modelin öngördüğü en olası ilk 3 skorun ortalama toplam olasılık kütlesi **%29.48 (~%29.5)**'tir.
   - Fiilen gerçekleşen ilk 3 skor isabeti ise **%30.15 (~%30.1)** [%95 Wilson GA: `%23.1% - %38.3%`]'tir.

### C. Empirik Kalibrasyon ve Naif Taban Çizgisi Değerlendirmesi
1. **Kalibrasyon Değerlendirmesi:**
   - Modelin ilk 3 skor için atadığı teorik olasılık kütlesi (%29.5 - %32.1), gerçekleşen isabet oranlarının (%29.5 - %34.3) %95 Wilson güven aralığı içindedir. Modelin 'kusursuz kalibre edildiği' iddia edilemez; veriler mevcut gruplarda belirgin bir kalibrasyon sapmasının (miscalibration) gösterilemediğini ortaya koymaktadır.
2. **Model Naif Önsel Tahminden Üstün müdür?**
   - Ligin tarihsel olarak en sık biten 3 skoru `1-1`, `2-1` ve `1-0` ligdeki tüm maçların **%30.8'ini** oluşturmaktadır.
   - Modelin maça özel dinamik en iyi 3 skoru toplamda **%31.2** isabet sağlamaktadır.
   - Naif tahmine karşı fark yalnızca **+0.4 puan** olup istatistiksel olarak farksızdır ($p > 0.60$).
   - Başka bir deyişle, bağımsız Poisson/Dixon-Coles marjinal gol modeli, tekil skor seviyesinde tarihsel önsel sıklıktan anlamlı bir ek tahmin avantajı üretememektedir.

---

## 4. Çözüm Önerileri ve Simülasyon Motoru Yol Haritası

1. **Tekil Skor Yerine Skor Kümeleri / Aralık Piyasaları:**
   - Kullanıcıya yalnızca 3 tekil skor sunmak, özellikle yüksek varyanslı maçlarda gerçeği yansıtmaz.
   - Bunun yerine **"Skor Kümeleri"** (Örn: Favori Farklı Galibiyet: `2-0, 3-0, 3-1, 4-1` $\implies$ %45 Kapsama) veya Asya Handikap / Gol Aralığı (2-3 Gol, 4-6 Gol) türev olasılıkları raporlanmalıdır.
2. **Bivariate Copula (Çift Değişkenli Kopula) Entegrasyonu:**
   - Dixon-Coles $\rho$ parametresi yalnızca düşük skorları (0-0, 1-0, 0-1, 1-1) öteler.
   - Maçın 2.5 üstü skorlarında gollerin karşılıklı gelme eğilimini modellemek için Clayton veya Frank copula mimarisine geçilmelidir.
3. **Simülasyon Motorunda Olay Tabanlı Skor Çıkışı (`MatchEngine`):**
   - Statik olasılık matrisi yerine 90 dakikalık durum geçişleri (state-transition), yorgunluk ve tempo dinamiklerini simüle eden maç motoru, maçın kopma veya kapanma olasılıklarını skor matrisine yansıtacaktır.
