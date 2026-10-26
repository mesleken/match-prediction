# Süper Lig Gerçek Geçmiş Veri Walk-Forward Test Raporu (2021/22 - 2026/27)

> **Metodolojik Not:** Bu rapor, geleceğe bakış sızıntısı (lookahead leakage) ve döngüsel kadro kirliliği içermeyen, üretimdeki `config/model.yaml` yapılandırmasını birebir kullanan **Saf Dixon-Coles Modeli**'nin geçmiş out-of-sample başarısını ölçer.
> Model tahminleri; hem **Piyasa Kapanış Oranları (Closing Odds)** hem de **Naif Baz Çizgileri (Sürekli Çoğunluk Sınıfı / Lig Sıklıkları)** ile yan yana, adil ve şeffaf biçimde kıyaslanmıştır.

---

## 1. Özet Karnesi: Saf Model vs Piyasa Kapanışı vs Naif Çizgi

Toplam **1,793** maç incelenmiş, piyasa kapanış oranları eksiksiz mevcut olan **1,793** maç üzerinde eş zamanlı out-of-sample kıyaslama yapılmıştır:

| Metrik | Saf Model | Kapanış Piyasa Çizgisi | Naif Baz Çizgisi | Açıklama & Karşılaştırma |
| :--- | :---: | :---: | :---: | :--- |
| **RPS (Ranked Probability Score)** | **0.2022** | **0.1908** | **0.2276** | Piyasa RPS'de modelden 0.0114 puan daha keskin. |
| **RPS Skill Score (%)** | **%11.2** | **%16.2** | %0.0 | Naife göre kazanç. Model, piyasa üstünlüğünün **%69.1**'sini yakalıyor. |
| **1X2 İsabet Oranı** | **%53.4** | **%55.8** | ~%43.5 | En yüksek olasılıklı seçim isabeti (oran hesabı hariç). |
| **Üst / Alt 2.5 İsabeti** | **%55.0** | — | **%54.4** | Çoğunluk sınıfı taban çizgisine ("Hep Üst") göre fark: **+0.6%**. |
| **KG Var / Yok İsabeti** | **%53.7** | — | **%55.8** | Çoğunluk sınıfı taban çizgisine ("Hep KG Yok") göre fark: **-2.1%**. |
| **İlk 3 Skor İsabeti** | **%31.2** | — | **%30.8** | Ligde en sık biten 3 skora (`1-1`, `1-0`, `2-1`) göre fark: **+0.4%**. |

> [!IMPORTANT]
> **Bilgi Avantajı ve Beceri Tanımı (RPS Skill Score):** Saf Dixon-Coles modeli, naif lig frekansı taban çizgisine göre %11.2'lik bir RPS iyileşmesi (skill score) sağlarken; kapanış piyasası %16.2'lik bir iyileşme sağlamaktadır. Bu durum, modelin naif tabandan piyasaya uzanan tahmin üstünlüğü aralığının **%69.1**'sini (formül: $(RPS_{naive} - RPS_{mod}) / (RPS_{naive} - RPS_{mkt}) = (0.2276 - 0.2022) / (0.2276 - 0.1908) = 69.1\%$) istatistiksel maç motoruyla tek başına yakaladığını gösterir. Bu metrik, "model piyasa bilgisinin %69.1'ini öğrendi" şeklinde değil, "naif tabana kıyasla piyasanın yakaladığı ek tahmin kalitesinin %69.1'ini tek başına elde edebildi" şeklinde yorumlanmalıdır. Kapanış oranlarının modelden daha keskin olmasının temel nedeni; piyasanın son dakika ilk 11 değişikliklerini, motivasyon/hedef durumlarını ve canlı bahis likiditesini doğrudan fiyatlamasıdır.

---

## 2. Sezon Sezon Model - Piyasa Makası

Model ile piyasa kapanış oranları arasındaki fark yıllara göre incelendiğinde:

| Sezon | Maç | Model RPS | Piyasa RPS | Piyasa Farkı ($\Delta$) | Model 1X2 | Piyasa 1X2 | Üst 2.5 (Mod / Baz) | İlk 3 Skor (Mod / Baz) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2122** | 380 | 0.2088 | 0.2045 | +0.0043 | %53.9 | %52.6 | %51.8 / %53.9 | %34.7 / %32.6 |
| **2223** | 313 | 0.2087 | 0.1973 | +0.0114 | %53.4 | %55.3 | %55.6 / %58.5 | %25.2 / %27.5 |
| **2324** | 380 | 0.1972 | 0.1838 | +0.0134 | %54.7 | %55.5 | %53.9 / %51.3 | %32.1 / %31.8 |
| **2425** | 342 | 0.1970 | 0.1803 | +0.0167 | %55.6 | %61.7 | %55.3 / %55.3 | %34.5 / %29.5 |
| **2526** | 306 | 0.1945 | 0.1849 | +0.0096 | %50.0 | %54.6 | %57.2 / %52.9 | %31.4 / %33.3 |
| **2627** | 72 | 0.2229 | 0.2025 | +0.0204 | %48.6 | %54.2 | %63.9 / %58.3 | %18.1 / %26.4 |

> [!NOTE]
> **Sezon Bazlı Dinamikler ve Veri Notları:**
> - **2021/22 Sezonu:** Model ile piyasa arasındaki fark en düşük seviyededir (+0.0043 RPS). Model bu sezonda piyasa ile başa baş bir 1X2 isabeti (%53.9 vs %52.6) yakalamıştır.
> - **2022/23 Sezonu (313 Maçlık Örneklem Denetimi):** 2022/23 sezonu 19 takımlı olarak oynanmış olup resmi fikstürde toplam **342 karşılaşma** planlanmıştır ($19 \times 18 = 342$). 6 Şubat 2023 Kahramanmaraş depremleri sonrasında Hatayspor ve Gaziantep FK ligden çekilmiş; her iki kulübün 23. haftadan itibaren oynanmayan tüm maçları (14 Hatayspor + 14 Gaziantep FK + 1 karşılıklı randevu = **29 maç**) TFF tarafından 3-0 / 0-3 hükmen tescil edilmiştir. Veri ambarında bu 29 maç `status = 'forfeit'` olarak etiketlenmiş olup gerçek saha dinamiklerini yansıtmadığı için out-of-sample model testinden filtrelenmiştir ($342 - 29 = \mathbf{313}$ fiilen oynanmış maç).
> - **2024/25 Sezonu (+0.0167 RPS):** Lig içi bütçe ve kadro kutuplaşmasının zirve yaptığı bu sezonda piyasa kapanış oranları %61.7 gibi olağanüstü yüksek bir 1X2 isabetine ulaşmış, makas model aleyhine açılmıştır (nedensellik izole ablation testlerine muhtaçtır).
> - **2026/27 Sezonu (72 Maç - Devam Eden Sezon):** Yalnızca 72 maçlık erken bir örneklem olup, standart hatası ($\sigma / \sqrt{72}$) geniştir. Adım-adım Holm-Bonferroni yöntemi altında $p = 0.0217 < 0.0250$ olduğundan %5 aile-bazlı hata düzeyinde anlamlı kalmakta ($p_{\text{Holm}} = 0.0434$), ancak tek aşamalı katı Bonferroni eşiğini ($\alpha = 0.00833$) geçememektedir. Tam sezonlarla (342-380 maç) doğrudan karşılaştırılırken küçük örneklem varyansı dikkate alınmalıdır.
---

## 3. Skor İsabeti ve Güven Kademeleri Analizi

Modelin tahmin ettiği 1X2 güven kademeleri ile hem maç sonucu hem de "En Olası İlk 3 Skor" gerçekleşmeleri yan yana kıyaslanmıştır:

| Güven Kademesi | Maç | Ort. Fav Şansı | 1X2 İsabeti [%95 Wilson CI] | Tahmin İlk 3 Skor Kütlesi | Fiili İlk 3 Skor İsabeti [%95 CI] | Naif Skor İsabeti | Model RPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Düşük Güven (< %45)** | 764 | %40.3 | %45.3 [%41.8, %48.8] | %32.1 | **%29.7** [%26.6, %33.0] | %31.5 | 0.2265 |
| **Orta Güven (%45 - %55)** | 598 | %49.4 | %52.0 [%48.0, %56.0] | %31.4 | **%34.3** [%30.6, %38.2] | %31.6 | 0.2054 |
| **Yüksek Güven (%55 - %65)** | 295 | %59.4 | %65.1 [%59.5, %70.3] | %31.4 | **%29.5** [%24.6, %34.9] | %27.8 | 0.1768 |
| **Çok Yüksek Güven (≥ %65)** | 136 | %71.7 | %80.1 [%72.7, %86.0] | %29.5 | **%30.1** [%23.1, %38.3] | %30.1 | 0.1064 |

### Kritik Metodolojik Çıkarımlar:
1. **Skor İsabeti vs 1X2 Güven Gradyanı:** Modelin favoriye verdiği güven %40'tan %72'ye çıktığında 1X2 isabeti %45.3'ten %80.1'e yükselirken, ilk 3 skor isabeti %29.5 - %34.3 aralığında kalmaktadır. Bu bir "Poisson evrensel kuralı" değil; yüksek galibiyet olasılıklı maçlarda galibiyetin birden çok skora (2-0, 2-1, 3-0, 3-1, 1-0, 4-0 vb.) bölünmesinden kaynaklanan ampirik bir olasılık dağılmasıdır.
2. **Skor Olasılık Kütlesi Kalibrasyonu:** Modelin seçtiği ilk 3 skora atadığı teorik olasılık kütlesi (%29.5 - %32.1) ile fiilen gerçekleşen isabet oranları (%29.5 - %34.3) her kademede %95 güven aralığı içindedir; bu durum 'kusursuz kalibrasyonu' kanıtlamaz ancak gruplarda belirgin bir kalibrasyon sapması (miscalibration) gösterilemediğini ortaya koyar.
3. **Doğru Taban Çizgisi:** Skor isabeti "rastgele tahmin ~%1" ile kıyaslanamaz. Süper Lig'de her maça en sık görülen 3 skor (`1-1`, `1-0`, `2-1`) yazıldığında zaten **~%31** isabet elde edilmektedir. Model bu lig dağılımını yakalamaktadır ancak tekil skor kestiriminde naif sıklığın üzerinde belirgin ek üstünlük sağlamamaktadır.
4. **Bahis Stratejisi Uyarısı:** Oransız ham isabet (örneğin Çifte Şans'ta %78 veya favoride %80) bahis karlılığı için tek başına bir kanıt teşkil etmez; çünkü piyasa düşük oranlarla bu olasılığı zaten iskonto etmektedir. Model yalnızca piyasa ile kendi olasılığı arasında pozitif beklenen değer ($+EV$) ve kalibre edilmiş üstünlük bulduğu durumlarda kullanılmalıdır.
