# Kolmogorov Complexity Analysis - AI vs Human Text Detection

Bu proje, yapay zeka tarafından üretilen ve insan tarafından yazılan metinleri ayırt etmek için Kolmogorov karmaşıklık analizi kullanır.

## Proje Yapısı

```
bitirme/
├── code/                    # Kaynak kodları
├── data/                    # Veri dosyaları
├── outputs/                 # Üretilen görseller ve grafikler
├── docs/                    # Dokümantasyon
└── README.md               # Bu dosya
```

## Kod Dosyaları

### `code/individual_visualizations.py`
Ana görselleştirme scripti. Her grafik için ayrı PNG dosyası üretir.

**Kullanım:**
```bash
python individual_visualizations.py essays
python individual_visualizations.py tales
```

### `code/essay_kolmogorov_analysis.py`
Essay veri seti için kapsamlı Kolmogorov analizi.

### `code/kolmogorov_test_detailed.py`
Tales veri seti için detaylı kompresyon analizi.

## Veri Yapısı

### `data/`
- `ai_generated/` - Yapay zeka üretimi metinler
- `human_written/` - İnsan yazımı metinler
- `essay_analysis_results.csv` - Essay analiz sonuçları
- `kolmogorov_detailed_results.csv` - Tales analiz sonuçları
- `requirements.txt` - Python bağımlılıkları

## Çıktılar

### `outputs/essays_visualizations/`
Essay veri seti için 8 adet görsel:
1. Dağılım karşılaştırma histogramı
2. Kutu grafiği istatistiksel karşılaştırma
3. Algoritma performans karşılaştırması
4. İstatistiksel anlamlılık testleri
5. Algoritma korelasyon matrisi
6. Temel bileşen analizi (PCA)
7. Random Forest özellik önemi
8. K-means kümeleme analizi

### `outputs/tales_visualizations/`
Tales veri seti için aynı 8 görsel.

### `docs/`
- `essays_chart_documentation.txt` - Essay grafikleri açıklamaları
- `tales_chart_documentation.txt` - Tales grafikleri açıklamaları

## Ana Bulgular

### Essays Dataset:
- İnsan metinleri %10.0 daha karmaşık (ortalama)
- İstatistiksel anlamlılık: p = 0.004513
- En iyi ayırt edici algoritma: BZ2 (%11.0 fark)
- Random Forest modeli: %100 doğruluk

### Tales Dataset:
- İnsan metinleri %36.5 daha karmaşık (ortalama)
- LZMA algoritması en yüksek farkı gösterir
- Belirgin istatistiksel anlamlılık

## Metodoloji

Kolmogorov karmaşıklığı, metinlerin sıkıştırılabilirlik oranları üzerinden tahmin edilir:
- Düşük sıkıştırma oranı = Yüksek karmaşıklık
- Yapay zeka metinleri daha öngörülebilir desenler içerir
- İnsan metinleri daha rastgele ve yaratıcı içerik barındırır

## Kullanılan Algoritmalar

- **LZMA**: Yüksek sıkıştırma, metin analizi için ideal
- **BZ2**: Burrows-Wheeler dönüşümü, desen tespiti
- **GZIP**: Hızlı, yaygın kullanım
- **Brotli**: Modern algoritma, metin için optimize