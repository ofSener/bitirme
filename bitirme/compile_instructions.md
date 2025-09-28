# LaTeX DERLEME TALİMATLARI

## ONLINE DERLEME (ÖNERİLEN - HEMEN KULLANILABİLİR)

### Seçenek 1: Overleaf (En Kolay)
1. https://www.overleaf.com adresine gidin
2. Ücretsiz hesap oluşturun
3. "New Project" → "Upload Project" seçin
4. `kolmogorov_thesis_final.tex` dosyasını yükleyin
5. Otomatik olarak derlenecek ve PDF indirebilirsiniz

### Seçenek 2: LaTeX Base
1. https://latexbase.com adresine gidin
2. Sol panele `kolmogorov_thesis_final.tex` içeriğini yapıştırın
3. "Compile" butonuna tıklayın
4. PDF'i indirin

### Seçenek 3: Papeeria
1. https://papeeria.com adresine gidin
2. Ücretsiz hesap oluşturun
3. Yeni proje oluşturun ve dosyayı yükleyin
4. Compile edin

## LOKAL KURULUM

### MacOS için:
```bash
# Tam paket (4GB - tüm özellikler)
brew install --cask mactex

# Minimal paket (90MB - temel özellikler)
brew install --cask basictex

# Kurulumdan sonra terminal'i yeniden başlatın
# Sonra:
cd /Users/ofs/stajj/bitirme
pdflatex kolmogorov_thesis_final.tex
pdflatex kolmogorov_thesis_final.tex  # İkinci kez referanslar için
```

## HAZIR PDF İÇİN

Eğer hemen PDF'e ihtiyacınız varsa:
1. Overleaf.com kullanın (en hızlı)
2. Veya BasicTeX kurun: `brew install --cask basictex`
3. Terminal'i restart edin
4. `pdflatex kolmogorov_thesis_final.tex` komutunu çalıştırın

## NOT
- Grafikleri görmek için PNG dosyalarının `figures/` klasöründe olduğundan emin olun
- İlk derlemede uyarılar normal, ikinci derlemede düzelir