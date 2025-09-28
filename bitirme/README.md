# Kolmogorov Complexity Graduation Project

## Detecting AI-Generated Text Using Compression-Based Analysis

### 🎓 Gebze Technical University - Computer Engineering
**Author:** Ömer Faruk Şener  
**Student ID:** 171044068  
**Advisor:** Dr. Öğr. Üyesi Tülay AYYILDIZ

## 📊 Project Overview

This graduation project investigates whether Kolmogorov complexity can distinguish between AI-generated and human-written texts using compression algorithms as practical approximations.

### Key Findings:
- **10%** complexity difference in academic essays
- **36%** complexity difference in creative fiction
- **85-92%** detection accuracy achieved
- Statistical significance: p < 0.001

## 🔬 Methodology

We analyzed 166 texts (60 essays + 106 tales) using 6 compression algorithms:
- LZMA
- BZ2 (Brotli)
- GZIP
- ZLIB
- ZSTD
- Brotli

## 📈 Results

### Essays Dataset
- AI Mean Complexity: 0.352
- Human Mean Complexity: 0.387
- Difference: 10.0%

### Tales Dataset  
- AI Mean Complexity: 0.309
- Human Mean Complexity: 0.422
- Difference: 36.5%

## 📁 Repository Structure

```
.
├── kolmogorov_thesis_final.tex    # Complete LaTeX thesis
├── essay_kolmogorov_analysis.py   # Main analysis code
├── figures/                        # Visualization outputs
│   ├── essays_visualizations/      # 5 figures for essays
│   └── tales_visualizations/       # 8 figures for tales
├── essay_analysis_results.csv      # Essays dataset results
├── kolmogorov_detailed_results.csv # Tales dataset results
└── quick_compile.sh                # LaTeX compilation script
```

## 🚀 Usage

### Running the Analysis
```python
python essay_kolmogorov_analysis.py
```

### Compiling the Thesis
```bash
./quick_compile.sh
# Or upload kolmogorov_thesis_final.tex to Overleaf
```

## 📖 Citation

If you use this work, please cite:
```
Şener, Ö.F. (2025). Detecting AI-Generated Text Using Compression-Based 
Kolmogorov Complexity Analysis. Graduation Project, Gebze Technical University, 
Department of Computer Engineering.
```

## 📝 License

This project is submitted as a graduation thesis to Gebze Technical University.

---
*September 2025*
