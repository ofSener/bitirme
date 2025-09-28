"""
Enhanced Kolmogorov Complexity Analysis for Essays
Comprehensive comparison between AI-generated and Human-written essays
"""

import os
import json
import lzma
import bz2
import gzip
import zlib
from pathlib import Path
from typing import Dict, List, Tuple
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import compression libraries
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EssayKolmogorovAnalyzer:
    def __init__(self, base_dir: str):
        """Initialize analyzer with base directory"""
        self.base_dir = Path(base_dir)
        self.ai_dir = self.base_dir / "data" / "ai_generated" / "essays"
        self.human_dir = self.base_dir / "data" / "human_written" / "english" / "essays" / "Computer_Science"
        self.results = {'ai': [], 'human': []}
        self.algorithms = ['lzma', 'bz2', 'gzip', 'zlib', 'zstd', 'brotli']

    def compress_text_all_methods(self, text: str) -> Dict:
        """Compress text with all available methods"""
        text_bytes = text.encode('utf-8')
        original_size = len(text_bytes)

        results = {
            'original_size': original_size,
            'compressions': {}
        }

        # LZMA
        try:
            compressed = lzma.compress(text_bytes, preset=9)
            results['compressions']['lzma'] = {
                'size': len(compressed),
                'ratio': len(compressed) / original_size,
                'reduction': (1 - len(compressed) / original_size) * 100
            }
        except:
            results['compressions']['lzma'] = None

        # BZ2
        try:
            compressed = bz2.compress(text_bytes, compresslevel=9)
            results['compressions']['bz2'] = {
                'size': len(compressed),
                'ratio': len(compressed) / original_size,
                'reduction': (1 - len(compressed) / original_size) * 100
            }
        except:
            results['compressions']['bz2'] = None

        # GZIP
        try:
            compressed = gzip.compress(text_bytes, compresslevel=9)
            results['compressions']['gzip'] = {
                'size': len(compressed),
                'ratio': len(compressed) / original_size,
                'reduction': (1 - len(compressed) / original_size) * 100
            }
        except:
            results['compressions']['gzip'] = None

        # ZLIB
        try:
            compressed = zlib.compress(text_bytes, level=9)
            results['compressions']['zlib'] = {
                'size': len(compressed),
                'ratio': len(compressed) / original_size,
                'reduction': (1 - len(compressed) / original_size) * 100
            }
        except:
            results['compressions']['zlib'] = None

        # Zstandard
        if ZSTD_AVAILABLE:
            try:
                cctx = zstd.ZstdCompressor(level=22)
                compressed = cctx.compress(text_bytes)
                results['compressions']['zstd'] = {
                    'size': len(compressed),
                    'ratio': len(compressed) / original_size,
                    'reduction': (1 - len(compressed) / original_size) * 100
                }
            except:
                results['compressions']['zstd'] = None

        # Brotli
        if BROTLI_AVAILABLE:
            try:
                compressed = brotli.compress(text_bytes, quality=11)
                results['compressions']['brotli'] = {
                    'size': len(compressed),
                    'ratio': len(compressed) / original_size,
                    'reduction': (1 - len(compressed) / original_size) * 100
                }
            except:
                results['compressions']['brotli'] = None

        return results

    def analyze_text_complexity(self, text: str) -> Dict:
        """Analyze text complexity with multiple metrics"""
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?')

        # Calculate various metrics
        metrics = {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': sentences,
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / sentences if sentences > 0 else len(words),
            'unique_words': len(set(words)),
            'lexical_diversity': len(set(words)) / len(words) if words else 0,
        }

        return metrics

    def process_directory(self, dir_path: Path, source_type: str):
        """Process all essays in a directory"""
        if not dir_path.exists():
            print(f"Warning: {dir_path} does not exist")
            return []

        results = []
        files = list(dir_path.glob("*.txt"))

        print(f"Processing {len(files)} {source_type} essays...")

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()

                if len(text.strip()) == 0:
                    print(f"Skipping empty file: {file_path.name}")
                    continue

                # Get compression results
                compression_results = self.compress_text_all_methods(text)

                # Get text metrics
                text_metrics = self.analyze_text_complexity(text)

                # Calculate average compression ratio
                ratios = []
                for algo, data in compression_results['compressions'].items():
                    if data and 'ratio' in data:
                        ratios.append(data['ratio'])

                avg_complexity = statistics.mean(ratios) if ratios else 0

                result = {
                    'filename': file_path.name,
                    'source': source_type,
                    'metrics': text_metrics,
                    'compression': compression_results,
                    'avg_complexity': avg_complexity
                }

                results.append(result)

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                continue

        return results

    def run_analysis(self):
        """Run complete analysis on essays"""
        print("="*60)
        print("ESSAY KOLMOGOROV COMPLEXITY ANALYSIS")
        print("="*60)

        # Process AI essays
        self.results['ai'] = self.process_directory(self.ai_dir, 'AI')

        # Process Human essays
        self.results['human'] = self.process_directory(self.human_dir, 'Human')

        print(f"\nProcessed {len(self.results['ai'])} AI essays")
        print(f"Processed {len(self.results['human'])} human essays")

        return self.results

    def statistical_analysis(self):
        """Perform statistical tests"""
        if not self.results['ai'] or not self.results['human']:
            print("No data to analyze")
            return

        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)

        # Extract complexity scores
        ai_complexities = [r['avg_complexity'] for r in self.results['ai']]
        human_complexities = [r['avg_complexity'] for r in self.results['human']]

        # T-test
        t_stat, p_value = stats.ttest_ind(ai_complexities, human_complexities)

        # Mann-Whitney U test (non-parametric)
        u_stat, p_value_mw = stats.mannwhitneyu(ai_complexities, human_complexities)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(ai_complexities)-1)*np.var(ai_complexities) +
                              (len(human_complexities)-1)*np.var(human_complexities)) /
                             (len(ai_complexities) + len(human_complexities) - 2))
        cohens_d = (np.mean(human_complexities) - np.mean(ai_complexities)) / pooled_std

        print(f"\nT-test results:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

        print(f"\nMann-Whitney U test:")
        print(f"  U-statistic: {u_stat:.4f}")
        print(f"  p-value: {p_value_mw:.6f}")

        print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
        print(f"  Interpretation: ", end="")
        if abs(cohens_d) < 0.2:
            print("Negligible")
        elif abs(cohens_d) < 0.5:
            print("Small")
        elif abs(cohens_d) < 0.8:
            print("Medium")
        else:
            print("Large")

    def print_summary_table(self):
        """Print detailed summary table"""
        print("\n" + "="*60)
        print("COMPRESSION ALGORITHM PERFORMANCE")
        print("="*60)

        table_data = []

        for algo in self.algorithms:
            ai_ratios = []
            human_ratios = []

            for result in self.results['ai']:
                if algo in result['compression']['compressions'] and result['compression']['compressions'][algo]:
                    ai_ratios.append(result['compression']['compressions'][algo]['ratio'])

            for result in self.results['human']:
                if algo in result['compression']['compressions'] and result['compression']['compressions'][algo]:
                    human_ratios.append(result['compression']['compressions'][algo]['ratio'])

            if ai_ratios and human_ratios:
                ai_mean = statistics.mean(ai_ratios)
                human_mean = statistics.mean(human_ratios)
                difference = human_mean - ai_mean
                percent_diff = (difference / ai_mean) * 100

                table_data.append([
                    algo.upper(),
                    f"{ai_mean:.4f}",
                    f"{human_mean:.4f}",
                    f"{difference:.4f}",
                    f"{percent_diff:.2f}%"
                ])

        headers = ["Algorithm", "AI Avg", "Human Avg", "Difference", "% Diff"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    def save_results(self):
        """Save analysis results"""
        output = {
            'summary': {
                'ai_count': len(self.results['ai']),
                'human_count': len(self.results['human']),
                'ai_avg_complexity': statistics.mean([r['avg_complexity'] for r in self.results['ai']]) if self.results['ai'] else 0,
                'human_avg_complexity': statistics.mean([r['avg_complexity'] for r in self.results['human']]) if self.results['human'] else 0,
            },
            'detailed_results': self.results
        }

        # Save JSON to data folder
        output_dir = self.base_dir / "data"
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / 'essay_analysis_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        # Create DataFrame for CSV
        rows = []
        for source in ['ai', 'human']:
            for item in self.results[source]:
                row = {
                    'source': source,
                    'filename': item['filename'],
                    'word_count': item['metrics']['word_count'],
                    'char_count': item['metrics']['char_count'],
                    'lexical_diversity': item['metrics']['lexical_diversity'],
                    'avg_complexity': item['avg_complexity']
                }

                # Add compression ratios
                for algo in self.algorithms:
                    if algo in item['compression']['compressions'] and item['compression']['compressions'][algo]:
                        row[f'{algo}_ratio'] = item['compression']['compressions'][algo]['ratio']

                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_dir / 'essay_analysis_results.csv', index=False)

        print("\nResults saved to:")
        print(f"  - {output_dir / 'essay_analysis_results.json'}")
        print(f"  - {output_dir / 'essay_analysis_results.csv'}")


def create_advanced_visualizations(analyzer):
    """Create comprehensive visualizations for essay analysis"""
    # Create outputs directory
    output_dir = analyzer.base_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    # 1. COMPLEXITY DISTRIBUTION COMPARISON
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Overall complexity distribution
    ax = axes[0, 0]
    ai_complexities = [r['avg_complexity'] for r in analyzer.results['ai']]
    human_complexities = [r['avg_complexity'] for r in analyzer.results['human']]

    ax.hist(ai_complexities, bins=15, alpha=0.7, color='#FF6B6B', label='AI', density=True)
    ax.hist(human_complexities, bins=15, alpha=0.7, color='#4ECDC4', label='Human', density=True)
    ax.set_xlabel('Compression Complexity')
    ax.set_ylabel('Density')
    ax.set_title('Overall Complexity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Word count vs Complexity
    ax = axes[0, 1]
    ai_words = [r['metrics']['word_count'] for r in analyzer.results['ai']]
    human_words = [r['metrics']['word_count'] for r in analyzer.results['human']]

    ax.scatter(ai_words, ai_complexities, alpha=0.6, c='#FF6B6B', label='AI', s=30)
    ax.scatter(human_words, human_complexities, alpha=0.6, c='#4ECDC4', label='Human', s=30)
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Complexity')
    ax.set_title('Word Count vs Complexity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Lexical diversity comparison
    ax = axes[0, 2]
    ai_diversity = [r['metrics']['lexical_diversity'] for r in analyzer.results['ai']]
    human_diversity = [r['metrics']['lexical_diversity'] for r in analyzer.results['human']]

    bp = ax.boxplot([ai_diversity, human_diversity], labels=['AI', 'Human'],
                     patch_artist=True, showmeans=True)
    colors = ['#FF6B6B', '#4ECDC4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Lexical Diversity')
    ax.set_title('Vocabulary Richness Comparison')
    ax.grid(True, alpha=0.3)

    # Algorithm performance comparison
    ax = axes[1, 0]
    algorithms = ['lzma', 'bz2', 'gzip', 'brotli']
    ai_means = []
    human_means = []

    for algo in algorithms:
        ai_ratios = []
        human_ratios = []

        for r in analyzer.results['ai']:
            if algo in r['compression']['compressions'] and r['compression']['compressions'][algo]:
                ai_ratios.append(r['compression']['compressions'][algo]['ratio'])

        for r in analyzer.results['human']:
            if algo in r['compression']['compressions'] and r['compression']['compressions'][algo]:
                human_ratios.append(r['compression']['compressions'][algo]['ratio'])

        ai_means.append(statistics.mean(ai_ratios) if ai_ratios else 0)
        human_means.append(statistics.mean(human_ratios) if human_ratios else 0)

    x = np.arange(len(algorithms))
    width = 0.35

    bars1 = ax.bar(x - width/2, ai_means, width, label='AI', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, human_means, width, label='Human', color='#4ECDC4', alpha=0.8)

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Compression Ratio')
    ax.set_title('Algorithm Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algorithms])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reduction percentage by algorithm
    ax = axes[1, 1]
    ai_reductions = []
    human_reductions = []

    for algo in algorithms:
        ai_red = []
        human_red = []

        for r in analyzer.results['ai']:
            if algo in r['compression']['compressions'] and r['compression']['compressions'][algo]:
                ai_red.append(r['compression']['compressions'][algo]['reduction'])

        for r in analyzer.results['human']:
            if algo in r['compression']['compressions'] and r['compression']['compressions'][algo]:
                human_red.append(r['compression']['compressions'][algo]['reduction'])

        ai_reductions.append(statistics.mean(ai_red) if ai_red else 0)
        human_reductions.append(statistics.mean(human_red) if human_red else 0)

    x = np.arange(len(algorithms))
    bars1 = ax.bar(x - width/2, ai_reductions, width, label='AI', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, human_reductions, width, label='Human', color='#4ECDC4', alpha=0.8)

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Size Reduction (%)')
    ax.set_title('Compression Efficiency')
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algorithms])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Statistical significance visualization
    ax = axes[1, 2]
    differences = [h - a for h, a in zip(human_means, ai_means)]
    colors = ['#2ECC71' if d > 0 else '#E74C3C' for d in differences]

    bars = ax.bar([a.upper() for a in algorithms], differences, color=colors, alpha=0.8)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Complexity Difference (Human - AI)')
    ax.set_title('Discrimination Power by Algorithm')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3)

    for bar, diff in zip(bars, differences):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{diff:.3f}', ha='center',
                va='bottom' if height > 0 else 'top')

    plt.suptitle('Essay Kolmogorov Complexity Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'essay_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'essay_analysis_comprehensive.png'}")

    # 2. DETAILED METRICS COMPARISON
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Average word length distribution
    ax = axes[0, 0]
    ai_avg_word = [r['metrics']['avg_word_length'] for r in analyzer.results['ai']]
    human_avg_word = [r['metrics']['avg_word_length'] for r in analyzer.results['human']]

    parts = ax.violinplot([ai_avg_word, human_avg_word], positions=[0, 1],
                          showmeans=True, showmedians=True)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['AI', 'Human'])
    ax.set_ylabel('Average Word Length')
    ax.set_title('Word Length Distribution')
    ax.grid(True, alpha=0.3)

    # Sentence length comparison
    ax = axes[0, 1]
    ai_sent_len = [r['metrics']['avg_sentence_length'] for r in analyzer.results['ai']]
    human_sent_len = [r['metrics']['avg_sentence_length'] for r in analyzer.results['human']]

    ax.boxplot([ai_sent_len, human_sent_len], labels=['AI', 'Human'],
               patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax.set_ylabel('Average Sentence Length (words)')
    ax.set_title('Sentence Complexity')
    ax.grid(True, alpha=0.3)

    # Unique words vs Total words
    ax = axes[1, 0]
    ai_unique = [r['metrics']['unique_words'] for r in analyzer.results['ai']]
    human_unique = [r['metrics']['unique_words'] for r in analyzer.results['human']]

    ax.scatter(ai_words, ai_unique, alpha=0.6, c='#FF6B6B', label='AI', s=30)
    ax.scatter(human_words, human_unique, alpha=0.6, c='#4ECDC4', label='Human', s=30)
    ax.set_xlabel('Total Words')
    ax.set_ylabel('Unique Words')
    ax.set_title('Vocabulary Usage Pattern')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Complexity vs Lexical Diversity
    ax = axes[1, 1]
    ax.scatter(ai_diversity, ai_complexities, alpha=0.6, c='#FF6B6B', label='AI', s=30)
    ax.scatter(human_diversity, human_complexities, alpha=0.6, c='#4ECDC4', label='Human', s=30)
    ax.set_xlabel('Lexical Diversity')
    ax.set_ylabel('Compression Complexity')
    ax.set_title('Diversity vs Complexity Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Essay Linguistic Metrics Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'essay_metrics_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'essay_metrics_analysis.png'}")

    # 3. ALGORITHM CORRELATION HEATMAP
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Prepare data for correlation
    algorithms = ['lzma', 'bz2', 'gzip', 'brotli']

    # AI correlation matrix
    ai_data = {algo: [] for algo in algorithms}
    for r in analyzer.results['ai']:
        for algo in algorithms:
            if algo in r['compression']['compressions'] and r['compression']['compressions'][algo]:
                ai_data[algo].append(r['compression']['compressions'][algo]['ratio'])

    # Ensure all lists have same length (pad with NaN if needed)
    max_len = max(len(ai_data[algo]) for algo in algorithms)
    for algo in algorithms:
        while len(ai_data[algo]) < max_len:
            ai_data[algo].append(np.nan)

    ai_df = pd.DataFrame(ai_data)
    ai_corr = ai_df.corr()

    sns.heatmap(ai_corr, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, ax=ax1,
                cbar_kws={"shrink": 0.8})
    ax1.set_title('AI Essays: Algorithm Correlation')

    # Human correlation matrix
    human_data = {algo: [] for algo in algorithms}
    for r in analyzer.results['human']:
        for algo in algorithms:
            if algo in r['compression']['compressions'] and r['compression']['compressions'][algo]:
                human_data[algo].append(r['compression']['compressions'][algo]['ratio'])

    max_len = max(len(human_data[algo]) for algo in algorithms)
    for algo in algorithms:
        while len(human_data[algo]) < max_len:
            human_data[algo].append(np.nan)

    human_df = pd.DataFrame(human_data)
    human_corr = human_df.corr()

    sns.heatmap(human_corr, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, ax=ax2,
                cbar_kws={"shrink": 0.8})
    ax2.set_title('Human Essays: Algorithm Correlation')

    plt.suptitle('Compression Algorithm Correlation Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'essay_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'essay_correlation_heatmap.png'}")


def main():
    """Main execution function"""
    base_dir = r"C:\Users\hzome\OneDrive\Masaüstü\bitirme"

    # Initialize analyzer
    analyzer = EssayKolmogorovAnalyzer(base_dir)

    # Run analysis
    analyzer.run_analysis()

    # Print summary table
    analyzer.print_summary_table()

    # Statistical analysis
    analyzer.statistical_analysis()

    # Create visualizations
    create_advanced_visualizations(analyzer)

    # Save results
    analyzer.save_results()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - essay_analysis_results.json")
    print("  - essay_analysis_results.csv")
    print("  - essay_analysis_comprehensive.png")
    print("  - essay_metrics_analysis.png")
    print("  - essay_correlation_heatmap.png")


if __name__ == "__main__":
    main()