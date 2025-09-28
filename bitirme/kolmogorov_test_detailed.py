"""
Enhanced Kolmogorov Complexity Test with Detailed Statistics
Shows compression ratios for each algorithm and detailed comparisons
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
from tabulate import tabulate
import pandas as pd

# Try to import optional libraries
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    print("Warning: zstandard not installed")

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False
    print("Warning: brotli not installed")

try:
    import pyppmd
    PPMD_AVAILABLE = True
except ImportError:
    PPMD_AVAILABLE = False
    print("Warning: pyppmd not installed")


class DetailedKolmogorovTest:
    def __init__(self, base_dir: str):
        """Initialize with base directory"""
        self.base_dir = Path(base_dir)
        self.ai_dir = self.base_dir / "data" / "ai_generated"
        self.human_dir = self.base_dir / "data" / "human_written" / "english"
        self.all_results = []

    def compress_text_detailed(self, text: str) -> Dict[str, Dict]:
        """Compress text and return detailed statistics"""
        text_bytes = text.encode('utf-8')
        original_size = len(text_bytes)

        results = {
            'original': {
                'size': original_size,
                'ratio': 1.0,
                'reduction': 0.0
            }
        }

        # LZMA (7-zip algorithm)
        try:
            compressed = lzma.compress(text_bytes, preset=9)
            size = len(compressed)
            results['lzma'] = {
                'size': size,
                'ratio': size / original_size,
                'reduction': ((original_size - size) / original_size) * 100
            }
        except:
            results['lzma'] = None

        # BZ2 (Burrows-Wheeler)
        try:
            compressed = bz2.compress(text_bytes, compresslevel=9)
            size = len(compressed)
            results['bz2'] = {
                'size': size,
                'ratio': size / original_size,
                'reduction': ((original_size - size) / original_size) * 100
            }
        except:
            results['bz2'] = None

        # Gzip (DEFLATE)
        try:
            compressed = gzip.compress(text_bytes, compresslevel=9)
            size = len(compressed)
            results['gzip'] = {
                'size': size,
                'ratio': size / original_size,
                'reduction': ((original_size - size) / original_size) * 100
            }
        except:
            results['gzip'] = None

        # Zlib (DEFLATE without headers)
        try:
            compressed = zlib.compress(text_bytes, level=9)
            size = len(compressed)
            results['zlib'] = {
                'size': size,
                'ratio': size / original_size,
                'reduction': ((original_size - size) / original_size) * 100
            }
        except:
            results['zlib'] = None

        # Zstandard (Facebook's algorithm)
        if ZSTD_AVAILABLE:
            try:
                cctx = zstd.ZstdCompressor(level=22)
                compressed = cctx.compress(text_bytes)
                size = len(compressed)
                results['zstd'] = {
                    'size': size,
                    'ratio': size / original_size,
                    'reduction': ((original_size - size) / original_size) * 100
                }
            except:
                results['zstd'] = None

        # Brotli (Google's algorithm)
        if BROTLI_AVAILABLE:
            try:
                compressed = brotli.compress(text_bytes, quality=11)
                size = len(compressed)
                results['brotli'] = {
                    'size': size,
                    'ratio': size / original_size,
                    'reduction': ((original_size - size) / original_size) * 100
                }
            except:
                results['brotli'] = None

        # PPMd (Statistical compression)
        if PPMD_AVAILABLE:
            try:
                import pyppmd
                compressed = pyppmd.compress(text_bytes, level=9)
                size = len(compressed)
                results['ppmd'] = {
                    'size': size,
                    'ratio': size / original_size,
                    'reduction': ((original_size - size) / original_size) * 100
                }
            except:
                results['ppmd'] = None

        return results

    def process_file(self, file_path: Path, source_type: str) -> Dict:
        """Process single file with detailed analysis"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if len(text.strip()) == 0:
                return None

            compression_results = self.compress_text_detailed(text)

            # Calculate average complexity across all methods
            ratios = []
            for method, data in compression_results.items():
                if method != 'original' and data and 'ratio' in data:
                    ratios.append(data['ratio'])

            return {
                'filename': file_path.name,
                'source': source_type,
                'original_size': len(text),
                'word_count': len(text.split()),
                'char_count': len(text),
                'compression': compression_results,
                'avg_complexity': statistics.mean(ratios) if ratios else 0
            }

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def compare_tales(self) -> Dict:
        """Compare AI vs Human tales with detailed statistics"""
        print("\nProcessing AI-generated tales...")
        ai_path = self.ai_dir / "tales"
        ai_results = []

        for file_path in ai_path.glob("*.txt"):
            result = self.process_file(file_path, "AI")
            if result:
                ai_results.append(result)
                self.all_results.append(result)

        print(f"Processed {len(ai_results)} AI tales")

        print("\nProcessing Human-written tales...")
        human_path = self.human_dir / "tales"
        human_results = []

        for file_path in human_path.glob("*.txt"):
            result = self.process_file(file_path, "Human")
            if result:
                human_results.append(result)
                self.all_results.append(result)

        print(f"Processed {len(human_results)} human tales")

        return {
            'ai': ai_results,
            'human': human_results
        }

    def print_detailed_statistics(self, results: Dict):
        """Print comprehensive statistics"""
        print("\n" + "="*80)
        print("DETAILED KOLMOGOROV COMPLEXITY ANALYSIS")
        print("="*80)

        # Prepare summary statistics for each compression method
        methods = ['lzma', 'bz2', 'gzip', 'zlib', 'zstd', 'brotli', 'ppmd']

        for source in ['ai', 'human']:
            print(f"\n{source.upper()} TEXTS STATISTICS:")
            print("-"*60)

            source_data = results[source]
            if not source_data:
                print(f"No {source} data available")
                continue

            # Calculate average sizes and ratios for each method
            method_stats = {}
            for method in methods:
                sizes = []
                ratios = []
                reductions = []

                for file_data in source_data:
                    if method in file_data['compression'] and file_data['compression'][method]:
                        sizes.append(file_data['compression'][method]['size'])
                        ratios.append(file_data['compression'][method]['ratio'])
                        reductions.append(file_data['compression'][method]['reduction'])

                if sizes:
                    method_stats[method] = {
                        'avg_size': statistics.mean(sizes),
                        'avg_ratio': statistics.mean(ratios),
                        'avg_reduction': statistics.mean(reductions),
                        'count': len(sizes)
                    }

            # Create table for display
            table_data = []
            avg_original = statistics.mean([f['original_size'] for f in source_data])

            for method, stats in method_stats.items():
                table_data.append([
                    method.upper(),
                    f"{avg_original:.0f} -> {stats['avg_size']:.0f}",
                    f"{stats['avg_ratio']:.3f}",
                    f"{stats['avg_reduction']:.1f}%",
                    stats['count']
                ])

            headers = ["Method", "Size Change", "Ratio", "Reduction", "Files"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))

            # Overall statistics
            avg_complexity = statistics.mean([f['avg_complexity'] for f in source_data])
            avg_words = statistics.mean([f['word_count'] for f in source_data])
            print(f"\nOverall Average Complexity: {avg_complexity:.4f}")
            print(f"Average Word Count: {avg_words:.1f}")
            print(f"Total Files Analyzed: {len(source_data)}")

    def compare_algorithms(self, results: Dict):
        """Compare effectiveness of different algorithms"""
        print("\n" + "="*80)
        print("ALGORITHM COMPARISON (AI vs Human Discrimination)")
        print("="*80)

        methods = ['lzma', 'bz2', 'gzip', 'zlib', 'zstd', 'brotli', 'ppmd']
        comparison_data = []

        for method in methods:
            ai_ratios = []
            human_ratios = []

            # Collect ratios for AI texts
            for file_data in results['ai']:
                if method in file_data['compression'] and file_data['compression'][method]:
                    ai_ratios.append(file_data['compression'][method]['ratio'])

            # Collect ratios for Human texts
            for file_data in results['human']:
                if method in file_data['compression'] and file_data['compression'][method]:
                    human_ratios.append(file_data['compression'][method]['ratio'])

            if ai_ratios and human_ratios:
                ai_avg = statistics.mean(ai_ratios)
                human_avg = statistics.mean(human_ratios)
                difference = human_avg - ai_avg
                percentage = (difference / ai_avg) * 100 if ai_avg > 0 else 0

                comparison_data.append([
                    method.upper(),
                    f"{ai_avg:.3f}",
                    f"{human_avg:.3f}",
                    f"{difference:.3f}",
                    f"{percentage:.1f}%"
                ])

        headers = ["Algorithm", "AI Ratio", "Human Ratio", "Difference", "% Diff"]
        print(tabulate(comparison_data, headers=headers, tablefmt="grid"))

        # Find best discriminator
        if comparison_data:
            best = max(comparison_data, key=lambda x: float(x[4].strip('%')))
            print(f"\n[BEST] Best Discriminator: {best[0]} with {best[4]} difference")

    def save_detailed_results(self, results: Dict):
        """Save all results to multiple formats"""
        # Save detailed JSON to data folder
        data_dir = self.base_dir / "data"
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / "kolmogorov_detailed_results.json"

        # Prepare clean results for JSON
        clean_results = {
            'summary': {
                'ai_files': len(results['ai']),
                'human_files': len(results['human']),
                'ai_avg_complexity': statistics.mean([f['avg_complexity'] for f in results['ai']]),
                'human_avg_complexity': statistics.mean([f['avg_complexity'] for f in results['human']])
            },
            'files': self.all_results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2)
        print(f"\nDetailed results saved to: {output_path}")

        # Save CSV for further analysis
        csv_data = []
        for file_data in self.all_results:
            row = {
                'filename': file_data['filename'],
                'source': file_data['source'],
                'original_size': file_data['original_size'],
                'word_count': file_data['word_count'],
                'avg_complexity': file_data['avg_complexity']
            }

            # Add compression ratios
            for method in ['lzma', 'bz2', 'gzip', 'zlib', 'zstd', 'brotli', 'ppmd']:
                if method in file_data['compression'] and file_data['compression'][method]:
                    row[f'{method}_ratio'] = file_data['compression'][method]['ratio']
                    row[f'{method}_size'] = file_data['compression'][method]['size']
                else:
                    row[f'{method}_ratio'] = None
                    row[f'{method}_size'] = None

            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        csv_path = data_dir / "kolmogorov_detailed_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"CSV results saved to: {csv_path}")

    def print_sample_comparisons(self, results: Dict, num_samples: int = 3):
        """Show specific file comparisons"""
        print("\n" + "="*80)
        print("SAMPLE FILE COMPARISONS")
        print("="*80)

        # Pick random samples
        import random
        ai_samples = random.sample(results['ai'], min(num_samples, len(results['ai'])))
        human_samples = random.sample(results['human'], min(num_samples, len(results['human'])))

        for i, (ai, human) in enumerate(zip(ai_samples, human_samples), 1):
            print(f"\n--- Comparison #{i} ---")
            print(f"AI File: {ai['filename'][:40]}")
            print(f"Human File: {human['filename'][:40]}")

            table_data = []
            for method in ['lzma', 'bz2', 'gzip', 'zstd', 'brotli']:
                if method in ai['compression'] and ai['compression'][method]:
                    if method in human['compression'] and human['compression'][method]:
                        ai_size = ai['compression'][method]['size']
                        human_size = human['compression'][method]['size']
                        table_data.append([
                            method.upper(),
                            f"{ai['original_size']} -> {ai_size}",
                            f"{human['original_size']} -> {human_size}",
                            f"{ai['compression'][method]['ratio']:.3f}",
                            f"{human['compression'][method]['ratio']:.3f}"
                        ])

            headers = ["Method", "AI (orig->comp)", "Human (orig->comp)", "AI Ratio", "Human Ratio"]
            print(tabulate(table_data, headers=headers, tablefmt="simple"))


def main():
    # Set base directory
    base_directory = r"C:\Users\hzome\OneDrive\Masaüstü\bitirme"

    print("Starting Enhanced Kolmogorov Complexity Test...")
    print(f"Base directory: {base_directory}")

    # Create tester
    tester = DetailedKolmogorovTest(base_directory)

    # Run comparison
    results = tester.compare_tales()

    # Print detailed statistics
    tester.print_detailed_statistics(results)

    # Compare algorithms
    tester.compare_algorithms(results)

    # Show sample comparisons
    tester.print_sample_comparisons(results)

    # Save results
    tester.save_detailed_results(results)

    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()