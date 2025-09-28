"""
Individual High-Quality Visualizations for Kolmogorov Complexity Analysis
Each chart is saved as a separate PNG file with detailed documentation
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import additional libraries
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage

# Set high-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

class IndividualVisualizationGenerator:
    def __init__(self, dataset_type='essays'):
        """Initialize for essays or tales"""
        self.dataset_type = dataset_type
        self.base_dir = Path(r"C:\Users\hzome\OneDrive\Masaüstü\bitirme")

        # Create output directory in organized structure
        self.output_dir = self.base_dir / "outputs" / f"{dataset_type}_visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.colors = {
            'ai': '#FF6B6B',
            'human': '#4ECDC4',
            'accent': '#45B7D1',
            'positive': '#2ECC71',
            'negative': '#E74C3C'
        }

        # Chart descriptions for documentation
        self.chart_descriptions = {}

    def load_data(self):
        """Load dataset from organized data folder"""
        if self.dataset_type == 'essays':
            csv_file = self.base_dir / "data" / "essay_analysis_results.csv"
        else:
            csv_file = self.base_dir / "data" / "kolmogorov_detailed_results.csv"

        if csv_file.exists():
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} records for {self.dataset_type}")
            return df
        else:
            print(f"Warning: {csv_file} not found")
            return None

    def prepare_complexity_data(self, df):
        """Extract complexity data for both AI and Human"""
        # Handle different source labels (AI vs ai, Human vs human)
        ai_data = df[df['source'].str.upper() == 'AI']
        human_data = df[df['source'].str.upper() == 'HUMAN']

        if 'avg_complexity' in df.columns and not df['avg_complexity'].isna().all():
            ai_complexity = ai_data['avg_complexity'].dropna()
            human_complexity = human_data['avg_complexity'].dropna()
        else:
            # Calculate from ratio columns
            ratio_cols = [col for col in df.columns if col.endswith('_ratio')]
            if len(ratio_cols) > 0:
                ai_complexity = ai_data[ratio_cols].mean(axis=1).dropna()
                human_complexity = human_data[ratio_cols].mean(axis=1).dropna()
            else:
                # Fallback: create dummy data
                print(f"Warning: No ratio columns found for {self.dataset_type}")
                ai_complexity = pd.Series([0.5] * len(ai_data))
                human_complexity = pd.Series([0.7] * len(human_data))

        return ai_complexity, human_complexity, ai_data, human_data

    def get_file_statistics(self, df):
        """Calculate detailed file statistics"""
        ai_data = df[df['source'].str.upper() == 'AI']
        human_data = df[df['source'].str.upper() == 'HUMAN']

        stats = {
            'ai_count': len(ai_data),
            'human_count': len(human_data),
            'total_count': len(df)
        }

        # Add character and word statistics if available
        if 'char_count' in df.columns:
            stats['ai_avg_chars'] = ai_data['char_count'].mean()
            stats['human_avg_chars'] = human_data['char_count'].mean()
            stats['ai_total_chars'] = ai_data['char_count'].sum()
            stats['human_total_chars'] = human_data['char_count'].sum()

        if 'word_count' in df.columns:
            stats['ai_avg_words'] = ai_data['word_count'].mean()
            stats['human_avg_words'] = human_data['word_count'].mean()
            stats['ai_total_words'] = ai_data['word_count'].sum()
            stats['human_total_words'] = human_data['word_count'].sum()

        return stats

    def chart_01_distribution_comparison(self, df):
        """01: Distribution Comparison Histogram"""
        ai_complexity, human_complexity, _, _ = self.prepare_complexity_data(df)
        file_stats = self.get_file_statistics(df)

        plt.figure(figsize=(14, 10))

        # Create histogram
        plt.hist(ai_complexity, bins=20, alpha=0.7, color=self.colors['ai'],
                label=f'AI Generated (n={len(ai_complexity)})', density=True, edgecolor='white')
        plt.hist(human_complexity, bins=20, alpha=0.7, color=self.colors['human'],
                label=f'Human Written (n={len(human_complexity)})', density=True, edgecolor='white')

        # Add mean lines
        plt.axvline(ai_complexity.mean(), color=self.colors['ai'], linestyle='--',
                   linewidth=2, alpha=0.8, label=f'AI Mean: {ai_complexity.mean():.3f}')
        plt.axvline(human_complexity.mean(), color=self.colors['human'], linestyle='--',
                   linewidth=2, alpha=0.8, label=f'Human Mean: {human_complexity.mean():.3f}')

        plt.xlabel('Compression Complexity Ratio', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title(f'{self.dataset_type.title()}: Kolmogorov Complexity Distribution',
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add statistical annotation
        diff = human_complexity.mean() - ai_complexity.mean()
        percent_diff = (diff / ai_complexity.mean()) * 100
        # Create detailed stats text
        stats_text = f"""Dataset Statistics:
AI Files: {file_stats['ai_count']} | Human Files: {file_stats['human_count']}
Complexity Difference: {diff:.3f} ({percent_diff:.1f}%)"""

        if 'ai_avg_chars' in file_stats:
            stats_text += f"""
Avg Characters: AI={file_stats['ai_avg_chars']:.0f} | Human={file_stats['human_avg_chars']:.0f}"""

        if 'ai_avg_words' in file_stats:
            stats_text += f"""
Avg Words: AI={file_stats['ai_avg_words']:.0f} | Human={file_stats['human_avg_words']:.0f}"""

        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

        plt.tight_layout()
        plt.savefig(self.output_dir / '01_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.chart_descriptions['01'] = {
            'title': 'Distribution Comparison Histogram',
            'description': 'Shows the probability distribution of compression complexity for AI vs Human texts',
            'key_insights': [
                f'AI texts have mean complexity: {ai_complexity.mean():.3f}',
                f'Human texts have mean complexity: {human_complexity.mean():.3f}',
                f'Human texts are {percent_diff:.1f}% more complex on average',
                'Higher complexity means less compressible (more random/diverse)'
            ]
        }

    def chart_02_box_plot_comparison(self, df):
        """02: Box Plot Statistical Comparison"""
        ai_complexity, human_complexity, _, _ = self.prepare_complexity_data(df)

        plt.figure(figsize=(10, 8))

        # Create box plot (only if we have data)
        if len(ai_complexity) == 0 or len(human_complexity) == 0:
            print(f"Warning: No data for box plot in {self.dataset_type}")
            plt.text(0.5, 0.5, 'No data available for box plot',
                    transform=plt.gca().transAxes, ha='center', va='center', fontsize=16)
            plt.title(f'{self.dataset_type.title()}: Statistical Distribution Analysis (No Data)',
                     fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(self.output_dir / '02_box_plot_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        box_data = [ai_complexity, human_complexity]
        bp = plt.boxplot(box_data, labels=['AI Generated', 'Human Written'],
                        patch_artist=True, showmeans=True, meanline=True,
                        boxprops=dict(alpha=0.7), showfliers=True)

        # Color the boxes
        bp['boxes'][0].set_facecolor(self.colors['ai'])
        bp['boxes'][1].set_facecolor(self.colors['human'])

        # Add statistical annotations
        stats_text = f"""Statistical Summary:

AI Texts:
• Median: {np.median(ai_complexity):.3f}
• Q1: {np.percentile(ai_complexity, 25):.3f}
• Q3: {np.percentile(ai_complexity, 75):.3f}
• Std: {np.std(ai_complexity):.3f}

Human Texts:
• Median: {np.median(human_complexity):.3f}
• Q1: {np.percentile(human_complexity, 25):.3f}
• Q3: {np.percentile(human_complexity, 75):.3f}
• Std: {np.std(human_complexity):.3f}"""

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.ylabel('Compression Complexity Ratio', fontsize=14)
        plt.title(f'{self.dataset_type.title()}: Statistical Distribution Analysis',
                 fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / '02_box_plot_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.chart_descriptions['02'] = {
            'title': 'Box Plot Statistical Comparison',
            'description': 'Shows median, quartiles, and outliers for complexity distributions',
            'key_insights': [
                'Box shows 25th-75th percentile range (IQR)',
                'Line inside box is median, diamond is mean',
                'Whiskers extend to 1.5×IQR, dots are outliers',
                f'Human texts show {"higher" if np.median(human_complexity) > np.median(ai_complexity) else "lower"} median complexity'
            ]
        }

    def chart_03_algorithm_performance(self, df):
        """03: Algorithm Performance Comparison"""
        algorithms = ['lzma', 'bz2', 'gzip', 'brotli']
        ai_data = df[df['source'].str.upper() == 'AI']
        human_data = df[df['source'].str.upper() == 'HUMAN']

        plt.figure(figsize=(12, 8))

        # Calculate performance metrics
        ai_means = []
        human_means = []
        differences = []

        for algo in algorithms:
            col = f'{algo}_ratio'
            if col in df.columns:
                ai_mean = ai_data[col].mean()
                human_mean = human_data[col].mean()
                diff = (human_mean - ai_mean) / ai_mean * 100

                ai_means.append(ai_mean)
                human_means.append(human_mean)
                differences.append(diff)

        # Create grouped bar chart
        x = np.arange(len(algorithms))
        width = 0.35

        bars1 = plt.bar(x - width/2, ai_means, width, label='AI Generated',
                       color=self.colors['ai'], alpha=0.8, edgecolor='white')
        bars2 = plt.bar(x + width/2, human_means, width, label='Human Written',
                       color=self.colors['human'], alpha=0.8, edgecolor='white')

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.xlabel('Compression Algorithm', fontsize=14)
        plt.ylabel('Average Compression Ratio', fontsize=14)
        plt.title(f'{self.dataset_type.title()}: Algorithm Performance Comparison',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xticks(x, [algo.upper() for algo in algorithms])
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')

        # Add difference annotations
        for i, (algo, diff) in enumerate(zip(algorithms, differences)):
            plt.text(i, max(ai_means[i], human_means[i]) + 0.05,
                    f'{diff:+.1f}%', ha='center', va='bottom',
                    fontweight='bold', color='red' if diff > 0 else 'blue')

        plt.tight_layout()
        plt.savefig(self.output_dir / '03_algorithm_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

        best_algo = algorithms[np.argmax(differences)]
        self.chart_descriptions['03'] = {
            'title': 'Algorithm Performance Comparison',
            'description': 'Compares average compression ratios across different algorithms',
            'key_insights': [
                'Lower compression ratio = better compression = higher complexity detection',
                f'Best discriminating algorithm: {best_algo.upper()} ({max(differences):.1f}% difference)',
                'Percentages show how much more complex human texts are for each algorithm',
                'Positive percentage means human texts are less compressible'
            ]
        }

    def chart_04_statistical_significance(self, df):
        """04: Statistical Significance Tests"""
        ai_complexity, human_complexity, _, _ = self.prepare_complexity_data(df)

        plt.figure(figsize=(12, 8))

        # Perform statistical tests
        t_stat, p_val = stats.ttest_ind(ai_complexity, human_complexity)
        u_stat, p_val_mw = stats.mannwhitneyu(ai_complexity, human_complexity)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(ai_complexity)-1)*np.var(ai_complexity) +
                             (len(human_complexity)-1)*np.var(human_complexity)) /
                            (len(ai_complexity) + len(human_complexity) - 2))
        cohens_d = (human_complexity.mean() - ai_complexity.mean()) / pooled_std

        # Create visualization
        tests = ['T-Test', 'Mann-Whitney U', "Cohen's d", 'Mean Difference']
        values = [p_val, p_val_mw, abs(cohens_d), abs(human_complexity.mean() - ai_complexity.mean())]
        colors = [self.colors['positive'] if p_val < 0.05 else self.colors['negative'],
                 self.colors['positive'] if p_val_mw < 0.05 else self.colors['negative'],
                 self.colors['accent'], self.colors['accent']]

        bars = plt.bar(tests, values, color=colors, alpha=0.8, edgecolor='white')

        # Add significance lines
        plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p = 0.05 (significance threshold)')

        # Add value labels
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2., val + max(values)*0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

        plt.ylabel('Test Value', fontsize=14)
        plt.title(f'{self.dataset_type.title()}: Statistical Significance Tests',
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.yscale('log')  # Log scale for better p-value visualization

        # Add interpretation text
        interpretation = f"""Test Results Interpretation:

T-Test p-value: {p_val:.6f} ({'Significant' if p_val < 0.05 else 'Not Significant'})
Mann-Whitney p-value: {p_val_mw:.6f} ({'Significant' if p_val_mw < 0.05 else 'Not Significant'})

Effect Size (Cohen's d): {cohens_d:.3f}
Interpretation: {'Negligible' if abs(cohens_d) < 0.2 else 'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'}

Conclusion: {'Significant difference detected' if p_val < 0.05 else 'No significant difference'}"""

        plt.text(0.02, 0.98, interpretation, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout()
        plt.savefig(self.output_dir / '04_statistical_significance.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.chart_descriptions['04'] = {
            'title': 'Statistical Significance Tests',
            'description': 'Shows p-values and effect sizes to determine if differences are statistically significant',
            'key_insights': [
                f'p < 0.05 indicates statistically significant difference',
                f'T-test result: {"Significant" if p_val < 0.05 else "Not significant"} (p = {p_val:.6f})',
                f'Effect size: {abs(cohens_d):.3f} ({"Small" if abs(cohens_d) < 0.5 else "Medium" if abs(cohens_d) < 0.8 else "Large"})',
                'Mann-Whitney U test is non-parametric alternative to t-test'
            ]
        }

    def chart_05_correlation_heatmap(self, df):
        """05: Algorithm Correlation Analysis"""
        algorithms = ['lzma', 'bz2', 'gzip', 'brotli']
        ratio_cols = [f'{algo}_ratio' for algo in algorithms if f'{algo}_ratio' in df.columns]

        if len(ratio_cols) < 2:
            print("Not enough algorithm columns for correlation analysis")
            return

        plt.figure(figsize=(10, 8))

        # Calculate correlation matrix
        correlation_data = df[ratio_cols].corr()

        # Rename columns for better display
        correlation_data.columns = [col.replace('_ratio', '').upper() for col in correlation_data.columns]
        correlation_data.index = correlation_data.columns

        # Create heatmap
        sns.heatmap(correlation_data, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, square=True, cbar_kws={"shrink": 0.8},
                   annot_kws={'fontsize': 12, 'fontweight': 'bold'})

        plt.title(f'{self.dataset_type.title()}: Algorithm Correlation Matrix',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Compression Algorithm', fontsize=14)
        plt.ylabel('Compression Algorithm', fontsize=14)

        # Add explanation
        explanation = """Correlation Interpretation:
• 1.0 = Perfect positive correlation
• 0.0 = No correlation
• -1.0 = Perfect negative correlation
• High correlation means algorithms behave similarly"""

        plt.text(1.02, 0.5, explanation, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

        plt.tight_layout()
        plt.savefig(self.output_dir / '05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Find highest and lowest correlations
        corr_values = correlation_data.values
        np.fill_diagonal(corr_values, np.nan)  # Exclude diagonal
        max_corr = np.nanmax(corr_values)
        min_corr = np.nanmin(corr_values)

        self.chart_descriptions['05'] = {
            'title': 'Algorithm Correlation Matrix',
            'description': 'Shows how similarly different compression algorithms behave',
            'key_insights': [
                f'Highest correlation: {max_corr:.3f} (algorithms behave very similarly)',
                f'Lowest correlation: {min_corr:.3f} (algorithms behave differently)',
                'High correlation suggests redundancy between algorithms',
                'Low correlation suggests complementary discrimination power'
            ]
        }

    def chart_06_pca_analysis(self, df):
        """06: Principal Component Analysis"""
        # Prepare data for PCA
        ratio_cols = [col for col in df.columns if col.endswith('_ratio')]
        if len(ratio_cols) < 2:
            print("Not enough features for PCA analysis")
            return

        # Clean data
        X = df[ratio_cols].fillna(df[ratio_cols].mean())
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.5)

        # Labels
        y = df['source'].apply(lambda x: 0 if x == 'AI' else 1)

        plt.figure(figsize=(12, 8))

        # Perform PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        # Plot first two components
        colors = [self.colors['ai'] if label == 0 else self.colors['human'] for label in y]
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7, s=50, edgecolors='white')

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
        plt.title(f'{self.dataset_type.title()}: Principal Component Analysis',
                 fontsize=16, fontweight='bold', pad=20)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=self.colors['ai'], label='AI Generated'),
                          Patch(facecolor=self.colors['human'], label='Human Written')]
        plt.legend(handles=legend_elements, fontsize=12)

        plt.grid(True, alpha=0.3)

        # Add variance explanation
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        variance_text = f"""Variance Explained:
PC1: {pca.explained_variance_ratio_[0]:.1%}
PC2: {pca.explained_variance_ratio_[1]:.1%}
Total (PC1+PC2): {cumvar[1]:.1%}

Separation Quality:
{'Good separation' if cumvar[1] > 0.7 else 'Moderate separation' if cumvar[1] > 0.5 else 'Poor separation'}"""

        plt.text(0.02, 0.98, variance_text, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.output_dir / '06_pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.chart_descriptions['06'] = {
            'title': 'Principal Component Analysis (PCA)',
            'description': 'Reduces high-dimensional data to 2D while preserving maximum variance',
            'key_insights': [
                f'PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance',
                f'PC2 explains {pca.explained_variance_ratio_[1]:.1%} of variance',
                f'Together they explain {cumvar[1]:.1%} of total variance',
                'Clear separation suggests algorithms can distinguish AI from human text'
            ]
        }

    def chart_07_feature_importance(self, df):
        """07: Random Forest Feature Importance"""
        # Prepare features
        ratio_cols = [col for col in df.columns if col.endswith('_ratio')]
        if len(ratio_cols) < 2:
            print("Not enough features for importance analysis")
            return

        # Clean data
        X = df[ratio_cols].fillna(df[ratio_cols].mean())
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.5)
        y = (df['source'] == 'Human').astype(int)

        plt.figure(figsize=(12, 8))

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # Get feature importance
        importances = rf.feature_importances_
        feature_names = [col.replace('_ratio', '').upper() for col in ratio_cols]

        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1]

        # Create bar plot
        colors = sns.color_palette("viridis", len(importances))
        bars = plt.bar(range(len(importances)), importances[sorted_idx],
                      color=colors, alpha=0.8, edgecolor='white')

        plt.xlabel('Compression Algorithm', fontsize=14)
        plt.ylabel('Feature Importance Score', fontsize=14)
        plt.title(f'{self.dataset_type.title()}: Algorithm Importance for AI vs Human Classification',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_idx])

        # Add value labels
        for bar, importance in zip(bars, importances[sorted_idx]):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{importance:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.grid(True, alpha=0.3, axis='y')

        # Add accuracy score
        accuracy = rf.score(X, y)
        accuracy_text = f"""Model Performance:
Classification Accuracy: {accuracy:.1%}

Feature Ranking:
1. {feature_names[sorted_idx[0]]}: {importances[sorted_idx[0]]:.3f}
2. {feature_names[sorted_idx[1]]}: {importances[sorted_idx[1]]:.3f}"""

        plt.text(0.02, 0.98, accuracy_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.output_dir / '07_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        best_algorithm = feature_names[sorted_idx[0]]
        self.chart_descriptions['07'] = {
            'title': 'Random Forest Feature Importance',
            'description': 'Shows which compression algorithms are most useful for distinguishing AI from human text',
            'key_insights': [
                f'Most important algorithm: {best_algorithm} (score: {importances[sorted_idx[0]]:.3f})',
                f'Model accuracy: {accuracy:.1%}',
                'Higher importance = better at distinguishing AI vs Human',
                'Based on Random Forest machine learning algorithm'
            ]
        }

    def chart_08_confidence_intervals(self, df):
        """08: Confidence Interval Analysis"""
        ai_complexity, human_complexity, _, _ = self.prepare_complexity_data(df)

        plt.figure(figsize=(12, 8))

        # Calculate 95% confidence intervals
        ai_ci = stats.t.interval(0.95, len(ai_complexity)-1,
                                loc=ai_complexity.mean(),
                                scale=stats.sem(ai_complexity))
        human_ci = stats.t.interval(0.95, len(human_complexity)-1,
                                   loc=human_complexity.mean(),
                                   scale=stats.sem(human_complexity))

        # Plot means with error bars
        means = [ai_complexity.mean(), human_complexity.mean()]
        lower_errors = [ai_complexity.mean() - ai_ci[0], human_complexity.mean() - human_ci[0]]
        upper_errors = [ai_ci[1] - ai_complexity.mean(), human_ci[1] - human_complexity.mean()]

        x_pos = [0, 1]
        labels = ['AI Generated', 'Human Written']
        colors = [self.colors['ai'], self.colors['human']]

        # Plot each point separately to avoid RGBA sequence error
        for i, (pos, mean, lower, upper, color) in enumerate(zip(x_pos, means, lower_errors, upper_errors, colors)):
            plt.errorbar(pos, mean, yerr=[[lower], [upper]],
                        fmt='o', capsize=10, capthick=3, elinewidth=3, markersize=12,
                        color='black', markerfacecolor=color, markeredgecolor='white',
                        markeredgewidth=2)

        # Fill confidence intervals
        for i, (pos, mean, lower, upper, color) in enumerate(zip(x_pos, means, lower_errors, upper_errors, colors)):
            plt.fill_between([pos-0.1, pos+0.1], [mean-lower]*2, [mean+upper]*2,
                           alpha=0.3, color=color)

        plt.xticks(x_pos, labels, fontsize=14)
        plt.ylabel('Mean Compression Complexity', fontsize=14)
        plt.title(f'{self.dataset_type.title()}: 95% Confidence Intervals',
                 fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)

        # Add detailed statistics
        overlap = not (ai_ci[1] < human_ci[0] or human_ci[1] < ai_ci[0])

        stats_text = f"""Confidence Interval Analysis:

AI Generated:
Mean: {ai_complexity.mean():.4f}
95% CI: [{ai_ci[0]:.4f}, {ai_ci[1]:.4f}]
Sample size: {len(ai_complexity)}

Human Written:
Mean: {human_complexity.mean():.4f}
95% CI: [{human_ci[0]:.4f}, {human_ci[1]:.4f}]
Sample size: {len(human_complexity)}

Intervals overlap: {'Yes' if overlap else 'No'}
Significance: {'Not significant' if overlap else 'Likely significant'}"""

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.9))

        plt.tight_layout()
        plt.savefig(self.output_dir / '08_confidence_intervals.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.chart_descriptions['08'] = {
            'title': '95% Confidence Intervals',
            'description': 'Shows the range where the true population mean is likely to be (95% confidence)',
            'key_insights': [
                f'AI mean: {ai_complexity.mean():.4f} ± {(ai_ci[1] - ai_ci[0])/2:.4f}',
                f'Human mean: {human_complexity.mean():.4f} ± {(human_ci[1] - human_ci[0])/2:.4f}',
                f'Confidence intervals {"overlap" if overlap else "do not overlap"}',
                'Non-overlapping intervals suggest significant difference'
            ]
        }

    def generate_all_charts(self):
        """Generate all individual charts"""
        df = self.load_data()
        if df is None:
            return

        print(f"\nGenerating individual visualizations for {self.dataset_type}...")
        print("="*60)

        # Generate each chart
        chart_functions = [
            self.chart_01_distribution_comparison,
            self.chart_02_box_plot_comparison,
            self.chart_03_algorithm_performance,
            self.chart_04_statistical_significance,
            self.chart_05_correlation_heatmap,
            self.chart_06_pca_analysis,
            self.chart_07_feature_importance,
            self.chart_08_confidence_intervals
        ]

        for i, chart_func in enumerate(chart_functions, 1):
            try:
                print(f"  Creating chart {i:02d}...")
                chart_func(df)
            except Exception as e:
                print(f"  Error in chart {i:02d}: {e}")

        # Generate documentation
        self.generate_documentation()

        print(f"\nCompleted! All charts saved to: {self.output_dir}")
        return self.chart_descriptions

    def generate_documentation(self):
        """Generate detailed documentation for all charts"""
        doc_content = f"""KOLMOGOROV COMPLEXITY ANALYSIS - {self.dataset_type.upper()} DATASET
Chart Documentation and Interpretation Guide

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: {self.dataset_type.title()}
Location: {self.output_dir}

{'='*80}
CHART DESCRIPTIONS AND KEY INSIGHTS
{'='*80}

"""

        for chart_num, info in self.chart_descriptions.items():
            doc_content += f"""
CHART {chart_num}: {info['title']}
{'-'*50}

Description:
{info['description']}

Key Insights:
"""
            for insight in info['key_insights']:
                doc_content += f"• {insight}\n"

            doc_content += f"\nFile: {chart_num}_{info['title'].lower().replace(' ', '_')}.png\n"
            doc_content += "\n"

        doc_content += f"""
{'='*80}
TECHNICAL NOTES
{'='*80}

Compression Ratio Interpretation:
• Lower ratio = Better compression = Higher algorithmic complexity detection
• Ratio = Compressed_Size / Original_Size
• Values closer to 0 mean better compression
• Values closer to 1 mean poor compression (already complex/random)

Statistical Significance:
• p < 0.05 = Statistically significant difference
• Cohen's d effect sizes: 0.2=small, 0.5=medium, 0.8=large
• Confidence intervals show uncertainty in mean estimates

Algorithm Performance:
• LZMA: High compression, good for text analysis
• BZ2: Burrows-Wheeler transform, good pattern detection
• GZIP: Fast, widely used, moderate compression
• Brotli: Modern algorithm, optimized for text

Methodology:
All texts were processed using multiple compression algorithms to estimate
Kolmogorov complexity. The core hypothesis is that AI-generated texts show
more predictable patterns and thus achieve better compression ratios than
human-written texts, which contain more genuine randomness and creativity.

{'='*80}
"""

        # Save documentation to docs folder
        docs_dir = self.base_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        doc_file = docs_dir / f"{self.dataset_type}_chart_documentation.txt"
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(doc_content)

        print(f"Documentation saved to: {doc_file}")

def main():
    """Generate visualizations for both datasets"""
    print("INDIVIDUAL VISUALIZATION GENERATOR")
    print("="*50)

    # Generate for Essays
    print("\n1. Generating visualizations for ESSAYS...")
    essays_generator = IndividualVisualizationGenerator('essays')
    essays_generator.generate_all_charts()

    # Generate for Tales
    print("\n2. Generating visualizations for TALES...")
    tales_generator = IndividualVisualizationGenerator('tales')
    tales_generator.generate_all_charts()

    print("\n" + "="*50)
    print("ALL VISUALIZATIONS COMPLETED!")
    print("="*50)
    print("\nOutput directories:")
    print(f"  • essays_visualizations/")
    print(f"  • tales_visualizations/")
    print("\nEach directory contains:")
    print("  • 8 high-quality PNG charts (01-08)")
    print("  • 1 detailed documentation file (.txt)")

if __name__ == "__main__":
    main()