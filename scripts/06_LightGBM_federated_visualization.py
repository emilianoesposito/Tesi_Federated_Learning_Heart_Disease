#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 06_federated_visualization.py
Description: Academic visualization using only real data from federated learning experiment
            No simulated or artificial data - only actual regional characteristics
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
FEDERATED_RESULTS_DIR = 'results_LightGBM_federated'
FEDERATED_DATA_DIR = 'data/federated'
VISUALIZATION_DIR = os.path.join(FEDERATED_RESULTS_DIR, 'visualizations')

# Academic plotting style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.linewidth': 1.2,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.major.size': 5,
    'ytick.minor.size': 3,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'text.usetex': False
})

def setup_academic_style():
    """Setup academic publication style with color palette."""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    colors = {
        'centralized': '#1f77b4',     # Blue
        'regional': '#ff7f0e',        # Orange  
        'federated': '#2ca02c',       # Green
        'performance': '#d62728',     # Red
        'neutral': '#7f7f7f',         # Gray
        'accent': '#9467bd',          # Purple
        'success': '#17becf',         # Cyan
        'warning': '#bcbd22'          # Olive
    }
    
    return colors

def load_real_data():
    """Load all real data from federated learning experiment."""
    logger.info("Loading real experimental data...")
    
    data = {}
    
    # Load performance comparison
    comparison_path = os.path.join(FEDERATED_RESULTS_DIR, 'complete_model_comparison.csv')
    if os.path.exists(comparison_path):
        data['comparison'] = pd.read_csv(comparison_path)
        logger.info(f"Loaded performance comparison data")
    
    # Load regional datasets
    regions = ['CPI_Verona', 'CPI_Vicenza', 'CPI_Padova', 'CPI_Treviso', 'CPI_Venezia']
    regional_data = {}
    
    for region in regions:
        filepath = os.path.join(FEDERATED_DATA_DIR, f"{region}_training_data.csv")
        if os.path.exists(filepath):
            regional_data[region] = pd.read_csv(filepath)
            logger.info(f"Loaded {region}: {len(regional_data[region])} samples")
        else:
            logger.warning(f"Regional data not found: {filepath}")
    
    data['regional_datasets'] = regional_data
    
    # Load experiment metadata
    metadata_path = os.path.join(FEDERATED_RESULTS_DIR, 'experiment_metadata.json')
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            data['metadata'] = json.load(f)
    
    return data

def analyze_regional_characteristics(regional_datasets):
    """Analyze real characteristics of each regional dataset."""
    logger.info("Analyzing real regional characteristics...")
    
    analysis = {}
    
    for region, dataset in regional_datasets.items():
        region_analysis = {
            'total_samples': len(dataset),
            'positive_rate': dataset['outcome'].mean(),
            'avg_attitude_score': dataset['attitude_score'].mean(),
            'avg_compatibility_score': dataset['compatibility_score'].mean(),
            'avg_distance_km': dataset['distance_km'].mean(),
            'avg_experience': dataset['years_experience'].mean()
        }
        
        # Analyze categorical distributions
        if 'education_level' in dataset.columns:
            region_analysis['education_dist'] = dataset['education_level'].value_counts(normalize=True)
        
        # Analyze disability types (look for columns starting with 'dis_')
        disability_cols = [col for col in dataset.columns if col.startswith('dis_')]
        if disability_cols:
            disability_dist = {}
            for col in disability_cols:
                disability_type = col.replace('dis_', '').replace('_', ' ').title()
                disability_dist[disability_type] = dataset[col].sum()
            region_analysis['disability_dist'] = disability_dist
        
        # Analyze sector distributions (look for columns starting with 'sector_')
        sector_cols = [col for col in dataset.columns if col.startswith('sector_')]
        if sector_cols:
            sector_dist = {}
            for col in sector_cols:
                sector_type = col.replace('sector_', '').replace('_', ' ').title()
                sector_dist[sector_type] = dataset[col].sum()
            region_analysis['sector_dist'] = sector_dist
        
        analysis[region] = region_analysis
    
    return analysis

def create_performance_comparison_real(comparison_df, colors):
    """Create performance comparison using real experimental results."""
    if comparison_df is None or comparison_df.empty:
        return
    
    logger.info("Creating real performance comparison visualization...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, height_ratios=[2, 1.5, 1.5], width_ratios=[2, 1.5, 1.5])
    
    # Main performance comparison
    ax_main = fig.add_subplot(gs[0, :])
    
    regional_data = comparison_df[comparison_df['Region'] != 'AVERAGE'].copy()
    avg_data = comparison_df[comparison_df['Region'] == 'AVERAGE'].iloc[0]
    
    # Performance metrics comparison
    metrics = ['Centralized_F1', 'Regional_F1', 'True_Federated_F1']
    metric_labels = ['Centralized', 'Regional', 'Federated']
    metric_colors = [colors['centralized'], colors['regional'], colors['federated']]
    
    x = np.arange(len(regional_data))
    width = 0.25
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
        offset = (i - 1) * width
        bars = ax_main.bar(x + offset, regional_data[metric], width, 
                          label=label, color=color, alpha=0.8, 
                          edgecolor='white', linewidth=1.5)
        
        # Add actual values on bars
        for bar in bars:
            height = bar.get_height()
            ax_main.text(bar.get_x() + bar.get_width()/2., height + 0.0002,
                        f'{height:.4f}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
    
    ax_main.set_xlabel('Employment Centers (CPI)', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
    ax_main.set_title('Real Performance Comparison: Three Learning Approaches', 
                     fontsize=16, fontweight='bold', pad=20)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels([r.replace('CPI_', '') for r in regional_data['Region']], 
                           fontsize=12, fontweight='bold')
    ax_main.legend(loc='upper right', fontsize=12)
    ax_main.grid(True, alpha=0.3, axis='y')
    
    # Calculate and display real statistical differences
    ax_stats = fig.add_subplot(gs[1, 0])
    
    real_differences = {
        'Fed - Cent': avg_data['True_Federated_F1'] - avg_data['Centralized_F1'],
        'Reg - Cent': avg_data['Regional_F1'] - avg_data['Centralized_F1'],
        'Fed - Reg': avg_data['True_Federated_F1'] - avg_data['Regional_F1']
    }
    
    diff_colors = [colors['federated'] if v >= 0 else colors['performance'] for v in real_differences.values()]
    bars = ax_stats.bar(real_differences.keys(), real_differences.values(), 
                       color=diff_colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    for bar, value in zip(bars, real_differences.values()):
        height = bar.get_height()
        y_pos = height + 0.00005 if height >= 0 else height - 0.00015
        ax_stats.text(bar.get_x() + bar.get_width()/2., y_pos,
                     f'{value:+.4f}', ha='center', va='bottom' if height >= 0 else 'top',
                     fontsize=11, fontweight='bold')
    
    ax_stats.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax_stats.set_ylabel('F1-Score Difference', fontsize=12, fontweight='bold')
    ax_stats.set_title('Real Performance Differences', fontsize=12, fontweight='bold')
    ax_stats.grid(True, alpha=0.3, axis='y')
    ax_stats.tick_params(axis='x', rotation=30)
    
    # Multi-metric real performance heatmap
    ax_heatmap = fig.add_subplot(gs[1, 1:])
    
    metrics_full = ['F1', 'Accuracy', 'ROC_AUC']
    model_types = ['Centralized', 'Regional', 'Federated']
    
    heatmap_data = []
    for model_type in model_types:
        row = []
        for metric in metrics_full:
            col_name = f'{model_type}_{metric}' if model_type != 'Federated' else f'True_Federated_{metric}'
            row.append(avg_data[col_name])
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data, index=model_types, columns=metrics_full)
    
    sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'Performance Score'}, ax=ax_heatmap,
                square=True, linewidths=1, cbar=True)
    ax_heatmap.set_title('Real Multi-Metric Performance', fontsize=12, fontweight='bold')
    ax_heatmap.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
    ax_heatmap.set_ylabel('Model Types', fontsize=12, fontweight='bold')
    
    # Real regional sample sizes
    ax_samples = fig.add_subplot(gs[2, :])
    
    test_samples = regional_data['Test_Samples'].values
    regions_clean = [r.replace('CPI_', '') for r in regional_data['Region']]
    
    bars = ax_samples.bar(regions_clean, test_samples, color=colors['neutral'], 
                         alpha=0.7, edgecolor='white', linewidth=1.5)
    
    for bar, samples in zip(bars, test_samples):
        height = bar.get_height()
        ax_samples.text(bar.get_x() + bar.get_width()/2., height + 50,
                       f'{samples:,}', ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
    
    ax_samples.set_xlabel('Employment Centers', fontsize=14, fontweight='bold')
    ax_samples.set_ylabel('Test Samples', fontsize=14, fontweight='bold')
    ax_samples.set_title('Real Regional Test Sample Distribution', fontsize=14, fontweight='bold')
    ax_samples.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'real_performance_comparison.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info("Created real performance comparison visualization")

def create_regional_characteristics_analysis(regional_analysis, colors):
    """Create comprehensive analysis of real regional characteristics."""
    logger.info("Creating real regional characteristics analysis...")
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1, 1])
    
    regions = list(regional_analysis.keys())
    regions_clean = [r.replace('CPI_', '') for r in regions]
    
    # 1. Sample sizes and positive rates
    ax1 = fig.add_subplot(gs[0, :2])
    
    sample_sizes = [regional_analysis[r]['total_samples'] for r in regions]
    positive_rates = [regional_analysis[r]['positive_rate'] for r in regions]
    
    ax1_twin = ax1.twinx()
    
    bars1 = ax1.bar([i-0.2 for i in range(len(regions))], sample_sizes, 0.4,
                   label='Total Samples', color=colors['neutral'], alpha=0.7)
    bars2 = ax1_twin.bar([i+0.2 for i in range(len(regions))], positive_rates, 0.4,
                        label='Success Rate', color=colors['success'], alpha=0.7)
    
    ax1.set_xlabel('Employment Centers', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Samples', fontsize=12, fontweight='bold', color=colors['neutral'])
    ax1_twin.set_ylabel('Success Rate', fontsize=12, fontweight='bold', color=colors['success'])
    ax1.set_title('Real Regional Sample Sizes and Success Rates', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(regions)))
    ax1.set_xticklabels(regions_clean)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, sample_sizes):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1000,
                f'{value:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar, value in zip(bars2, positive_rates):
        ax1_twin.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Real attitude and compatibility scores
    ax2 = fig.add_subplot(gs[0, 2:])
    
    attitude_scores = [regional_analysis[r]['avg_attitude_score'] for r in regions]
    compatibility_scores = [regional_analysis[r]['avg_compatibility_score'] for r in regions]
    
    x = np.arange(len(regions))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, attitude_scores, width, 
                   label='Attitude Score', color=colors['regional'], alpha=0.7)
    bars2 = ax2.bar(x + width/2, compatibility_scores, width,
                   label='Compatibility Score', color=colors['federated'], alpha=0.7)
    
    ax2.set_xlabel('Employment Centers', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax2.set_title('Real Attitude and Compatibility Scores', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions_clean)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Real disability type distributions
    ax3 = fig.add_subplot(gs[1, :2])
    
    # Get disability distributions from first region that has data
    disability_data = None
    for region in regions:
        if 'disability_dist' in regional_analysis[region]:
            disability_data = regional_analysis[region]['disability_dist']
            break
    
    if disability_data:
        disability_types = list(disability_data.keys())
        region_disability_counts = []
        
        for region in regions:
            if 'disability_dist' in regional_analysis[region]:
                counts = [regional_analysis[region]['disability_dist'].get(dt, 0) for dt in disability_types]
                region_disability_counts.append(counts)
            else:
                region_disability_counts.append([0] * len(disability_types))
        
        region_disability_counts = np.array(region_disability_counts).T
        
        # Create stacked bar chart
        bottom = np.zeros(len(regions))
        disability_colors = plt.cm.Set3(np.linspace(0, 1, len(disability_types)))
        
        for i, (disability_type, color) in enumerate(zip(disability_types, disability_colors)):
            ax3.bar(regions_clean, region_disability_counts[i], bottom=bottom,
                   label=disability_type, color=color, alpha=0.8)
            bottom += region_disability_counts[i]
        
        ax3.set_xlabel('Employment Centers', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
        ax3.set_title('Real Disability Type Distribution by Region', fontsize=14, fontweight='bold')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Real sector distributions
    ax4 = fig.add_subplot(gs[1, 2:])
    
    # Get sector distributions
    sector_data = None
    for region in regions:
        if 'sector_dist' in regional_analysis[region]:
            sector_data = regional_analysis[region]['sector_dist']
            break
    
    if sector_data:
        # Select top 6 sectors for visibility
        all_sectors = set()
        for region in regions:
            if 'sector_dist' in regional_analysis[region]:
                all_sectors.update(regional_analysis[region]['sector_dist'].keys())
        
        top_sectors = sorted(all_sectors)[:6]  # Limit to 6 for readability
        
        region_sector_counts = []
        for region in regions:
            if 'sector_dist' in regional_analysis[region]:
                counts = [regional_analysis[region]['sector_dist'].get(sector, 0) for sector in top_sectors]
                region_sector_counts.append(counts)
            else:
                region_sector_counts.append([0] * len(top_sectors))
        
        region_sector_counts = np.array(region_sector_counts).T
        
        # Create grouped bar chart
        x = np.arange(len(regions))
        width = 0.12
        sector_colors = plt.cm.Set2(np.linspace(0, 1, len(top_sectors)))
        
        for i, (sector, color) in enumerate(zip(top_sectors, sector_colors)):
            offset = (i - len(top_sectors)/2) * width
            ax4.bar(x + offset, region_sector_counts[i], width,
                   label=sector.replace('_', ' ').title(), color=color, alpha=0.8)
        
        ax4.set_xlabel('Employment Centers', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Number of Opportunities', fontsize=12, fontweight='bold')
        ax4.set_title('Real Sector Distribution by Region', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(regions_clean)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Real distance and experience analysis
    ax5 = fig.add_subplot(gs[2, :2])
    
    distances = [regional_analysis[r]['avg_distance_km'] for r in regions]
    experiences = [regional_analysis[r]['avg_experience'] for r in regions]
    
    # Scatter plot with regional labels
    scatter = ax5.scatter(distances, experiences, s=300, alpha=0.7, 
                         c=range(len(regions)), cmap='viridis', edgecolor='white', linewidth=2)
    
    for i, region_clean in enumerate(regions_clean):
        ax5.annotate(region_clean, (distances[i], experiences[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=11, fontweight='bold')
    
    ax5.set_xlabel('Average Distance (km)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Average Experience (years)', fontsize=12, fontweight='bold')
    ax5.set_title('Real Distance vs Experience by Region', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Real performance correlation with characteristics
    ax6 = fig.add_subplot(gs[2, 2:])
    
    # Get F1 scores for correlation analysis (need to load comparison data)
    characteristics = ['Sample Size', 'Success Rate', 'Avg Distance', 'Avg Experience']
    char_values = [
        sample_sizes,
        positive_rates, 
        distances,
        experiences
    ]
    
    # Create correlation matrix visualization
    char_matrix = np.array(char_values).T
    char_df = pd.DataFrame(char_matrix, columns=characteristics, index=regions_clean)
    
    correlation_matrix = char_df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f', cbar_kws={'label': 'Correlation'},
                ax=ax6, linewidths=0.5)
    ax6.set_title('Real Regional Characteristics Correlation', fontsize=14, fontweight='bold')
    
    # 7. Box plots for score distributions
    ax7 = fig.add_subplot(gs[3, :2])
    
    score_data = []
    score_labels = []
    
    for region in regions:
        region_clean = region.replace('CPI_', '')
        score_data.extend([
            regional_analysis[region]['avg_attitude_score'],
            regional_analysis[region]['avg_compatibility_score']
        ])
        score_labels.extend([f'{region_clean}\nAttitude', f'{region_clean}\nCompatibility'])
    
    # Reshape for box plot
    attitude_scores_all = [regional_analysis[r]['avg_attitude_score'] for r in regions]
    compatibility_scores_all = [regional_analysis[r]['avg_compatibility_score'] for r in regions]
    
    box_data = [attitude_scores_all, compatibility_scores_all]
    box_labels = ['Attitude Scores', 'Compatibility Scores']
    
    bp = ax7.boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor(colors['regional'])
    bp['boxes'][1].set_facecolor(colors['federated'])
    
    ax7.set_ylabel('Score Value', fontsize=12, fontweight='bold')
    ax7.set_title('Real Score Distribution Across All Regions', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Regional efficiency metrics
    ax8 = fig.add_subplot(gs[3, 2:])
    
    # Calculate efficiency as success rate / (distance + 1) to avoid division by zero
    efficiency_scores = []
    for region in regions:
        efficiency = (regional_analysis[region]['positive_rate'] / 
                     (regional_analysis[region]['avg_distance_km'] + 1))
        efficiency_scores.append(efficiency)
    
    bars = ax8.bar(regions_clean, efficiency_scores, color=colors['accent'], alpha=0.7,
                   edgecolor='white', linewidth=1.5)
    
    for bar, efficiency in zip(bars, efficiency_scores):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{efficiency:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    ax8.set_xlabel('Employment Centers', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Efficiency Score', fontsize=12, fontweight='bold')
    ax8.set_title('Real Regional Efficiency (Success Rate / Distance)', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'real_regional_characteristics.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info("Created real regional characteristics analysis")

def create_real_data_summary_dashboard(comparison_df, regional_analysis, colors):
    """Create comprehensive summary dashboard with real data only (no overlapping labels)."""
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator
    from textwrap import wrap

    try:
        import seaborn as sns
        HAS_SNS = True
    except Exception:
        HAS_SNS = False

    def _get_color(name, default): return colors.get(name, default)
    def _wrap_text(s, w): return "\n".join(wrap(s, width=w))
    def _rotate_xticks(ax, rot=20):
        for t in ax.get_xticklabels():
            t.set_rotation(rot); t.set_ha("right")
    def _wrap_xticks(ax, width=14):
        lbls = [_wrap_text(t.get_text(), width) for t in ax.get_xticklabels()]
        ax.set_xticklabels(lbls); plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    def _pad_ylim(ax, vals, frac=0.18, min_span=1e-3):
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        span = max(vmax - vmin, min_span)
        ax.set_ylim(vmin - frac*span, vmax + frac*span)
    def _autolabel(ax, bars, fmt="{:.3f}", inside=True):
        ymin, ymax = ax.get_ylim()
        for b in bars:
            v = b.get_height(); x = b.get_x() + b.get_width()/2
            if inside and v > ymin + 0.25*(ymax-ymin):
                ax.text(x, v - 0.03*(ymax-ymin), fmt.format(v),
                        ha="center", va="top", fontsize=11, fontweight="bold", color="white")
            else:
                ax.text(x, v + 0.01*(ymax-ymin), fmt.format(v),
                        ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})
    fig = plt.figure(figsize=(26, 18), constrained_layout=True)
 
    gs = GridSpec(5, 6, figure=fig,
                  height_ratios=[0.55, 2.2, 2.2, 2.9, 1.5],
                  width_ratios=[1, 1, 1, 1, 1, 1])

    # --- Title ---
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.text(0.5, 0.72,
                  'Real Data Analysis: Federated Learning for Disability Employment Matching',
                  ha='center', va='center', fontsize=26, fontweight='bold', transform=ax_title.transAxes)
    ax_title.text(0.5, 0.22,
                  'Veneto Region Employment Centers • Real Experimental Results • No Simulated Data',
                  ha='center', va='center', fontsize=16, color='gray', style='italic',
                  transform=ax_title.transAxes)
    ax_title.axis('off')

    # --- Top row  ---
    if comparison_df is not None and not comparison_df.empty:
        avg_row = comparison_df[comparison_df['Region'] == 'AVERAGE'].iloc[0]
        reg_df = comparison_df[comparison_df['Region'] != 'AVERAGE'].copy()

        ax_perf = fig.add_subplot(gs[1, :2])
        labels = ['Centralized', 'Regional', 'Federated']
        vals = [avg_row['Centralized_F1'], avg_row['Regional_F1'], avg_row['True_Federated_F1']]
        cols = [_get_color('centralized', '#4472c4'),
                _get_color('regional', '#ed7d31'),
                _get_color('federated', '#2ca02c')]
        bars = ax_perf.bar(labels, vals, color=cols, alpha=0.85, edgecolor='white', linewidth=1.8)
        _pad_ylim(ax_perf, vals, 0.25, 0.0015); _autolabel(ax_perf, bars, "{:.4f}", inside=False)
        ax_perf.set_ylabel('F1-Score', fontweight='bold'); ax_perf.set_title('Real Average Performance Results', fontweight='bold')
        ax_perf.grid(True, alpha=0.25, axis='y'); ax_perf.yaxis.set_major_locator(MaxNLocator(4))

        ax_var = fig.add_subplot(gs[1, 2:4])
        regions_clean = [r.replace('CPI_', '') for r in reg_df['Region']]
        x = np.arange(len(regions_clean)); w = 0.27
        bars_c = ax_var.bar(x - w, reg_df['Centralized_F1'].values, w, label='Centralized',
                            color=_get_color('centralized', '#4472c4'), alpha=0.8)
        bars_r = ax_var.bar(x,       reg_df['Regional_F1'].values,    w, label='Regional',
                            color=_get_color('regional', '#ed7d31'), alpha=0.8)
        bars_f = ax_var.bar(x + w, reg_df['True_Federated_F1'].values, w, label='Federated',
                            color=_get_color('federated', '#2ca02c'), alpha=0.8)
        _pad_ylim(ax_var, np.r_[reg_df['Centralized_F1'], reg_df['Regional_F1'], reg_df['True_Federated_F1']], 0.22)
        ax_var.set_ylabel('F1-Score', fontweight='bold'); ax_var.set_title('Real Regional Performance Comparison', fontweight='bold')
        ax_var.set_xticks(x); ax_var.set_xticklabels(regions_clean); _rotate_xticks(ax_var, 18)
        ax_var.legend(frameon=False, ncols=3, loc='upper left'); ax_var.grid(True, alpha=0.25, axis='y')
        ax_var.yaxis.set_major_locator(MaxNLocator(5))

        ax_samp = fig.add_subplot(gs[1, 4:])
        samples = reg_df['Test_Samples'].values
        b = ax_samp.bar(regions_clean, samples, color=_get_color('neutral', '#9aa0a6'),
                        alpha=0.85, edgecolor='white', linewidth=1.5)
        _rotate_xticks(ax_samp, 18)
        ax_samp.set_ylabel('Test Samples', fontweight='bold'); ax_samp.set_title('Real Test Sample Sizes', fontweight='bold')
        ax_samp.grid(True, alpha=0.25, axis='y'); ax_samp.yaxis.set_major_locator(MaxNLocator(5))
        pad = (max(samples)-min(samples) if len(samples)>1 else max(samples))*0.02 + 120
        for bi, v in zip(b, samples):
            ax_samp.text(bi.get_x()+bi.get_width()/2, bi.get_height()+pad, f"{int(v):,}",
                         ha='center', va='bottom', fontsize=10, fontweight='bold', clip_on=False)

    # --- Middle row ---
    regions = list(regional_analysis.keys())
    regions_clean_all = [r.replace('CPI_', '') for r in regions]

    ax_scores = fig.add_subplot(gs[2, :2])
    att = [regional_analysis[r]['avg_attitude_score'] for r in regions]
    comp = [regional_analysis[r]['avg_compatibility_score'] for r in regions]
    x = np.arange(len(regions)); w = 0.36
    b1 = ax_scores.bar(x - w/2, att,  w, label='Attitude Score',      color=_get_color('regional', '#ed7d31'), alpha=0.8)
    b2 = ax_scores.bar(x + w/2, comp, w, label='Compatibility Score', color=_get_color('federated', '#2ca02c'), alpha=0.8)
    _pad_ylim(ax_scores, np.r_[att, comp], 0.18); _autolabel(ax_scores, b1, "{:.3f}", True); _autolabel(ax_scores, b2, "{:.3f}", True)
    ax_scores.set_ylabel('Average Score', fontweight='bold'); ax_scores.set_title('Real Attitude & Compatibility Scores', fontweight='bold')
    ax_scores.set_xticks(x); ax_scores.set_xticklabels(regions_clean_all); _rotate_xticks(ax_scores, 12)
    ax_scores.legend(frameon=False, ncols=2, loc='upper left'); ax_scores.grid(True, alpha=0.25, axis='y')

    ax_dist = fig.add_subplot(gs[2, 2:4])
    distances = [regional_analysis[r]['avg_distance_km'] for r in regions]
    experiences = [regional_analysis[r]['avg_experience'] for r in regions]
    sc = ax_dist.scatter(distances, experiences, s=360, alpha=0.8, c=np.arange(len(regions)),
                         cmap='viridis', edgecolor='white', linewidth=1.8)
    offsets = [(8,10),(-10,10),(10,-10),(-10,-10),(12,6),(-12,6),(6,-12),(-6,-12)]
    for i, name in enumerate(regions_clean_all):
        dx, dy = offsets[i % len(offsets)]
        ax_dist.annotate(name, (distances[i], experiences[i]),
                         xytext=(dx, dy), textcoords='offset points',
                         fontsize=11, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.25", facecolor='white', alpha=0.85), zorder=5)
    ax_dist.set_xlabel('Average Distance (km)', fontweight='bold')
    ax_dist.set_ylabel('Average Experience (years)', fontweight='bold')
    ax_dist.set_title('Real Distance vs Experience', fontweight='bold')
    ax_dist.grid(True, alpha=0.25); ax_dist.xaxis.set_major_locator(MaxNLocator(5)); ax_dist.yaxis.set_major_locator(MaxNLocator(5))

    ax_succ = fig.add_subplot(gs[2, 4:])
    success = [regional_analysis[r]['positive_rate'] for r in regions]
    bb = ax_succ.bar(regions_clean_all, success, color=_get_color('success', '#1f77b4'),
                     alpha=0.85, edgecolor='white', linewidth=1.5)
    _pad_ylim(ax_succ, success, 0.18); _autolabel(ax_succ, bb, "{:.3f}", True)
    ax_succ.set_ylabel('Success Rate', fontweight='bold'); ax_succ.set_title('Real Employment Success Rates', fontweight='bold')
    ax_succ.set_xticks(np.arange(len(regions_clean_all))); ax_succ.set_xticklabels(regions_clean_all); _rotate_xticks(ax_succ, 12)
    ax_succ.grid(True, alpha=0.25, axis='y'); ax_succ.yaxis.set_major_locator(MaxNLocator(5))

    # --- THIRD ROW: heatmaps fixed with subgridspec + wrapped tick labels ---
    # Left heatmap (disability) with own colorbar axis
    left_gs = gs[3, :3].subgridspec(1, 10, wspace=0.05)
    ax_dis = fig.add_subplot(left_gs[:, :9])
    ax_dis_cbar = fig.add_subplot(left_gs[:, 9])

    has_dis = any('disability_dist' in regional_analysis[r] for r in regions)
    if has_dis:
        all_types = set()
        for r in regions:
            all_types.update(regional_analysis[r].get('disability_dist', {}).keys())
        types = sorted(list(all_types))[:8]
        mat = [[regional_analysis[r].get('disability_dist', {}).get(t, 0) for t in types] for r in regions]
        df_dis = pd.DataFrame(mat, index=regions_clean_all,
                              columns=[t.replace('_', ' ').title() for t in types])
        if HAS_SNS:
            annot = np.array([[f"{int(v):,}" for v in row] for row in df_dis.values])
            sns.heatmap(df_dis, ax=ax_dis, cbar_ax=ax_dis_cbar, cmap='Blues',
                        annot=annot, fmt="", annot_kws={"size": 9})
        else:
            im = ax_dis.imshow(df_dis.values, aspect='auto'); fig.colorbar(im, cax=ax_dis_cbar)
            ax_dis.set_xticks(range(len(df_dis.columns))); ax_dis.set_xticklabels(df_dis.columns)
            ax_dis.set_yticks(range(len(df_dis.index)));  ax_dis.set_yticklabels(df_dis.index)
        ax_dis.set_title('Real Disability Type Distribution by Region', fontweight='bold')
        ax_dis.set_xlabel('Disability Types', fontweight='bold'); ax_dis.set_ylabel('Employment Centers', fontweight='bold')
        _wrap_xticks(ax_dis, width=16)
    else:
        ax_dis.text(0.5, 0.5, 'Disability distribution data\nnot available in current dataset',
                    ha='center', va='center', fontsize=14, style='italic', transform=ax_dis.transAxes)
        ax_dis.set_title('Disability Distribution Analysis', fontweight='bold'); ax_dis.set_xticks([]); ax_dis.set_yticks([])
        ax_dis_cbar.axis('off')

    # Right heatmap (sectors) with own colorbar axis
    right_gs = gs[3, 3:].subgridspec(1, 10, wspace=0.05)
    ax_sec = fig.add_subplot(right_gs[:, :9])
    ax_sec_cbar = fig.add_subplot(right_gs[:, 9])

    has_sec = any('sector_dist' in regional_analysis[r] for r in regions)
    if has_sec:
        all_sectors = set()
        for r in regions:
            all_sectors.update(regional_analysis[r].get('sector_dist', {}).keys())
        totals = {s: sum(regional_analysis[r].get('sector_dist', {}).get(s, 0) for r in regions) for s in all_sectors}
        top_sectors = [s for s, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:6]]
        mat = [[regional_analysis[r].get('sector_dist', {}).get(s, 0) for s in top_sectors] for r in regions]
        df_sec = pd.DataFrame(mat, index=regions_clean_all,
                              columns=[s.replace('_', ' ').title() for s in top_sectors])
        if HAS_SNS:
            annot = np.array([[f"{int(v):,}" for v in row] for row in df_sec.values])
            sns.heatmap(df_sec, ax=ax_sec, cbar_ax=ax_sec_cbar, cmap='Oranges',
                        annot=annot, fmt="", annot_kws={"size": 9})
        else:
            im = ax_sec.imshow(df_sec.values, aspect='auto'); fig.colorbar(im, cax=ax_sec_cbar)
            ax_sec.set_xticks(range(len(df_sec.columns))); ax_sec.set_xticklabels(df_sec.columns)
            ax_sec.set_yticks(range(len(df_sec.index)));  ax_sec.set_yticklabels(df_sec.index)
        ax_sec.set_title('Real Sector Distribution by Region', fontweight='bold')
        ax_sec.set_xlabel('Economic Sectors', fontweight='bold'); ax_sec.set_ylabel('Employment Centers', fontweight='bold')
        _wrap_xticks(ax_sec, width=16)
    else:
        ax_sec.text(0.5, 0.5, 'Sector distribution data\nnot available in current dataset',
                    ha='center', va='center', fontsize=14, style='italic', transform=ax_sec.transAxes)
        ax_sec.set_title('Sector Distribution Analysis', fontweight='bold'); ax_sec.set_xticks([]); ax_sec.set_yticks([])
        ax_sec_cbar.axis('off')

    # --- Bottom: findings ---
    ax_find = fig.add_subplot(gs[4, :])
    try:
        perf_diff = (avg_row['True_Federated_F1'] - avg_row['Centralized_F1']) if comparison_df is not None else 0.0
        fed_f1 = avg_row['True_Federated_F1']; cen_f1 = avg_row['Centralized_F1']
    except Exception:
        perf_diff = fed_f1 = cen_f1 = 0.0
    success_rates = [regional_analysis[r]['positive_rate'] for r in regions] if regions else [0]
    avg_success = float(np.mean(success_rates)) if success_rates else 0.0
    total_samples = sum(regional_analysis[r].get('total_samples', 0) for r in regions)
    findings = [
        f"Real Performance: Federated learning achieves {perf_diff:+.4f} F1-score difference vs centralized (Fed: {fed_f1:.4f}, Cent: {cen_f1:.4f})",
        f"Real Data Scale: {total_samples:,} total training samples across {len(regions)} employment centers with {avg_success:.1%} average success rate",
        f"Real Geographic Coverage: Average distance {np.mean(distances):.1f} km, experience range {min(experiences):.1f}-{max(experiences):.1f} years",
        "Real Implementation: No simulated data — all results from actual regional employment center characteristics and distributions",
    ]
    for i, line in enumerate(findings):
        ax_find.text(0.02, 0.9 - i*0.22, "• " + _wrap_text(line, 150),
                     fontsize=12, fontweight='bold', va='top', transform=ax_find.transAxes)
    ax_find.set_title('Key Findings from Real Data Analysis', fontsize=16, fontweight='bold', loc='left', pad=6)
    ax_find.axis('off')

    out_path = os.path.join(VISUALIZATION_DIR, 'real_data_comprehensive_dashboard.png')
    fig.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight'); plt.close(fig)
    logger.info("Created real data comprehensive dashboard without overlapping labels.")



def create_federated_vs_centralized_real_analysis(comparison_df, colors):
    """Create detailed comparison between federated and centralized approaches using real data."""
    logger.info("Creating real federated vs centralized analysis...")
    
    if comparison_df is None or comparison_df.empty:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    regional_data = comparison_df[comparison_df['Region'] != 'AVERAGE'].copy()
    avg_data = comparison_df[comparison_df['Region'] == 'AVERAGE'].iloc[0]
    
    # Real performance scatter plot
    centralized_scores = regional_data['Centralized_F1'].values
    federated_scores = regional_data['True_Federated_F1'].values
    regions_clean = [r.replace('CPI_', '') for r in regional_data['Region']]
    
    ax1.scatter(centralized_scores, federated_scores, s=300, alpha=0.7,
               c=range(len(centralized_scores)), cmap='viridis', 
               edgecolor='white', linewidth=2)
    
    # Add diagonal line for reference
    min_score = min(min(centralized_scores), min(federated_scores))
    max_score = max(max(centralized_scores), max(federated_scores))
    ax1.plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.7, linewidth=2)
    
    # Label points
    for i, region in enumerate(regions_clean):
        ax1.annotate(region, (centralized_scores[i], federated_scores[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Centralized F1-Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Federated F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('Real Performance: Federated vs Centralized', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, 'Points above diagonal:\nFederated > Centralized', 
            transform=ax1.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # Real improvement distribution
    improvements = federated_scores - centralized_scores
    
    ax2.hist(improvements, bins=6, color=colors['federated'], alpha=0.7,
            edgecolor='white', linewidth=1.5)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax2.axvline(x=improvements.mean(), color='darkgreen', linestyle='-', linewidth=2,
               label=f'Mean: {improvements.mean():.4f}')
    
    ax2.set_xlabel('F1-Score Improvement (Federated - Centralized)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Regions', fontsize=12, fontweight='bold')
    ax2.set_title('Real Improvement Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Real multi-metric comparison
    metrics = ['F1', 'Accuracy', 'ROC_AUC']
    centralized_values = [avg_data['Centralized_F1'], avg_data['Centralized_Accuracy'], avg_data['Centralized_ROC_AUC']]
    federated_values = [avg_data['True_Federated_F1'], avg_data['True_Federated_Accuracy'], avg_data['True_Federated_ROC_AUC']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, centralized_values, width, label='Centralized',
                   color=colors['centralized'], alpha=0.7)
    bars2 = ax3.bar(x + width/2, federated_values, width, label='Federated',
                   color=colors['federated'], alpha=0.7)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.4f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('Real Multi-Metric Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Real statistical significance test visualization
    from scipy import stats
    
    # Perform paired t-test on real data
    t_stat, p_value = stats.ttest_rel(federated_scores, centralized_scores)
    
    # Effect size (Cohen's d)
    cohen_d = (federated_scores.mean() - centralized_scores.mean()) / np.std(federated_scores - centralized_scores)
    
    # Visualization of statistical test
    ax4.bar(['T-Statistic', 'P-Value', 'Effect Size'], 
           [t_stat, p_value, cohen_d],
           color=[colors['accent'], colors['warning'], colors['success']], alpha=0.7)
    
    # Add significance threshold lines
    ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05 threshold')
    ax4.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Small effect size')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    
    # Add value labels
    values = [t_stat, p_value, cohen_d]
    labels = ['T-Statistic', 'P-Value', 'Effect Size']
    for i, (value, label) in enumerate(zip(values, labels)):
        ax4.text(i, value + 0.01, f'{value:.4f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    ax4.set_ylabel('Statistical Measure', fontsize=12, fontweight='bold')
    ax4.set_title('Real Statistical Significance Analysis', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add interpretation text
    interpretation = "Non-significant" if p_value > 0.05 else "Significant"
    effect_interpretation = "Negligible" if abs(cohen_d) < 0.2 else "Small" if abs(cohen_d) < 0.5 else "Medium"
    
    ax4.text(0.02, 0.98, f'Result: {interpretation} (p={p_value:.4f})\nEffect: {effect_interpretation} (d={cohen_d:.4f})',
            transform=ax4.transAxes, va='top', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'real_federated_vs_centralized_analysis.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info("Created real federated vs centralized analysis")

def main():
    """Main pipeline for real data visualization."""
    logger.info("Starting real data visualization pipeline")
    logger.info("=" * 60)
    
    try:
        # Setup
        colors = setup_academic_style()
        
        # Load all real data
        data = load_real_data()
        
        if not data:
            logger.error("No real data found")
            return
        
        # Analyze regional characteristics from real data
        regional_analysis = {}
        if 'regional_datasets' in data and data['regional_datasets']:
            regional_analysis = analyze_regional_characteristics(data['regional_datasets'])
        
        comparison_df = data.get('comparison')
        
        # Generate visualizations using only real data
        if comparison_df is not None:
            create_performance_comparison_real(comparison_df, colors)
            create_federated_vs_centralized_real_analysis(comparison_df, colors)
        
        if regional_analysis:
            create_regional_characteristics_analysis(regional_analysis, colors)
            
        if comparison_df is not None and regional_analysis:
            create_real_data_summary_dashboard(comparison_df, regional_analysis, colors)
        
        logger.info("=" * 60)
        logger.info("Real data visualization pipeline completed successfully")
        logger.info(f"Visualizations saved to: {VISUALIZATION_DIR}")
        logger.info("Generated real data figures:")
        logger.info("  - real_performance_comparison.png")
        logger.info("  - real_regional_characteristics.png")
        logger.info("  - real_data_comprehensive_dashboard.png")
        logger.info("  - real_federated_vs_centralized_analysis.png")
        logger.info("All visualizations use only real experimental data - no simulations")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Real data visualization pipeline failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()