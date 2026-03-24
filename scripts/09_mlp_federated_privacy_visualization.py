#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 09_mlp_federated_privacy_visualization.py
Description: Comprehensive visualization and comparison of classical vs privacy-preserving
             federated learning results. Analyzes performance trade-offs, privacy costs,
             training progression, and provides detailed comparative analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CLASSICAL_RESULTS_DIR = 'results_mlp_federated'
PRIVACY_RESULTS_DIR = 'results_mlp_federated_privacy'
VISUALIZATION_OUTPUT_DIR = 'visualizations_mlp_federated_comparison'
CENTRALIZED_RESULTS_DIR = 'results'

# Visualization settings
plt.style.use('default')  # Use default style for better compatibility
sns.set_palette("husl")
FIGURE_SIZE = (15, 10)
DPI = 150

class FederatedResultsAnalyzer:
    """
    Comprehensive analyzer for federated learning experiments comparing
    classical and privacy-preserving approaches.
    """
    
    def __init__(self):
        self.classical_data = {}
        self.privacy_data = {}
        self.centralized_data = {}
        
        # Create output directory
        os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
        
        logger.info("Initialized Federated Results Analyzer")
    
    def check_and_list_files(self) -> None:
        """Check what files are available in both directories."""
        logger.info(f"Checking files in {CLASSICAL_RESULTS_DIR}:")
        if os.path.exists(CLASSICAL_RESULTS_DIR):
            for file in os.listdir(CLASSICAL_RESULTS_DIR):
                logger.info(f"  - {file}")
        else:
            logger.warning(f"Directory {CLASSICAL_RESULTS_DIR} does not exist")
        
        logger.info(f"Checking files in {PRIVACY_RESULTS_DIR}:")
        if os.path.exists(PRIVACY_RESULTS_DIR):
            for file in os.listdir(PRIVACY_RESULTS_DIR):
                logger.info(f"  - {file}")
        else:
            logger.warning(f"Directory {PRIVACY_RESULTS_DIR} does not exist")
    
    def load_classical_results(self) -> bool:
        """Load classical federated learning results with detailed logging."""
        try:
            logger.info(f"Loading classical results from {CLASSICAL_RESULTS_DIR}")
            
            # Performance results
            perf_path = os.path.join(CLASSICAL_RESULTS_DIR, 'mlp_federated_performance.csv')
            if os.path.exists(perf_path):
                self.classical_data['performance'] = pd.read_csv(perf_path)
                logger.info(f"Loaded classical performance data: {len(self.classical_data['performance'])} regions")
                logger.info(f"Performance columns: {list(self.classical_data['performance'].columns)}")
                logger.info(f"Regions: {self.classical_data['performance']['Region'].tolist()}")
            else:
                logger.warning(f"Performance file not found: {perf_path}")
            
            # Training history
            history_path = os.path.join(CLASSICAL_RESULTS_DIR, 'federated_training_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.classical_data['history'] = json.load(f)
                logger.info(f"Loaded classical training history: {len(self.classical_data['history'])} rounds")
            else:
                logger.warning(f"History file not found: {history_path}")
            
            # Improvement summary
            improvement_path = os.path.join(CLASSICAL_RESULTS_DIR, 'federated_improvement_summary.csv')
            if os.path.exists(improvement_path):
                self.classical_data['improvement'] = pd.read_csv(improvement_path)
                logger.info(f"Loaded classical improvement summary: {len(self.classical_data['improvement'])} rounds")
            else:
                logger.warning(f"Improvement file not found: {improvement_path}")
            
            # Configuration
            config_path = os.path.join(CLASSICAL_RESULTS_DIR, 'mlp_federated_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.classical_data['config'] = json.load(f)
                logger.info("Loaded classical configuration")
            else:
                logger.warning(f"Config file not found: {config_path}")
            
            return len(self.classical_data) > 0
            
        except Exception as e:
            logger.error(f"Failed to load classical results: {e}")
            return False
    
    def load_privacy_results(self) -> bool:
        """Load privacy-preserving federated learning results with detailed logging."""
        try:
            logger.info(f"Loading privacy results from {PRIVACY_RESULTS_DIR}")
            
            # Performance results
            perf_path = os.path.join(PRIVACY_RESULTS_DIR, 'mlp_federated_privacy_performance.csv')
            if os.path.exists(perf_path):
                self.privacy_data['performance'] = pd.read_csv(perf_path)
                logger.info(f"Loaded privacy performance data: {len(self.privacy_data['performance'])} regions")
                logger.info(f"Performance columns: {list(self.privacy_data['performance'].columns)}")
                logger.info(f"Regions: {self.privacy_data['performance']['Region'].tolist()}")
            else:
                logger.warning(f"Performance file not found: {perf_path}")
            
            # Training history
            history_path = os.path.join(PRIVACY_RESULTS_DIR, 'federated_privacy_training_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.privacy_data['history'] = json.load(f)
                logger.info(f"Loaded privacy training history: {len(self.privacy_data['history'])} rounds")
            else:
                logger.warning(f"History file not found: {history_path}")
            
            # Improvement summary
            improvement_path = os.path.join(PRIVACY_RESULTS_DIR, 'federated_privacy_improvement_summary.csv')
            if os.path.exists(improvement_path):
                self.privacy_data['improvement'] = pd.read_csv(improvement_path)
                logger.info(f"Loaded privacy improvement summary: {len(self.privacy_data['improvement'])} rounds")
            else:
                logger.warning(f"Improvement file not found: {improvement_path}")
            
            # Configuration
            config_path = os.path.join(PRIVACY_RESULTS_DIR, 'mlp_federated_privacy_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.privacy_data['config'] = json.load(f)
                logger.info("Loaded privacy configuration")
            else:
                logger.warning(f"Config file not found: {config_path}")
            
            # Delta norm statistics (if available)
            delta_path = os.path.join(PRIVACY_RESULTS_DIR, 'delta_norm_statistics.json')
            if os.path.exists(delta_path):
                with open(delta_path, 'r') as f:
                    self.privacy_data['delta_norms'] = json.load(f)
                logger.info("Loaded delta norm statistics")
            else:
                logger.info("Delta norm statistics not available")
            
            return len(self.privacy_data) > 0
            
        except Exception as e:
            logger.error(f"Failed to load privacy results: {e}")
            return False
    
    def load_centralized_baseline(self) -> bool:
        """Load centralized baseline for comparison."""
        try:
            # Centralized metrics
            metrics_path = os.path.join(CENTRALIZED_RESULTS_DIR, 'metrics_summary.csv')
            if os.path.exists(metrics_path):
                metrics_df = pd.read_csv(metrics_path, index_col=0)
                if 'MLP_Optimized' in metrics_df.index:
                    self.centralized_data['metrics'] = {
                        'f1_score': metrics_df.loc['MLP_Optimized', 'f1_score'],
                        'accuracy': metrics_df.loc['MLP_Optimized', 'accuracy'],
                        'roc_auc': metrics_df.loc['MLP_Optimized', 'roc_auc']
                    }
                    logger.info("Loaded centralized baseline metrics")
                else:
                    logger.warning("MLP_Optimized not found in metrics summary")
            else:
                logger.warning(f"Centralized metrics file not found: {metrics_path}")
            
            return len(self.centralized_data) > 0
            
        except Exception as e:
            logger.error(f"Failed to load centralized baseline: {e}")
            return False
    
    def create_performance_comparison(self) -> None:
        """Create comprehensive performance comparison plots with better error handling."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
            fig.suptitle('Federated Learning Performance Comparison\nClassical vs Privacy-Preserving', 
                         fontsize=16, fontweight='bold')
            
            # Check if we have the required data
            classical_perf = self.classical_data.get('performance')
            privacy_perf = self.privacy_data.get('performance')
            
            if classical_perf is None or privacy_perf is None:
                # Create error message plot
                for ax in axes.flat:
                    ax.text(0.5, 0.5, 'Performance data not available\nCheck file paths and data loading', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                    ax.set_title('Data Loading Error')
                
                plt.tight_layout()
                plt.savefig(os.path.join(VISUALIZATION_OUTPUT_DIR, 'performance_comparison_error.png'), 
                           dpi=DPI, bbox_inches='tight')
                plt.show()
                return
            
            # Get regional data (exclude AVERAGE and WEIGHTED_AVERAGE)
            classical_regional = classical_perf[~classical_perf['Region'].isin(['AVERAGE', 'WEIGHTED_AVERAGE'])]
            privacy_regional = privacy_perf[~privacy_perf['Region'].isin(['AVERAGE', 'WEIGHTED_AVERAGE'])]
            
            if len(classical_regional) == 0 or len(privacy_regional) == 0:
                logger.warning("No regional data found for comparison")
                return
            
            # Get matching regions
            classical_regions = set(classical_regional['Region'].tolist())
            privacy_regions = set(privacy_regional['Region'].tolist())
            common_regions = list(classical_regions.intersection(privacy_regions))
            
            if not common_regions:
                logger.warning("No common regions found between classical and privacy experiments")
                return
            
            # Filter to common regions and sort
            common_regions.sort()
            classical_filtered = classical_regional[classical_regional['Region'].isin(common_regions)].sort_values('Region')
            privacy_filtered = privacy_regional[privacy_regional['Region'].isin(common_regions)].sort_values('Region')
            
            # 1. F1-Score Comparison by Region
            ax1 = axes[0, 0]
            x_pos = np.arange(len(common_regions))
            width = 0.35
            
            classical_f1 = classical_filtered['MLP_Federated_F1'].tolist()
            privacy_f1 = privacy_filtered['MLP_Federated_F1'].tolist()
            
            bars1 = ax1.bar(x_pos - width/2, classical_f1, width, label='Classical FL', alpha=0.8, color='skyblue')
            bars2 = ax1.bar(x_pos + width/2, privacy_f1, width, label='Privacy-Preserving FL', alpha=0.8, color='lightcoral')
            
            # Add centralized baseline if available
            if self.centralized_data.get('metrics'):
                centralized_f1 = self.centralized_data['metrics']['f1_score']
                ax1.axhline(y=centralized_f1, color='red', linestyle='--', alpha=0.7, 
                           label=f'Centralized Baseline ({centralized_f1:.3f})')
            
            ax1.set_xlabel('Region')
            ax1.set_ylabel('F1-Score')
            ax1.set_title('F1-Score by Region')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(common_regions, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            for bar in bars2:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            # 2. Accuracy Comparison
            ax2 = axes[0, 1]
            classical_acc = classical_filtered['MLP_Federated_Accuracy'].tolist()
            privacy_acc = privacy_filtered['MLP_Federated_Accuracy'].tolist()
            
            bars3 = ax2.bar(x_pos - width/2, classical_acc, width, label='Classical FL', alpha=0.8, color='skyblue')
            bars4 = ax2.bar(x_pos + width/2, privacy_acc, width, label='Privacy-Preserving FL', alpha=0.8, color='lightcoral')
            
            if self.centralized_data.get('metrics'):
                centralized_acc = self.centralized_data['metrics']['accuracy']
                ax2.axhline(y=centralized_acc, color='red', linestyle='--', alpha=0.7, 
                           label=f'Centralized Baseline ({centralized_acc:.3f})')
            
            ax2.set_xlabel('Region')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Accuracy by Region')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(common_regions, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. ROC-AUC Comparison (with NaN handling)
            ax3 = axes[1, 0]
            classical_auc = classical_filtered['MLP_Federated_ROC_AUC'].tolist()
            privacy_auc = privacy_filtered['MLP_Federated_ROC_AUC'].tolist()
            
            # Handle NaN values
            classical_auc_clean = [v if not pd.isna(v) else 0 for v in classical_auc]
            privacy_auc_clean = [v if not pd.isna(v) else 0 for v in privacy_auc]
            
            bars5 = ax3.bar(x_pos - width/2, classical_auc_clean, width, label='Classical FL', alpha=0.8, color='skyblue')
            bars6 = ax3.bar(x_pos + width/2, privacy_auc_clean, width, label='Privacy-Preserving FL', alpha=0.8, color='lightcoral')
            
            if self.centralized_data.get('metrics'):
                centralized_auc = self.centralized_data['metrics']['roc_auc']
                ax3.axhline(y=centralized_auc, color='red', linestyle='--', alpha=0.7, 
                           label=f'Centralized Baseline ({centralized_auc:.3f})')
            
            ax3.set_xlabel('Region')
            ax3.set_ylabel('ROC-AUC')
            ax3.set_title('ROC-AUC by Region')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(common_regions, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Performance Gap Analysis
            ax4 = axes[1, 1]
            f1_gaps = [p - c for c, p in zip(classical_f1, privacy_f1)]
            acc_gaps = [p - c for c, p in zip(classical_acc, privacy_acc)]
            auc_gaps = [p - c for c, p in zip(classical_auc_clean, privacy_auc_clean)]
            
            x_gap = np.arange(len(common_regions))
            width_gap = 0.25
            
            bars_f1 = ax4.bar(x_gap - width_gap, f1_gaps, width_gap, label='F1-Score Gap', alpha=0.8)
            bars_acc = ax4.bar(x_gap, acc_gaps, width_gap, label='Accuracy Gap', alpha=0.8)
            bars_auc = ax4.bar(x_gap + width_gap, auc_gaps, width_gap, label='ROC-AUC Gap', alpha=0.8)
            
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.set_xlabel('Region')
            ax4.set_ylabel('Performance Gap (Privacy - Classical)')
            ax4.set_title('Privacy Cost Analysis\n(Positive = Privacy Better)')
            ax4.set_xticks(x_gap)
            ax4.set_xticklabels(common_regions, rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATION_OUTPUT_DIR, 'performance_comparison.png'), 
                       dpi=DPI, bbox_inches='tight')
            plt.show()
            
            logger.info("Created performance comparison visualization")
            
        except Exception as e:
            logger.error(f"Failed to create performance comparison: {e}")
            # Create error plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Error creating performance comparison:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_title('Performance Comparison Error')
            plt.savefig(os.path.join(VISUALIZATION_OUTPUT_DIR, 'performance_comparison_error.png'), 
                       dpi=DPI, bbox_inches='tight')
            plt.show()
    
    def create_simple_summary_table(self) -> None:
        """Create a simple summary table comparing key metrics."""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')
            
            # Collect summary data
            summary_data = []
            
            # Classical results
            classical_perf = self.classical_data.get('performance')
            if classical_perf is not None:
                classical_avg = classical_perf[classical_perf['Region'] == 'AVERAGE']
                if not classical_avg.empty:
                    summary_data.append([
                        'Classical FL',
                        f"{classical_avg['MLP_Federated_F1'].iloc[0]:.3f}",
                        f"{classical_avg['MLP_Federated_Accuracy'].iloc[0]:.3f}",
                        f"{classical_avg['MLP_Federated_ROC_AUC'].iloc[0]:.3f}",
                        'None',
                        'Standard FedAvg'
                    ])
            
            # Privacy results
            privacy_perf = self.privacy_data.get('performance')
            if privacy_perf is not None:
                privacy_avg = privacy_perf[privacy_perf['Region'] == 'AVERAGE']
                if not privacy_avg.empty:
                    # Get privacy parameters
                    privacy_config = self.privacy_data.get('config', {})
                    privacy_infra = privacy_config.get('privacy_infrastructure', {})
                    epsilon = privacy_infra.get('dp_epsilon_total', 'N/A')
                    
                    summary_data.append([
                        'Privacy-Preserving FL',
                        f"{privacy_avg['MLP_Federated_F1'].iloc[0]:.3f}",
                        f"{privacy_avg['MLP_Federated_Accuracy'].iloc[0]:.3f}",
                        f"{privacy_avg['MLP_Federated_ROC_AUC'].iloc[0]:.3f}",
                        f"ε = {epsilon}",
                        'Secure Aggregation + DP'
                    ])
            
            # Centralized baseline
            if self.centralized_data.get('metrics'):
                cent_metrics = self.centralized_data['metrics']
                summary_data.append([
                    'Centralized Baseline',
                    f"{cent_metrics['f1_score']:.3f}",
                    f"{cent_metrics['accuracy']:.3f}",
                    f"{cent_metrics['roc_auc']:.3f}",
                    'None',
                    'Standard Training'
                ])
            
            if summary_data:
                headers = ['Approach', 'F1-Score', 'Accuracy', 'ROC-AUC', 'Privacy Budget', 'Method']
                
                # Create table
                table = ax.table(cellText=summary_data, colLabels=headers,
                               cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1, 2.5)
                
                # Style the table
                for i in range(len(headers)):
                    table[(0, i)].set_facecolor('#4CAF50')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                # Color code rows
                colors = ['#E3F2FD', '#FFF3E0', '#F3E5F5']
                for i in range(1, len(summary_data) + 1):
                    for j in range(len(headers)):
                        table[(i, j)].set_facecolor(colors[i-1] if i-1 < len(colors) else '#F5F5F5')
                
                ax.set_title('Federated Learning Results Summary', fontsize=16, fontweight='bold', pad=20)
            else:
                ax.text(0.5, 0.5, 'No data available for summary table', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('Summary Table - No Data')
            
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATION_OUTPUT_DIR, 'summary_table.png'), 
                       dpi=DPI, bbox_inches='tight')
            plt.show()
            
            logger.info("Created summary table")
            
        except Exception as e:
            logger.error(f"Failed to create summary table: {e}")
    
    def create_training_progression_plot(self) -> None:
        """Create training progression visualization."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Training Progression Comparison', fontsize=16, fontweight='bold')
            
            # Get improvement data
            classical_improvement = self.classical_data.get('improvement')
            privacy_improvement = self.privacy_data.get('improvement')
            
            if classical_improvement is None or privacy_improvement is None:
                for ax in axes:
                    ax.text(0.5, 0.5, 'Training progression data not available', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title('Data Not Available')
                plt.tight_layout()
                plt.savefig(os.path.join(VISUALIZATION_OUTPUT_DIR, 'training_progression_error.png'), 
                           dpi=DPI, bbox_inches='tight')
                plt.show()
                return
            
            # 1. F1-Score Progression
            ax1 = axes[0]
            if 'Round' in classical_improvement.columns and 'Unweighted_Test_F1' in classical_improvement.columns:
                rounds = classical_improvement['Round'].tolist()
                classical_f1 = classical_improvement['Unweighted_Test_F1'].tolist()
                privacy_f1 = privacy_improvement['Unweighted_Test_F1'].tolist()
                
                ax1.plot(rounds, classical_f1, 'o-', label='Classical FL', linewidth=2, markersize=6)
                ax1.plot(rounds, privacy_f1, 's-', label='Privacy-Preserving FL', linewidth=2, markersize=6)
                
                if self.centralized_data.get('metrics'):
                    centralized_f1 = self.centralized_data['metrics']['f1_score']
                    ax1.axhline(y=centralized_f1, color='red', linestyle='--', alpha=0.7, 
                               label=f'Centralized Baseline ({centralized_f1:.3f})')
            
            ax1.set_xlabel('Round')
            ax1.set_ylabel('F1-Score')
            ax1.set_title('F1-Score Progression')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Improvement per Round
            ax2 = axes[1]
            if 'Round' in classical_improvement.columns and 'Unweighted_Improvement' in classical_improvement.columns:
                classical_imp = classical_improvement['Unweighted_Improvement'].tolist()
                privacy_imp = privacy_improvement['Unweighted_Improvement'].tolist()
                
                ax2.bar(np.array(rounds) - 0.2, classical_imp, 0.4, label='Classical FL', alpha=0.8)
                ax2.bar(np.array(rounds) + 0.2, privacy_imp, 0.4, label='Privacy-Preserving FL', alpha=0.8)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            ax2.set_xlabel('Round')
            ax2.set_ylabel('F1-Score Improvement')
            ax2.set_title('Per-Round Improvement')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATION_OUTPUT_DIR, 'training_progression.png'), 
                       dpi=DPI, bbox_inches='tight')
            plt.show()
            
            logger.info("Created training progression visualization")
            
        except Exception as e:
            logger.error(f"Failed to create training progression: {e}")
    
    def generate_simple_report(self) -> None:
        """Generate a simple text report with available data."""
        report_path = os.path.join(VISUALIZATION_OUTPUT_DIR, 'comparison_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Federated Learning Comparison Report\n")
            f.write("## Classical vs Privacy-Preserving Approaches\n\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data availability
            f.write("## Data Availability\n\n")
            f.write(f"- Classical results loaded: {len(self.classical_data)} datasets\n")
            f.write(f"- Privacy results loaded: {len(self.privacy_data)} datasets\n")
            f.write(f"- Centralized baseline loaded: {len(self.centralized_data)} datasets\n\n")
            
            # Performance summary
            classical_perf = self.classical_data.get('performance')
            privacy_perf = self.privacy_data.get('performance')
            
            if classical_perf is not None and privacy_perf is not None:
                f.write("## Performance Summary\n\n")
                
                classical_avg = classical_perf[classical_perf['Region'] == 'AVERAGE']
                privacy_avg = privacy_perf[privacy_perf['Region'] == 'AVERAGE']
                
                if not classical_avg.empty and not privacy_avg.empty:
                    classical_f1 = classical_avg['MLP_Federated_F1'].iloc[0]
                    privacy_f1 = privacy_avg['MLP_Federated_F1'].iloc[0]
                    f1_gap = classical_f1 - privacy_f1
                    
                    f.write(f"- **Classical FL F1-Score**: {classical_f1:.3f}\n")
                    f.write(f"- **Privacy-Preserving FL F1-Score**: {privacy_f1:.3f}\n")
                    f.write(f"- **Privacy Cost**: {f1_gap:+.3f} F1-Score points\n\n")
                    
                    if abs(f1_gap) < 0.05:
                        f.write("**Assessment**: Privacy preservation achieved with minimal utility loss\n\n")
                    elif f1_gap > 0:
                        f.write("**Assessment**: Privacy preservation comes with moderate utility cost\n\n")
                    else:
                        f.write("**Assessment**: Privacy-preserving approach shows competitive performance\n\n")
            
            # Privacy configuration
            privacy_config = self.privacy_data.get('config', {})
            privacy_infra = privacy_config.get('privacy_infrastructure', {})
            
            if privacy_infra:
                f.write("## Privacy Configuration\n\n")
                f.write(f"- **Privacy Budget (ε)**: {privacy_infra.get('dp_epsilon_total', 'N/A')}\n")
                f.write(f"- **Privacy Budget (δ)**: {privacy_infra.get('dp_delta', 'N/A')}\n")
                f.write(f"- **Noise Multiplier**: {privacy_infra.get('dp_noise_multiplier', 'N/A')}\n")
                f.write(f"- **Shamir Threshold**: {privacy_infra.get('shamir_threshold', 'N/A')}-of-{privacy_infra.get('shamir_participants', 'N/A')}\n")
                f.write(f"- **Double DP Prevention**: {privacy_infra.get('double_dp_prevention_enabled', False)}\n\n")
            
            # Files generated
            f.write("## Files Generated\n\n")
            f.write("- performance_comparison.png - Regional performance comparison\n")
            f.write("- summary_table.png - Key metrics summary table\n")
            f.write("- training_progression.png - Training progression over rounds\n")
            f.write("- comparison_report.md - This report\n")
        
        logger.info(f"Generated simple report: {report_path}")
    
    def run_complete_analysis(self) -> None:
        """Run the complete analysis with improved error handling."""
        logger.info("Starting comprehensive federated learning analysis")
        
        # Check available files first
        self.check_and_list_files()
        
        # Load all data
        logger.info("Loading experimental results...")
        classical_loaded = self.load_classical_results()
        privacy_loaded = self.load_privacy_results()
        centralized_loaded = self.load_centralized_baseline()
        
        if not classical_loaded and not privacy_loaded:
            logger.error("Failed to load any results. Check directory paths and file names.")
            return
        
        if classical_loaded:
            logger.info("Classical results loaded successfully")
        else:
            logger.warning("Classical results not loaded")
        
        if privacy_loaded:
            logger.info("Privacy results loaded successfully")
        else:
            logger.warning("Privacy results not loaded")
        
        # Generate visualizations with error handling
        try:
            logger.info("Generating performance comparison...")
            self.create_performance_comparison()
        except Exception as e:
            logger.error(f"Failed to create performance comparison: {e}")
        
        try:
            logger.info("Generating summary table...")
            self.create_simple_summary_table()
        except Exception as e:
            logger.error(f"Failed to create summary table: {e}")
        
        try:
            logger.info("Generating training progression...")
            self.create_training_progression_plot()
        except Exception as e:
            logger.error(f"Failed to create training progression: {e}")
        
        try:
            logger.info("Generating summary report...")
            self.generate_simple_report()
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
        
        logger.info(f"Analysis complete! Check {VISUALIZATION_OUTPUT_DIR} for outputs")


def main():
    """Main function to run the complete analysis."""
    print("=" * 60)
    print("FEDERATED LEARNING RESULTS VISUALIZATION")
    print("=" * 60)
    print(f"Looking for results in:")
    print(f"  Classical: {CLASSICAL_RESULTS_DIR}")
    print(f"  Privacy:   {PRIVACY_RESULTS_DIR}")
    print(f"  Output:    {VISUALIZATION_OUTPUT_DIR}")
    print("=" * 60)
    
    analyzer = FederatedResultsAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 09_mlp_federated_privacy_visualization.py
Description: Comprehensive visualization and comparison of classical vs privacy-preserving
             federated learning results. Analyzes performance trade-offs, privacy costs,
             training progression, and provides detailed comparative analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CLASSICAL_RESULTS_DIR = 'results_mlp_federated'
PRIVACY_RESULTS_DIR = 'results_mlp_federated_privacy'
VISUALIZATION_OUTPUT_DIR = 'visualizations_federated_comparison'
CENTRALIZED_RESULTS_DIR = 'results'

# Visualization settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
FIGURE_SIZE = (12, 8)
DPI = 300

class FederatedResultsAnalyzer:
    """
    Comprehensive analyzer for federated learning experiments comparing
    classical and privacy-preserving approaches.
    """
    
    def __init__(self):
        self.classical_data = {}
        self.privacy_data = {}
        self.centralized_data = {}
        self.comparison_data = {}
        
        # Create output directory
        os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
        
        logger.info("Initialized Federated Results Analyzer")
    
    def load_classical_results(self) -> bool:
        """Load classical federated learning results."""
        try:
            # Performance results
            perf_path = os.path.join(CLASSICAL_RESULTS_DIR, 'mlp_federated_performance.csv')
            if os.path.exists(perf_path):
                self.classical_data['performance'] = pd.read_csv(perf_path)
                logger.info(f"Loaded classical performance data: {len(self.classical_data['performance'])} regions")
            
            # Training history
            history_path = os.path.join(CLASSICAL_RESULTS_DIR, 'federated_training_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.classical_data['history'] = json.load(f)
                logger.info(f"Loaded classical training history: {len(self.classical_data['history'])} rounds")
            
            # Improvement summary
            improvement_path = os.path.join(CLASSICAL_RESULTS_DIR, 'federated_improvement_summary.csv')
            if os.path.exists(improvement_path):
                self.classical_data['improvement'] = pd.read_csv(improvement_path)
                logger.info(f"Loaded classical improvement summary: {len(self.classical_data['improvement'])} rounds")
            
            # Configuration
            config_path = os.path.join(CLASSICAL_RESULTS_DIR, 'mlp_federated_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.classical_data['config'] = json.load(f)
                logger.info("Loaded classical configuration")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load classical results: {e}")
            return False
    
    def load_privacy_results(self) -> bool:
        """Load privacy-preserving federated learning results."""
        try:
            # Performance results
            perf_path = os.path.join(PRIVACY_RESULTS_DIR, 'mlp_federated_privacy_performance.csv')
            if os.path.exists(perf_path):
                self.privacy_data['performance'] = pd.read_csv(perf_path)
                logger.info(f"Loaded privacy performance data: {len(self.privacy_data['performance'])} regions")
            
            # Training history
            history_path = os.path.join(PRIVACY_RESULTS_DIR, 'federated_privacy_training_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.privacy_data['history'] = json.load(f)
                logger.info(f"Loaded privacy training history: {len(self.privacy_data['history'])} rounds")
            
            # Improvement summary
            improvement_path = os.path.join(PRIVACY_RESULTS_DIR, 'federated_privacy_improvement_summary.csv')
            if os.path.exists(improvement_path):
                self.privacy_data['improvement'] = pd.read_csv(improvement_path)
                logger.info(f"Loaded privacy improvement summary: {len(self.privacy_data['improvement'])} rounds")
            
            # Configuration
            config_path = os.path.join(PRIVACY_RESULTS_DIR, 'mlp_federated_privacy_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.privacy_data['config'] = json.load(f)
                logger.info("Loaded privacy configuration")
            
            # Delta norm statistics (if available)
            delta_path = os.path.join(PRIVACY_RESULTS_DIR, 'delta_norm_statistics.json')
            if os.path.exists(delta_path):
                with open(delta_path, 'r') as f:
                    self.privacy_data['delta_norms'] = json.load(f)
                logger.info("Loaded delta norm statistics")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load privacy results: {e}")
            return False
    
    def load_centralized_baseline(self) -> bool:
        """Load centralized baseline for comparison."""
        try:
            # Centralized metrics
            metrics_path = os.path.join(CENTRALIZED_RESULTS_DIR, 'metrics_summary.csv')
            if os.path.exists(metrics_path):
                metrics_df = pd.read_csv(metrics_path, index_col=0)
                if 'MLP_Optimized' in metrics_df.index:
                    self.centralized_data['metrics'] = {
                        'f1_score': metrics_df.loc['MLP_Optimized', 'f1_score'],
                        'accuracy': metrics_df.loc['MLP_Optimized', 'accuracy'],
                        'roc_auc': metrics_df.loc['MLP_Optimized', 'roc_auc']
                    }
                    logger.info("Loaded centralized baseline metrics")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load centralized baseline: {e}")
            return False
    
    def create_performance_comparison(self) -> None:
        """Create comprehensive performance comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Federated Learning Performance Comparison\nClassical vs Privacy-Preserving', 
                     fontsize=16, fontweight='bold')
        
        # Prepare data for comparison
        classical_perf = self.classical_data.get('performance', pd.DataFrame())
        privacy_perf = self.privacy_data.get('performance', pd.DataFrame())
        
        if classical_perf.empty or privacy_perf.empty:
            logger.warning("Missing performance data for comparison")
            return
        
        # Get regional data (exclude AVERAGE and WEIGHTED_AVERAGE)
        classical_regional = classical_perf[~classical_perf['Region'].isin(['AVERAGE', 'WEIGHTED_AVERAGE'])]
        privacy_regional = privacy_perf[~privacy_perf['Region'].isin(['AVERAGE', 'WEIGHTED_AVERAGE'])]
        
        # 1. F1-Score Comparison by Region
        ax1 = axes[0, 0]
        regions = classical_regional['Region'].tolist()
        x_pos = np.arange(len(regions))
        width = 0.35
        
        classical_f1 = classical_regional['MLP_Federated_F1'].tolist()
        privacy_f1 = privacy_regional['MLP_Federated_F1'].tolist()
        
        bars1 = ax1.bar(x_pos - width/2, classical_f1, width, label='Classical FL', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, privacy_f1, width, label='Privacy-Preserving FL', alpha=0.8)
        
        # Add centralized baseline if available
        if self.centralized_data.get('metrics'):
            centralized_f1 = self.centralized_data['metrics']['f1_score']
            ax1.axhline(y=centralized_f1, color='red', linestyle='--', alpha=0.7, 
                       label=f'Centralized Baseline ({centralized_f1:.3f})')
        
        ax1.set_xlabel('Region')
        ax1.set_ylabel('F1-Score')
        ax1.set_title('F1-Score by Region')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(regions, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Accuracy Comparison
        ax2 = axes[0, 1]
        classical_acc = classical_regional['MLP_Federated_Accuracy'].tolist()
        privacy_acc = privacy_regional['MLP_Federated_Accuracy'].tolist()
        
        bars3 = ax2.bar(x_pos - width/2, classical_acc, width, label='Classical FL', alpha=0.8)
        bars4 = ax2.bar(x_pos + width/2, privacy_acc, width, label='Privacy-Preserving FL', alpha=0.8)
        
        if self.centralized_data.get('metrics'):
            centralized_acc = self.centralized_data['metrics']['accuracy']
            ax2.axhline(y=centralized_acc, color='red', linestyle='--', alpha=0.7, 
                       label=f'Centralized Baseline ({centralized_acc:.3f})')
        
        ax2.set_xlabel('Region')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy by Region')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(regions, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. ROC-AUC Comparison
        ax3 = axes[1, 0]
        classical_auc = classical_regional['MLP_Federated_ROC_AUC'].tolist()
        privacy_auc = privacy_regional['MLP_Federated_ROC_AUC'].tolist()
        
        bars5 = ax3.bar(x_pos - width/2, classical_auc, width, label='Classical FL', alpha=0.8)
        bars6 = ax3.bar(x_pos + width/2, privacy_auc, width, label='Privacy-Preserving FL', alpha=0.8)
        
        if self.centralized_data.get('metrics'):
            centralized_auc = self.centralized_data['metrics']['roc_auc']
            ax3.axhline(y=centralized_auc, color='red', linestyle='--', alpha=0.7, 
                       label=f'Centralized Baseline ({centralized_auc:.3f})')
        
        ax3.set_xlabel('Region')
        ax3.set_ylabel('ROC-AUC')
        ax3.set_title('ROC-AUC by Region')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(regions, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Gap Analysis
        ax4 = axes[1, 1]
        f1_gaps = [p - c for c, p in zip(classical_f1, privacy_f1)]
        acc_gaps = [p - c for c, p in zip(classical_acc, privacy_acc)]
        auc_gaps = [p - c for c, p in zip(classical_auc, privacy_auc)]
        
        ax4.bar(x_pos - width, f1_gaps, width, label='F1-Score Gap', alpha=0.8)
        ax4.bar(x_pos, acc_gaps, width, label='Accuracy Gap', alpha=0.8)
        ax4.bar(x_pos + width, auc_gaps, width, label='ROC-AUC Gap', alpha=0.8)
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('Region')
        ax4.set_ylabel('Performance Gap (Privacy - Classical)')
        ax4.set_title('Privacy Cost Analysis\n(Positive = Privacy Better)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(regions, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_OUTPUT_DIR, 'performance_comparison.png'), 
                   dpi=DPI, bbox_inches='tight')
        plt.show()
        
        logger.info("Created performance comparison visualization")
    
    def create_training_progression_analysis(self) -> None:
        """Analyze and visualize training progression for both approaches."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Progression Analysis\nClassical vs Privacy-Preserving FL', 
                     fontsize=16, fontweight='bold')
        
        # Get improvement data
        classical_improvement = self.classical_data.get('improvement', pd.DataFrame())
        privacy_improvement = self.privacy_data.get('improvement', pd.DataFrame())
        
        if classical_improvement.empty or privacy_improvement.empty:
            logger.warning("Missing improvement data for progression analysis")
            return
        
        rounds = classical_improvement['Round'].tolist()
        
        # 1. F1-Score Progression (Unweighted)
        ax1 = axes[0, 0]
        classical_f1 = classical_improvement['Unweighted_Test_F1'].tolist()
        privacy_f1 = privacy_improvement['Unweighted_Test_F1'].tolist()
        
        ax1.plot(rounds, classical_f1, 'o-', label='Classical FL', linewidth=2, markersize=6)
        ax1.plot(rounds, privacy_f1, 's-', label='Privacy-Preserving FL', linewidth=2, markersize=6)
        
        if self.centralized_data.get('metrics'):
            centralized_f1 = self.centralized_data['metrics']['f1_score']
            ax1.axhline(y=centralized_f1, color='red', linestyle='--', alpha=0.7, 
                       label=f'Centralized Baseline ({centralized_f1:.3f})')
        
        ax1.set_xlabel('Round')
        ax1.set_ylabel('F1-Score (Unweighted)')
        ax1.set_title('Training Progression: F1-Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. F1-Score Progression (Weighted)
        ax2 = axes[0, 1]
        classical_f1_w = classical_improvement['Weighted_Test_F1'].tolist()
        privacy_f1_w = privacy_improvement['Weighted_Test_F1'].tolist()
        
        ax2.plot(rounds, classical_f1_w, 'o-', label='Classical FL', linewidth=2, markersize=6)
        ax2.plot(rounds, privacy_f1_w, 's-', label='Privacy-Preserving FL', linewidth=2, markersize=6)
        
        if self.centralized_data.get('metrics'):
            centralized_f1 = self.centralized_data['metrics']['f1_score']
            ax2.axhline(y=centralized_f1, color='red', linestyle='--', alpha=0.7, 
                       label=f'Centralized Baseline ({centralized_f1:.3f})')
        
        ax2.set_xlabel('Round')
        ax2.set_ylabel('F1-Score (Weighted)')
        ax2.set_title('Training Progression: F1-Score (Weighted)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Improvement per Round
        ax3 = axes[1, 0]
        classical_imp = classical_improvement['Unweighted_Improvement'].tolist()
        privacy_imp = privacy_improvement['Unweighted_Improvement'].tolist()
        
        ax3.bar(np.array(rounds) - 0.2, classical_imp, 0.4, label='Classical FL', alpha=0.8)
        ax3.bar(np.array(rounds) + 0.2, privacy_imp, 0.4, label='Privacy-Preserving FL', alpha=0.8)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax3.set_xlabel('Round')
        ax3.set_ylabel('F1-Score Improvement')
        ax3.set_title('Per-Round Improvement')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative Improvement
        ax4 = axes[1, 1]
        classical_cumsum = np.cumsum(classical_imp)
        privacy_cumsum = np.cumsum(privacy_imp)
        
        ax4.plot(rounds, classical_cumsum, 'o-', label='Classical FL', linewidth=2, markersize=6)
        ax4.plot(rounds, privacy_cumsum, 's-', label='Privacy-Preserving FL', linewidth=2, markersize=6)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Cumulative F1-Score Improvement')
        ax4.set_title('Cumulative Learning Progress')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_OUTPUT_DIR, 'training_progression.png'), 
                   dpi=DPI, bbox_inches='tight')
        plt.show()
        
        logger.info("Created training progression analysis")
    
    def create_privacy_analysis(self) -> None:
        """Create privacy-specific analysis and visualizations."""
        if not self.privacy_data.get('config'):
            logger.warning("No privacy configuration data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Privacy Analysis\nDifferential Privacy and Secure Aggregation', 
                     fontsize=16, fontweight='bold')
        
        # Privacy configuration
        privacy_config = self.privacy_data['config']
        privacy_infra = privacy_config.get('privacy_infrastructure', {})
        
        # 1. Privacy Parameters Summary
        ax1 = axes[0, 0]
        ax1.axis('off')
        
        privacy_text = []
        if privacy_infra:
            privacy_text.extend([
                f"Privacy Budget (ε): {privacy_infra.get('dp_epsilon_total', 'N/A')}",
                f"Privacy Budget (δ): {privacy_infra.get('dp_delta', 'N/A')}",
                f"Noise Multiplier: {privacy_infra.get('dp_noise_multiplier', 'N/A'):.4f}",
                f"Shamir Threshold: {privacy_infra.get('shamir_threshold', 'N/A')}-of-{privacy_infra.get('shamir_participants', 'N/A')}",
                f"Rounds: {privacy_infra.get('num_rounds', 'N/A')}",
                f"Clients: {privacy_infra.get('num_clients', 'N/A')}",
                f"Strict DP Mode: {privacy_infra.get('dp_strict_mode', False)}",
                f"Double DP Prevention: {privacy_infra.get('double_dp_prevention_enabled', False)}"
            ])
        
        for i, text in enumerate(privacy_text):
            ax1.text(0.1, 0.9 - i*0.1, text, fontsize=12, transform=ax1.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        ax1.set_title('Privacy Configuration')
        
        # 2. Delta Norm Statistics (if available)
        ax2 = axes[0, 1]
        if 'delta_norms' in self.privacy_data:
            delta_norms = self.privacy_data['delta_norms']
            layer_names = []
            means = []
            stds = []
            
            for layer, stats in delta_norms.items():
                if any(major in layer for major in ['layer_0_weights', 'layer_1_weights', 'output']):
                    layer_names.append(layer.replace('_', '\n'))
                    means.append(stats['mean'])
                    stds.append(stats['std'])
            
            if layer_names:
                x_pos = np.arange(len(layer_names))
                ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8)
                ax2.set_xlabel('Layer')
                ax2.set_ylabel('Delta L2 Norm')
                ax2.set_title('Parameter Delta Norms\n(for DP Calibration)')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(layer_names, fontsize=10)
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No delta norm\ndata available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Delta Norm Analysis')
        else:
            ax2.text(0.5, 0.5, 'No delta norm\ndata available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Delta Norm Analysis')
        
        # 3. Privacy vs Utility Trade-off
        ax3 = axes[1, 0]
        
        # Compare final performance
        classical_perf = self.classical_data.get('performance', pd.DataFrame())
        privacy_perf = self.privacy_data.get('performance', pd.DataFrame())
        
        if not classical_perf.empty and not privacy_perf.empty:
            # Get average performance
            classical_avg = classical_perf[classical_perf['Region'] == 'AVERAGE']
            privacy_avg = privacy_perf[privacy_perf['Region'] == 'AVERAGE']
            
            if not classical_avg.empty and not privacy_avg.empty:
                metrics = ['F1-Score', 'Accuracy', 'ROC-AUC']
                classical_vals = [
                    classical_avg['MLP_Federated_F1'].iloc[0],
                    classical_avg['MLP_Federated_Accuracy'].iloc[0],
                    classical_avg['MLP_Federated_ROC_AUC'].iloc[0]
                ]
                privacy_vals = [
                    privacy_avg['MLP_Federated_F1'].iloc[0],
                    privacy_avg['MLP_Federated_Accuracy'].iloc[0],
                    privacy_avg['MLP_Federated_ROC_AUC'].iloc[0]
                ]
                
                x_pos = np.arange(len(metrics))
                width = 0.35
                
                bars1 = ax3.bar(x_pos - width/2, classical_vals, width, 
                               label='Classical FL', alpha=0.8)
                bars2 = ax3.bar(x_pos + width/2, privacy_vals, width, 
                               label='Privacy-Preserving FL', alpha=0.8)
                
                ax3.set_xlabel('Metric')
                ax3.set_ylabel('Performance')
                ax3.set_title('Privacy vs Utility Trade-off\n(Average Performance)')
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels(metrics)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Dropout Recovery Analysis (if available)
        ax4 = axes[1, 1]
        
        privacy_history = self.privacy_data.get('history', [])
        dropout_rounds = [r for r in privacy_history if r.get('dropout_test', False)]
        
        if dropout_rounds:
            round_nums = [r['round'] + 1 for r in dropout_rounds]
            f1_scores = [r['average_test_f1'] for r in dropout_rounds]
            dropped_counts = [len(r.get('dropped_regions', [])) for r in dropout_rounds]
            
            colors = ['red' if d > 0 else 'blue' for d in dropped_counts]
            scatter = ax4.scatter(round_nums, f1_scores, c=colors, s=100, alpha=0.7)
            
            for i, (round_num, f1, dropped) in enumerate(zip(round_nums, f1_scores, dropped_counts)):
                ax4.annotate(f'{dropped} dropped', (round_num, f1), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax4.set_xlabel('Round')
            ax4.set_ylabel('F1-Score')
            ax4.set_title('Dropout Recovery Testing\n(Red = Dropouts, Blue = No Dropouts)')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No dropout\ntesting data\navailable', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Dropout Recovery Analysis')
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_OUTPUT_DIR, 'privacy_analysis.png'), 
                   dpi=DPI, bbox_inches='tight')
        plt.show()
        
        logger.info("Created privacy analysis visualization")
    
    def create_comprehensive_summary(self) -> None:
        """Create a comprehensive summary comparison."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Federated Learning Analysis\nClassical vs Privacy-Preserving Comparison', 
                     fontsize=16, fontweight='bold')
        
        # 1. Overall Performance Summary
        ax1 = axes[0, 0]
        self._plot_overall_performance_summary(ax1)
        
        # 2. Training Efficiency
        ax2 = axes[0, 1]
        self._plot_training_efficiency(ax2)
        
        # 3. Regional Performance Variance
        ax3 = axes[0, 2]
        self._plot_regional_variance(ax3)
        
        # 4. Convergence Analysis
        ax4 = axes[1, 0]
        self._plot_convergence_analysis(ax4)
        
        # 5. Privacy Cost Summary
        ax5 = axes[1, 1]
        self._plot_privacy_cost_summary(ax5)
        
        # 6. Key Metrics Comparison
        ax6 = axes[1, 2]
        self._plot_key_metrics_table(ax6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_OUTPUT_DIR, 'comprehensive_summary.png'), 
                   dpi=DPI, bbox_inches='tight')
        plt.show()
        
        logger.info("Created comprehensive summary visualization")
    
    def _plot_overall_performance_summary(self, ax) -> None:
        """Plot overall performance summary."""
        classical_perf = self.classical_data.get('performance', pd.DataFrame())
        privacy_perf = self.privacy_data.get('performance', pd.DataFrame())
        
        if classical_perf.empty or privacy_perf.empty:
            ax.text(0.5, 0.5, 'No performance\ndata available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Overall Performance')
            return
        
        # Get weighted averages
        classical_weighted = classical_perf[classical_perf['Region'] == 'WEIGHTED_AVERAGE']
        privacy_weighted = privacy_perf[privacy_perf['Region'] == 'WEIGHTED_AVERAGE']
        
        if not classical_weighted.empty and not privacy_weighted.empty:
            metrics = ['F1', 'Accuracy', 'ROC-AUC']
            classical_vals = [
                classical_weighted['MLP_Federated_F1'].iloc[0],
                classical_weighted['MLP_Federated_Accuracy'].iloc[0],
                classical_weighted['MLP_Federated_ROC_AUC'].iloc[0]
            ]
            privacy_vals = [
                privacy_weighted['MLP_Federated_F1'].iloc[0],
                privacy_weighted['MLP_Federated_Accuracy'].iloc[0],
                privacy_weighted['MLP_Federated_ROC_AUC'].iloc[0]
            ]
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            classical_vals += classical_vals[:1]
            privacy_vals += privacy_vals[:1]
            
            ax.plot(angles, classical_vals, 'o-', linewidth=2, label='Classical FL')
            ax.plot(angles, privacy_vals, 's-', linewidth=2, label='Privacy-Preserving FL')
            ax.fill(angles, classical_vals, alpha=0.25)
            ax.fill(angles, privacy_vals, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.grid(True)
            ax.legend()
        
        ax.set_title('Overall Performance\n(Weighted Averages)')
    
    def _plot_training_efficiency(self, ax) -> None:
        """Plot training efficiency comparison."""
        classical_improvement = self.classical_data.get('improvement', pd.DataFrame())
        privacy_improvement = self.privacy_data.get('improvement', pd.DataFrame())
        
        if classical_improvement.empty or privacy_improvement.empty:
            ax.text(0.5, 0.5, 'No training\ndata available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Efficiency')
            return
        
        # Calculate training efficiency metrics
        classical_final = classical_improvement['Unweighted_Test_F1'].iloc[-1]
        classical_initial = classical_improvement['Unweighted_Test_F1'].iloc[0]
        classical_improvement_total = classical_final - classical_initial
        
        privacy_final = privacy_improvement['Unweighted_Test_F1'].iloc[-1]
        privacy_initial = privacy_improvement['Unweighted_Test_F1'].iloc[0]
        privacy_improvement_total = privacy_final - privacy_initial
        
        # Average training time per round
        classical_avg_time = classical_improvement['Training_Time'].mean()
        privacy_avg_time = privacy_improvement['Training_Time'].mean()
        
        # Create efficiency comparison
        categories = ['Total\nImprovement', 'Avg Time\nper Round']
        classical_vals = [classical_improvement_total, classical_avg_time]
        privacy_vals = [privacy_improvement_total, privacy_avg_time]
        
        x_pos = np.arange(len(categories))
        width = 0.35
        
        # Normalize values for visualization
        max_improvement = max(classical_improvement_total, privacy_improvement_total)
        max_time = max(classical_avg_time, privacy_avg_time)
        
        normalized_classical = [
            classical_improvement_total / max_improvement if max_improvement > 0 else 0,
            classical_avg_time / max_time if max_time > 0 else 0
        ]
        normalized_privacy = [
            privacy_improvement_total / max_improvement if max_improvement > 0 else 0,
            privacy_avg_time / max_time if max_time > 0 else 0
        ]
        
        bars1 = ax.bar(x_pos - width/2, normalized_classical, width, 
                      label='Classical FL', alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, normalized_privacy, width, 
                      label='Privacy-Preserving FL', alpha=0.8)
        
        # Add actual values as text
        for i, (bar1, bar2, cv, pv) in enumerate(zip(bars1, bars2, classical_vals, privacy_vals)):
            ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.01,
                   f'{cv:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 0.01,
                   f'{pv:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Normalized Value')
        ax.set_title('Training Efficiency')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_regional_variance(self, ax) -> None:
        """Plot regional performance variance."""
        classical_perf = self.classical_data.get('performance', pd.DataFrame())
        privacy_perf = self.privacy_data.get('performance', pd.DataFrame())
        
        if classical_perf.empty or privacy_perf.empty:
            ax.text(0.5, 0.5, 'No performance\ndata available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Regional Variance')
            return
        
        # Get regional data (exclude summaries)
        classical_regional = classical_perf[~classical_perf['Region'].isin(['AVERAGE', 'WEIGHTED_AVERAGE'])]
        privacy_regional = privacy_perf[~privacy_perf['Region'].isin(['AVERAGE', 'WEIGHTED_AVERAGE'])]
        
        # Calculate variance in F1 scores
        classical_f1_std = classical_regional['MLP_Federated_F1'].std()
        privacy_f1_std = privacy_regional['MLP_Federated_F1'].std()
        
        classical_f1_range = (classical_regional['MLP_Federated_F1'].max() - 
                             classical_regional['MLP_Federated_F1'].min())
        privacy_f1_range = (privacy_regional['MLP_Federated_F1'].max() - 
                           privacy_regional['MLP_Federated_F1'].min())
        
        # Create variance comparison
        metrics = ['Std Dev', 'Range']
        classical_vals = [classical_f1_std, classical_f1_range]
        privacy_vals = [privacy_f1_std, privacy_f1_range]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, classical_vals, width, 
                      label='Classical FL', alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, privacy_vals, width, 
                      label='Privacy-Preserving FL', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('F1-Score Variance')
        ax.set_title('Regional Performance Variance\n(Lower = More Consistent)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_convergence_analysis(self, ax) -> None:
        """Plot convergence analysis."""
        classical_improvement = self.classical_data.get('improvement', pd.DataFrame())
        privacy_improvement = self.privacy_data.get('improvement', pd.DataFrame())
        
        if classical_improvement.empty or privacy_improvement.empty:
            ax.text(0.5, 0.5, 'No training\ndata available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Convergence Analysis')
            return
        
        rounds = classical_improvement['Round'].tolist()
        classical_f1 = classical_improvement['Unweighted_Test_F1'].tolist()
        privacy_f1 = privacy_improvement['Unweighted_Test_F1'].tolist()
        
        # Plot convergence curves
        ax.plot(rounds, classical_f1, 'o-', label='Classical FL', linewidth=2)
        ax.plot(rounds, privacy_f1, 's-', label='Privacy-Preserving FL', linewidth=2)
        
        # Add trend lines
        classical_trend = np.polyfit(rounds, classical_f1, 1)
        privacy_trend = np.polyfit(rounds, privacy_f1, 1)
        
        ax.plot(rounds, np.poly1d(classical_trend)(rounds), '--', alpha=0.7, color='blue')
        ax.plot(rounds, np.poly1d(privacy_trend)(rounds), '--', alpha=0.7, color='orange')
        
        # Add convergence indicators
        classical_final_5 = np.std(classical_f1[-5:]) if len(classical_f1) >= 5 else np.std(classical_f1)
        privacy_final_5 = np.std(privacy_f1[-5:]) if len(privacy_f1) >= 5 else np.std(privacy_f1)
        
        ax.text(0.05, 0.95, f'Classical variance (last 5): {classical_final_5:.4f}', 
               transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.text(0.05, 0.85, f'Privacy variance (last 5): {privacy_final_5:.4f}', 
               transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        
        ax.set_xlabel('Round')
        ax.set_ylabel('F1-Score')
        ax.set_title('Convergence Analysis\n(with trend lines)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_privacy_cost_summary(self, ax) -> None:
        """Plot privacy cost summary."""
        classical_perf = self.classical_data.get('performance', pd.DataFrame())
        privacy_perf = self.privacy_data.get('performance', pd.DataFrame())
        
        if classical_perf.empty or privacy_perf.empty:
            ax.text(0.5, 0.5, 'No performance\ndata available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Privacy Cost')
            return
        
        # Calculate privacy costs (performance gaps)
        classical_avg = classical_perf[classical_perf['Region'] == 'AVERAGE']
        privacy_avg = privacy_perf[privacy_perf['Region'] == 'AVERAGE']
        
        if not classical_avg.empty and not privacy_avg.empty:
            f1_cost = classical_avg['MLP_Federated_F1'].iloc[0] - privacy_avg['MLP_Federated_F1'].iloc[0]
            acc_cost = classical_avg['MLP_Federated_Accuracy'].iloc[0] - privacy_avg['MLP_Federated_Accuracy'].iloc[0]
            auc_cost = classical_avg['MLP_Federated_ROC_AUC'].iloc[0] - privacy_avg['MLP_Federated_ROC_AUC'].iloc[0]
            
            metrics = ['F1-Score', 'Accuracy', 'ROC-AUC']
            costs = [f1_cost, acc_cost, auc_cost]
            colors = ['red' if c > 0 else 'green' for c in costs]
            
            bars = ax.bar(metrics, costs, color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Add value labels
            for bar, cost in zip(bars, costs):
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                offset = 0.01 if height >= 0 else -0.01
                ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                       f'{cost:+.3f}', ha='center', va=va, fontsize=10, fontweight='bold')
            
            ax.set_ylabel('Performance Cost\n(Classical - Privacy)')
            ax.set_title('Privacy Cost Analysis\n(Red = Cost, Green = Benefit)')
            ax.grid(True, alpha=0.3)
    
    def _plot_key_metrics_table(self, ax) -> None:
        """Plot key metrics comparison table."""
        ax.axis('off')
        
        # Collect key metrics
        metrics_data = []
        
        # Performance metrics
        classical_perf = self.classical_data.get('performance', pd.DataFrame())
        privacy_perf = self.privacy_data.get('performance', pd.DataFrame())
        
        if not classical_perf.empty and not privacy_perf.empty:
            classical_avg = classical_perf[classical_perf['Region'] == 'AVERAGE']
            privacy_avg = privacy_perf[privacy_perf['Region'] == 'AVERAGE']
            
            if not classical_avg.empty and not privacy_avg.empty:
                metrics_data.append(['Final F1-Score', 
                                   f"{classical_avg['MLP_Federated_F1'].iloc[0]:.3f}",
                                   f"{privacy_avg['MLP_Federated_F1'].iloc[0]:.3f}"])
                metrics_data.append(['Final Accuracy', 
                                   f"{classical_avg['MLP_Federated_Accuracy'].iloc[0]:.3f}",
                                   f"{privacy_avg['MLP_Federated_Accuracy'].iloc[0]:.3f}"])
                metrics_data.append(['Final ROC-AUC', 
                                   f"{classical_avg['MLP_Federated_ROC_AUC'].iloc[0]:.3f}",
                                   f"{privacy_avg['MLP_Federated_ROC_AUC'].iloc[0]:.3f}"])
        
        # Training metrics
        classical_improvement = self.classical_data.get('improvement', pd.DataFrame())
        privacy_improvement = self.privacy_data.get('improvement', pd.DataFrame())
        
        if not classical_improvement.empty and not privacy_improvement.empty:
            classical_total_imp = (classical_improvement['Unweighted_Test_F1'].iloc[-1] - 
                                 classical_improvement['Unweighted_Test_F1'].iloc[0])
            privacy_total_imp = (privacy_improvement['Unweighted_Test_F1'].iloc[-1] - 
                                privacy_improvement['Unweighted_Test_F1'].iloc[0])
            
            metrics_data.append(['Total Improvement', 
                               f"{classical_total_imp:+.3f}",
                               f"{privacy_total_imp:+.3f}"])
            
            metrics_data.append(['Avg Training Time', 
                               f"{classical_improvement['Training_Time'].mean():.1f}s",
                               f"{privacy_improvement['Training_Time'].mean():.1f}s"])
        
        # Privacy metrics
        privacy_config = self.privacy_data.get('config', {})
        privacy_infra = privacy_config.get('privacy_infrastructure', {})
        
        if privacy_infra:
            metrics_data.append(['Privacy Budget (ε)', 
                               'N/A',
                               f"{privacy_infra.get('dp_epsilon_total', 'N/A')}"])
            metrics_data.append(['Noise Multiplier', 
                               'N/A',
                               f"{privacy_infra.get('dp_noise_multiplier', 'N/A'):.4f}"])
        
        # Create table
        if metrics_data:
            table_data = [['Metric', 'Classical FL', 'Privacy-Preserving FL']] + metrics_data
            
            table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                           cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color code performance differences
            for i in range(1, len(table_data)):
                if len(table_data[i]) >= 3 and table_data[i][0] in ['Final F1-Score', 'Final Accuracy', 'Final ROC-AUC']:
                    try:
                        classical_val = float(table_data[i][1])
                        privacy_val = float(table_data[i][2])
                        if privacy_val > classical_val:
                            table[(i, 2)].set_facecolor('#C8E6C9')  # Light green
                        elif privacy_val < classical_val:
                            table[(i, 2)].set_facecolor('#FFCDD2')  # Light red
                    except ValueError:
                        pass
        
        ax.set_title('Key Metrics Comparison', fontweight='bold', pad=20)
    
    def generate_summary_report(self) -> None:
        """Generate a comprehensive text summary report."""
        report_path = os.path.join(VISUALIZATION_OUTPUT_DIR, 'comparison_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Federated Learning Comparison Report\n")
            f.write("## Classical vs Privacy-Preserving Approaches\n\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            self._write_executive_summary(f)
            
            # Performance Analysis
            f.write("## Performance Analysis\n\n")
            self._write_performance_analysis(f)
            
            # Privacy Analysis
            f.write("## Privacy Analysis\n\n")
            self._write_privacy_analysis(f)
            
            # Training Analysis
            f.write("## Training Analysis\n\n")
            self._write_training_analysis(f)
            
            # Recommendations
            f.write("## Recommendations\n\n")
            self._write_recommendations(f)
        
        logger.info(f"Generated comprehensive report: {report_path}")
    
    def _write_executive_summary(self, f) -> None:
        """Write executive summary section."""
        classical_perf = self.classical_data.get('performance', pd.DataFrame())
        privacy_perf = self.privacy_data.get('performance', pd.DataFrame())
        
        if not classical_perf.empty and not privacy_perf.empty:
            classical_avg = classical_perf[classical_perf['Region'] == 'AVERAGE']
            privacy_avg = privacy_perf[privacy_perf['Region'] == 'AVERAGE']
            
            if not classical_avg.empty and not privacy_avg.empty:
                classical_f1 = classical_avg['MLP_Federated_F1'].iloc[0]
                privacy_f1 = privacy_avg['MLP_Federated_F1'].iloc[0]
                f1_gap = classical_f1 - privacy_f1
                
                f.write(f"- **Classical FL Performance**: F1-Score = {classical_f1:.3f}\n")
                f.write(f"- **Privacy-Preserving FL Performance**: F1-Score = {privacy_f1:.3f}\n")
                f.write(f"- **Privacy Cost**: {f1_gap:+.3f} F1-Score points\n")
                
                if abs(f1_gap) < 0.05:
                    f.write("- **Assessment**: Privacy preservation achieved with minimal utility loss\n")
                elif f1_gap > 0:
                    f.write("- **Assessment**: Privacy preservation comes with moderate utility cost\n")
                else:
                    f.write("- **Assessment**: Privacy-preserving approach shows competitive performance\n")
        
        f.write("\n")
    
    def _write_performance_analysis(self, f) -> None:
        """Write performance analysis section."""
        classical_perf = self.classical_data.get('performance', pd.DataFrame())
        privacy_perf = self.privacy_data.get('performance', pd.DataFrame())
        
        if not classical_perf.empty and not privacy_perf.empty:
            # Regional analysis
            classical_regional = classical_perf[~classical_perf['Region'].isin(['AVERAGE', 'WEIGHTED_AVERAGE'])]
            privacy_regional = privacy_perf[~privacy_perf['Region'].isin(['AVERAGE', 'WEIGHTED_AVERAGE'])]
            
            f.write("### Regional Performance\n\n")
            f.write("| Region | Classical F1 | Privacy F1 | Difference |\n")
            f.write("|--------|-------------|------------|------------|\n")
            
            for _, row in classical_regional.iterrows():
                region = row['Region']
                classical_f1 = row['MLP_Federated_F1']
                privacy_row = privacy_regional[privacy_regional['Region'] == region]
                if not privacy_row.empty:
                    privacy_f1 = privacy_row['MLP_Federated_F1'].iloc[0]
                    diff = privacy_f1 - classical_f1
                    f.write(f"| {region} | {classical_f1:.3f} | {privacy_f1:.3f} | {diff:+.3f} |\n")
            
            f.write("\n")
            
            # Variance analysis
            classical_std = classical_regional['MLP_Federated_F1'].std()
            privacy_std = privacy_regional['MLP_Federated_F1'].std()
            
            f.write("### Performance Consistency\n\n")
            f.write(f"- **Classical FL Std Dev**: {classical_std:.3f}\n")
            f.write(f"- **Privacy-Preserving FL Std Dev**: {privacy_std:.3f}\n")
            
            if privacy_std < classical_std:
                f.write("- **Assessment**: Privacy-preserving approach shows more consistent performance\n")
            else:
                f.write("- **Assessment**: Classical approach shows more consistent performance\n")
        
        f.write("\n")
    
    def _write_privacy_analysis(self, f) -> None:
        """Write privacy analysis section."""
        privacy_config = self.privacy_data.get('config', {})
        privacy_infra = privacy_config.get('privacy_infrastructure', {})
        
        f.write("### Privacy Configuration\n\n")
        if privacy_infra:
            f.write(f"- **Privacy Budget (ε)**: {privacy_infra.get('dp_epsilon_total', 'N/A')}\n")
            f.write(f"- **Privacy Budget (δ)**: {privacy_infra.get('dp_delta', 'N/A')}\n")
            f.write(f"- **Noise Multiplier**: {privacy_infra.get('dp_noise_multiplier', 'N/A')}\n")
            f.write(f"- **Shamir Threshold**: {privacy_infra.get('shamir_threshold', 'N/A')}-of-{privacy_infra.get('shamir_participants', 'N/A')}\n")
            f.write(f"- **Double DP Prevention**: {privacy_infra.get('double_dp_prevention_enabled', False)}\n")
        
        f.write("\n### Privacy Guarantees\n\n")
        f.write("- Individual client updates never revealed to server\n")
        f.write("- Differential privacy applied with RDP composition\n")
        f.write("- Secure aggregation with dropout recovery\n")
        f.write("- Per-parameter seed shares for correct mask reconstruction\n")
        
        # Dropout analysis
        privacy_history = self.privacy_data.get('history', [])
        dropout_rounds = [r for r in privacy_history if r.get('dropout_test', False)]
        
        if dropout_rounds:
            f.write("\n### Dropout Recovery Testing\n\n")
            f.write(f"- **Dropout test rounds**: {len(dropout_rounds)}\n")
            for round_info in dropout_rounds:
                round_num = round_info['round'] + 1
                dropped_count = len(round_info.get('dropped_regions', []))
                f1_score = round_info['average_test_f1']
                f.write(f"- Round {round_num}: {dropped_count} clients dropped, F1={f1_score:.3f}\n")
        
        f.write("\n")
    
    def _write_training_analysis(self, f) -> None:
        """Write training analysis section."""
        classical_improvement = self.classical_data.get('improvement', pd.DataFrame())
        privacy_improvement = self.privacy_data.get('improvement', pd.DataFrame())
        
        if not classical_improvement.empty and not privacy_improvement.empty:
            f.write("### Training Progression\n\n")
            
            # Total improvement
            classical_total = (classical_improvement['Unweighted_Test_F1'].iloc[-1] - 
                             classical_improvement['Unweighted_Test_F1'].iloc[0])
            privacy_total = (privacy_improvement['Unweighted_Test_F1'].iloc[-1] - 
                           privacy_improvement['Unweighted_Test_F1'].iloc[0])
            
            f.write(f"- **Classical Total Improvement**: {classical_total:+.3f}\n")
            f.write(f"- **Privacy Total Improvement**: {privacy_total:+.3f}\n")
            
            # Training efficiency
            classical_avg_time = classical_improvement['Training_Time'].mean()
            privacy_avg_time = privacy_improvement['Training_Time'].mean()
            
            f.write(f"- **Classical Avg Training Time**: {classical_avg_time:.1f}s per round\n")
            f.write(f"- **Privacy Avg Training Time**: {privacy_avg_time:.1f}s per round\n")
            
            time_overhead = ((privacy_avg_time - classical_avg_time) / classical_avg_time * 100)
            f.write(f"- **Privacy Time Overhead**: {time_overhead:+.1f}%\n")
        
        f.write("\n")
    
    def _write_recommendations(self, f) -> None:
        """Write recommendations section."""
        f.write("### Technical Recommendations\n\n")
        
        # Performance-based recommendations
        classical_perf = self.classical_data.get('performance', pd.DataFrame())
        privacy_perf = self.privacy_data.get('performance', pd.DataFrame())
        
        if not classical_perf.empty and not privacy_perf.empty:
            classical_avg = classical_perf[classical_perf['Region'] == 'AVERAGE']
            privacy_avg = privacy_perf[privacy_perf['Region'] == 'AVERAGE']
            
            if not classical_avg.empty and not privacy_avg.empty:
                f1_gap = (classical_avg['MLP_Federated_F1'].iloc[0] - 
                         privacy_avg['MLP_Federated_F1'].iloc[0])
                
                if abs(f1_gap) < 0.02:
                    f.write("- **Privacy-preserving FL is recommended** for privacy-sensitive applications\n")
                    f.write("- Minimal utility loss makes privacy guarantees worthwhile\n")
                elif f1_gap > 0.05:
                    f.write("- **Consider privacy budget optimization** to reduce utility loss\n")
                    f.write("- Evaluate if stronger privacy guarantees are necessary\n")
                else:
                    f.write("- **Privacy-preserving FL shows good utility-privacy trade-off**\n")
                    f.write("- Consider for applications requiring formal privacy guarantees\n")
        
        f.write("\n### Implementation Recommendations\n\n")
        f.write("- Double DP noise prevention ensures correct privacy calibration\n")
        f.write("- Per-parameter seed shares enable robust dropout recovery\n")
        f.write("- Enhanced logging facilitates DP parameter tuning\n")
        
        f.write("\n### Future Work\n\n")
        f.write("- Consider adaptive privacy budgets based on model convergence\n")
        f.write("- Explore client-specific privacy requirements\n")
        f.write("- Investigate advanced aggregation methods for better utility\n")
        f.write("- Implement privacy accounting for production deployment\n")
    
    def run_complete_analysis(self) -> None:
        """Run the complete analysis and generate all visualizations."""
        logger.info("Starting comprehensive federated learning analysis")
        
        # Load all data
        logger.info("Loading experimental results...")
        classical_loaded = self.load_classical_results()
        privacy_loaded = self.load_privacy_results()
        centralized_loaded = self.load_centralized_baseline()
        
        if not classical_loaded:
            logger.error("Failed to load classical results")
            return
        
        if not privacy_loaded:
            logger.error("Failed to load privacy results")
            return
        
        # Generate visualizations
        logger.info("Generating performance comparison...")
        self.create_performance_comparison()
        
        logger.info("Generating training progression analysis...")
        self.create_training_progression_analysis()
        
        logger.info("Generating privacy analysis...")
        self.create_privacy_analysis()
        
        logger.info("Generating comprehensive summary...")
        self.create_comprehensive_summary()
        
        logger.info("Generating summary report...")
        self.generate_summary_report()
        
        logger.info(f"Analysis complete! All visualizations saved to: {VISUALIZATION_OUTPUT_DIR}")


def main():
    """Main function to run the complete analysis."""
    analyzer = FederatedResultsAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()