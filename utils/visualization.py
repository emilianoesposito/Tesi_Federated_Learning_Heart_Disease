# utils/visualization.py
# -*- coding: utf-8 -*-
"""
Utility di visualizzazione per il dataset Heart Disease UCI.
Include:
 - Analisi della distribuzione delle classi cliniche.
 - Heatmap delle correlazioni tra variabili mediche.
 - Confronto delle performance dei modelli (MLP vs LightGBM).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def visualize_distribution(df, save_dir='results/visualizations'):
    """Analisi del bilanciamento del target (Sani vs Malati)"""
    plt.figure(figsize=(8, 6))
    # 'target_label' è il nome definito negli script 01_generate_dataset.py e scoring.py
    ax = sns.countplot(data=df, x='target_label', palette='viridis')
    plt.title('Distribuzione Classi: Sani (0) vs Malati (1)')
    plt.xlabel('Diagnosi')
    plt.ylabel('Numero di Pazienti')

    # Aggiunta etichette
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x()+0.35, p.get_height()+1))

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'target_distribution.png'))
    plt.close()

def visualize_correlations(df, save_dir='results/visualizations'):
    """Heatmap delle correlazioni tra le 13 variabili cliniche"""
    plt.figure(figsize=(12, 10))
    # Selezioniamo solo le colonne numeriche rilevanti
    cols_to_corr = [c for c in df.columns if c not in ['outcome']]
    corr = df[cols_to_corr].corr()

    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Heatmap delle Correlazioni Cliniche')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'clinical_correlations.png'))
    plt.close()

def plot_metrics_comparison(metrics_df, save_dir='results'):
    """Confronto Accuracy, F1 e ROC-AUC"""
    # Adattiamo i nomi delle metriche usati nello script 03_train_models.py
    metrics = ['accuracy', 'f1_score', 'roc_auc']

    # Trasponiamo per il plotting, se necessario
    df_plot = metrics_df.set_index('model')[metrics]

    ax = df_plot.plot(kind='bar', figsize=(12, 7), colormap='viridis')
    plt.title('Confronto Performance Baseline: MLP vs LightGBM')
    plt.ylabel('Punteggio')
    plt.xlabel('Modello')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'baseline_metrics_comparison.png'))
    plt.close()

