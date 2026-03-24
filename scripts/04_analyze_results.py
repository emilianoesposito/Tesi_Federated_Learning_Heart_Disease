#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 04_analyze_results.py
Description: Genera il confronto visivo tra MLP e LightGBM (Baseline).
             Produce grafici di Accuracy, F1-Score e ROC-AUC.
Output: PNG plots in 'results/'
"""

import os
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# === Configurazione ===
metrics_path = "results/metrics_summary.csv"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

def main():
    # Caricamento dei risultati della Baseline
    if not os.path.exists(metrics_path):
        print(f"❌ Errore: Risultati non trovati in {metrics_path}. Esegui lo script 03.")
        return
    
    print("📊 Caricamento metriche Baseline...")
    df = pd.read_csv(metrics_path)

    # Setup stile visualizzazione
    sns.set(style="whitegrid", font_scale=1.1)

    # 1. Grafico a barre comparativo per le 3 metriche chiave
    metrics = ["accuracy", "f1_score", "roc_auc"]

    # Trasformazione del dataframe in formato "long" per Seaborn
    df_melted = df.melt(id_vars="model", value_vars=metrics, var_name="Metric", value_name="Score")

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=df_melted, x="Metric", y="Score", hue="model", palette="viridis")

    plt.title("Confronto Baseline: MLP vs LightGBM", fontsize=15, pad=20)
    plt.ylabel("Punteggio (0-1)")
    plt.ylim(0, 1.1)

    # Aggiunta dei valori numerici sopra le barre
    for p in ax.patches:
        height = p.get_height()
        if height > 0: # Evita di scrivere etichette se il valore è nullo
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', # 'bottom' mette l'erichetta SOPRA la barra
                        xytext=(0, 5),            # 5 punt di spazio sopra la barra
                        textcoords='offset points',
                        fontsize=10,
                        fontweight='bold')
        
    plt.legend(title="Modello", loc="lower right")
    plt.tight_layout()

    # Salvataggio grafico principale
    plot_path = os.path.join(output_dir, "baseline_comparison.png")
    plt.savefig(plot_path)
    print(f"✅ Grafico di confronto salvato in {plot_path}")

    # 2. Stampa riassunto
    print("\n" + "="*40)
    print("📋 RISULTATI")
    print("="*40)
    print(df.to_string(index=False))
    print("="*40)

if __name__ == "__main__":
    main()

