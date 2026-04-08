#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 04_analyze_results.py
Description: Genera il confronto visivo tra MLP e LightGBM (Baseline).
             Utilizza le utility di visualizzazione per produrre grafici professionali.
Output: PNG plots in 'results/visualizations/'
"""

import os
import sys
import pandas as pd

# Aggiunta root progetto per importare le utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.visualization import visualize_metrics_comparison

# === Configurazione Percorsi ===
METRICS_PATH = "results/metrics_summary.csv"
OUTPUT_DIR = "results/visualizations"

def main():
    # 1. Verifica esistenza dei risultati della Baseline (Script 03)
    if not os.path.exists(METRICS_PATH):
        print(f"❌ Errore: Risultati non trovati in {METRICS_PATH}.")
        print("💡 Esegui prima lo script '03_train_models.py'.")
        return
    
    # 2. Creazione cartella di output se non esiste
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Caricamento delle metriche
    print("📊 Caricamento metriche Baseline per il confronto visivo...")
    df_metrics = pd.read_csv(METRICS_PATH)

    # Mostriamo un'anteprima a terminale
    print("\n📈 Riepilogo Performance Baseline:")
    print(df_metrics.to_string(index=False))

    # 4. Generazione del grafico comparativo tramite UTILS
    # La funzione in utils gestisce automaticamente il formato 'long' (melt),
    # i colori, le etichette e il salvataggio ad alta risoluzione.
    print("\n🎨 Generazione del grafico a barre comparativo...")
    visualize_metrics_comparison(
        df_metrics, 
        save_dir=OUTPUT_DIR, 
        output_name="baseline_model_comparison.png"
    )

    print(f"\n✅ Analisi completata. Il grafico è stato salvato in: {OUTPUT_DIR}/baseline_model_comparison.png")

if __name__ == "__main__":
    main()