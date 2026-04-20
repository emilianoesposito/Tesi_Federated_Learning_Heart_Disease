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

def main():
    path = "results/metrics_summary.csv"
    output_dir = "results/visualizations"
    
    if not os.path.exists(path):
        print(f"❌ Errore: File {path} non trovato. Esegui prima lo script 03!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(path)
    
    print("📈 Riepilogo Performance Baseline:")
    print(df.to_string(index=False))
    
    # CORREZIONE: Aggiunto il terzo argomento 'baseline_comparison.png'
    print("\n🎨 Generazione del grafico comparativo...")
    visualize_metrics_comparison(
        df, 
        save_dir=output_dir, 
        output_name="baseline_comparison.png"
    )
    
    print(f"✅ Grafico salvato con successo in {output_dir}")

if __name__ == "__main__":
    main()