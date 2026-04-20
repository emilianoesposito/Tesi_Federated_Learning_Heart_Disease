#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 02_visualize_dataset.py
Descrizione: Analisi visiva del dataset di TRAINING (80%) per la tesi.
             Esclude il Test Set Globale per coerenza metodologica.
"""

import sys, os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.visualization import visualize_distribution, visualize_correlations

TRAINING_DATA_PATH = "data/processed/Enhanced_Training_Dataset.csv"
OUTPUT_DIR = "results/visualizations"

def main():
    if not os.path.exists(TRAINING_DATA_PATH):
        print("❌ Errore: Esegui lo script 01!"); sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(TRAINING_DATA_PATH)
    print(f"📊 Analisi su {len(df)} record di training...")
    
    visualize_distribution(df, save_dir=OUTPUT_DIR)
    visualize_correlations(df, save_dir=OUTPUT_DIR)
    print(f"✅ Grafici salvati in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()