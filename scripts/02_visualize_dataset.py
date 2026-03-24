#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 02_visualize_dataset.py
Description: Esegue l'analisi visiva del dataset Heart Disease UCI processato.
             Genera grafici sulla distribuzione delle patologie e correlazioni cliniche.
Output: Plot PNG in 'results/visualizations/' e riepiloghi statistici.
"""

import sys 
import os
import pandas as pd

# Aggiunge la root del progetto al sys.path per importare le utility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.visualization import visualize_distribution, visualize_correlations

# Percorso del dataset generato dallo script 01
TRAINING_DATA_PATH = "data/processed/Enhanced_Training_Dataset.csv"
OUTPUT_DIR = "results/visualizations"

def main():
    # Verifica l'esistenza del file
    if not os.path.exists(TRAINING_DATA_PATH):
        print(f"❌ Errore: Dataset non trovato in {TRAINING_DATA_PATH}")
        print("💡 Esegui prima 'python scripts/01_generate_dataset.py'")
        exit(1)

    # Assicura l'esistenza della cartella di output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("📊 Caricamento dei dati clinici per analisi visiva...")
    df = pd.read_csv(TRAINING_DATA_PATH)

    # Stampa un'anteprima delle feature (Analisi feature)
    print(f"✅ Caricati {df.shape[0]} record con {df.shape[1]} variabili.")
    print(f"🩺 Variabili rilevate: {', '.join(df.columns.tolist())}")

    print("📈 Generazione grafici di distribuzione e bilanciamento target...")
    # Questa funzione mostrerà il bilanciamento tra Sani (0) e Malati (1)
    visualize_distribution(df)

    print("🔗 Analisi delle correlazioni tra parametri clinici (es. Colesterolo vs Target)...")
    # Questa funzione genererà una heatmap delle correlazioni
    visualize_correlations(df)

    print("-" * 30)
    print(f"✅ Visualizzazioni salvate nella cartella: {OUTPUT_DIR}")
    print("💡 Controlla il grafico del target per verificare la necessità di SMOTE.")
    print("-" * 30)

if __name__ == "__main__":
    main()
