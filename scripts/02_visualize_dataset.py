#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 02_visualize_dataset.py
Description: Esegue l'analisi visiva del dataset Heart Disease UCI processato.
             Sfrutta le utility centralizzate per generare grafici ad alta risoluzione.
Output: Plot PNG in 'results/visualizations/'
"""

import sys 
import os
import pandas as pd

# Aggiunge la root del progetto al sys.path per importare le utility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importiamo le funzioni dalla tua cartella utils
from utils.visualization import visualize_distribution, visualize_correlations

# Percorsi configurati
TRAINING_DATA_PATH = "data/processed/Enhanced_Training_Dataset.csv"
OUTPUT_DIR = "results/visualizations"

def main():
    # 1. Verifica l'esistenza del dataset prodotto dallo script 01
    if not os.path.exists(TRAINING_DATA_PATH):
        print(f"❌ Errore: Dataset non trovato in {TRAINING_DATA_PATH}")
        print("💡 Esegui prima 'python scripts/01_generate_dataset.py'")
        sys.exit(1)

    # 2. Assicura l'esistenza della cartella di output per la tesi
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Caricamento dati
    print("📊 Caricamento dei dati clinici per analisi visiva...")
    df = pd.read_csv(TRAINING_DATA_PATH)

    # Riepilogo rapido a terminale
    print(f"✅ Caricati {df.shape[0]} record con {df.shape[1]} variabili cliniche.")
    print(f"🩺 Variabili rilevate: {', '.join(df.columns.tolist())}")

    # 4. Generazione Grafici tramite UTILS
    
    # Visualizzazione del bilanciamento del Target (Sani vs Malati)
    print("📈 Generazione grafico distribuzione target...")
    visualize_distribution(df, save_dir=OUTPUT_DIR)

    # Analisi delle correlazioni (es. Colesterolo, Pressione, Età rispetto al Target)
    print("🔗 Generazione heatmap delle correlazioni cliniche...")
    visualize_correlations(df, save_dir=OUTPUT_DIR)

    print(f"\n✨ Visualizzazione completata. I grafici per la tesi sono disponibili in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()