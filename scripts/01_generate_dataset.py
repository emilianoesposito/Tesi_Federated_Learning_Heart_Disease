#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 01_generate_dataset.py
Description: Carica il dataset Heart Disease UCI, applica il mapping delle feature 
             e genera il dataset processato per la Baseline
Output:
  - data/processed/Enhanced_Training_Dataset.csv
"""

import os
import sys
import pandas as pd

# Aggiunge la root del progetto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa lo scorer adattato per i dati UCI
from utils.scoring import ClinicalRiskScorer

# Definizione percorsi (Analisi e Adattamento)
RAW_DATA_PATH = "data/raw/Dataset_Pazienti.csv"
TRAINING_DATA_PATH = "data/processed/Enhanced_Training_Dataset.csv"

def main():
    # Verifica esistenza cartelle
    os.makedirs("data/processed", exist_ok=True)

    # Step 1: Caricamento dati reali UCI (Cleveland Dataset)
    if not os.path.exists(RAW_DATA_PATH):
        print("❌ Errore: Dataset non trovato in {RAW_DATA_PATH}. Esegui prima lo script di download.")
        return
    
    print("📥 Caricamento dataset Heart DIsease UCI...")
    df_raw = pd.read_csv(RAW_DATA_PATH) 

    # Step 2: Adattamento e binarizzazione del target
    # Utilizziamo la classe ClinicalRiskScorer per documentare la corrispondenza
    print("🔧 Analisi e adattamento delle 13 variabili cliniche...")
    scorer = ClinicalRiskScorer()

    # Il metodo compute_risk_scores ora gestisce il mapping:
    # age -> eta, trestbps -> pressione, chol -> colesterolo, ecc.
    df_processed = scorer.compute_risk_scores(df_raw)

    # Step 3: Salvataggio del dataset per la Baseline
    print(f"🧠 Generazione dataset per training (N. record: {len(df_processed)})...")

    # Salvataggio finale
    df_processed.to_csv(TRAINING_DATA_PATH, index=False) 

    print("-" * 30)
    print(f"✅ Baseline Dataset salvato: {TRAINING_DATA_PATH}")
    print(f"📊 Feature rilevate: {list(df_processed.columns)}")
    print("-" * 30)

if __name__ == "__main__":
    main()

