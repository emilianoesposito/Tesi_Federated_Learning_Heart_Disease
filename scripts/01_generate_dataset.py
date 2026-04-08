#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 01_generate_dataset.py
Description: Carica il dataset Heart Disease UCI, applica il mapping clinico tramite scoring.py
             e ripartisce i dati tra i 5 nodi ospedalieri tramite federated_data_splitter.py.
Output:
  - data/processed/Enhanced_Training_Dataset.csv (Dataset per Baseline)
  - data/federated/Ospedale_*.csv (Dataset per training federato)
"""

import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.scoring import ClinicalRiskScorer
from utils.federated_data_splitter import CardiacFederatedSplitter

def main():
    print("🚀 Script 01: Rigenerazione Dataset con Header...")
    
    raw_path = "data/raw/Dataset_Pazienti.csv"
    if not os.path.exists(raw_path):
        print(f"❌ Errore: Manca {raw_path}")
        return
        
    df_raw = pd.read_csv(raw_path)
    
    # 2. Trasformazione
    scorer = ClinicalRiskScorer()
    df_processed = scorer.compute_risk_scores(df_raw)
    
    # 3. Salvataggio Centralizzato
    os.makedirs("data/processed", exist_ok=True)
    df_processed.to_csv("data/processed/Enhanced_Training_Dataset.csv", index=False)
    
    # 4. Split Federato (Assicuriamoci che mantenga i nomi colonne)
    print("🏥 Ripartizione in corso...")
    splitter = CardiacFederatedSplitter()
    splitter.split_data(df_processed) 
    
    # FORZIAMO il controllo: se i file creati non hanno header, li sovrascriviamo
    hospitals = ['Ospedale_Roma', 'Ospedale_Milano', 'Ospedale_Napoli', 'Ospedale_Firenze', 'Ospedale_Rimini']
    for h in hospitals:
        path = f"data/federated/{h}_training_data.csv"
        if os.path.exists(path):
            temp_df = pd.read_csv(path)
            # Se la prima colonna si chiama '0', significa che l'header è andato perso
            if '0' in temp_df.columns:
                 # Ripristiniamo i nomi delle colonne dal dataframe processato
                 temp_df.columns = df_processed.columns
                 temp_df.to_csv(path, index=False)
                 print(f"✅ Header ripristinati per {h}")
    
    print("\n✨ Rigenerazione completata correttamente.")

if __name__ == "__main__":
    main()