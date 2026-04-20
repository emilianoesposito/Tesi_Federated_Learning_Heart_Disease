#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 01_generate_dataset.py
Descrizione: Implementazione rigorosa della separazione dei dati. 
             Estrae un Test Set Globale (20%) per la validazione finale 
             e ripartisce il restante 80% tra i 5 nodi ospedalieri.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)

from utils.scoring import ClinicalRiskScorer
from utils.federated_data_splitter import CardiacFederatedSplitter

def main():
    print("🚀 Script 01: Generazione Dataset e Test Set Globale...")
    raw_path = "data/raw/Dataset_Pazienti.csv"
    
    if not os.path.exists(raw_path):
        print(f"❌ Errore: Manca {raw_path}"); return
        
    df_raw = pd.read_csv(raw_path)
    scorer = ClinicalRiskScorer()
    df_processed = scorer.compute_risk_scores(df_raw)
    
    # Identificazione dinamica colonna target
    target_col = 'target_label' if 'target_label' in df_processed.columns else df_processed.columns[-1]
    print(f"🎯 Colonna target identificata: {target_col}")

    # Separazione Test Set Globale (20%)
    print("⚖️  Separazione del 20% dei dati per il Test Set Globale...")
    df_train_federated, df_global_test = train_test_split(
        df_processed, test_size=0.20, random_state=42, stratify=df_processed[target_col]
    )
    
    # Rinomina per coerenza totale negli script successivi
    df_train_federated = df_train_federated.rename(columns={target_col: 'target_label'})
    df_global_test = df_global_test.rename(columns={target_col: 'target_label'})

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    
    df_train_federated.to_csv("data/processed/Enhanced_Training_Dataset.csv", index=False)
    df_global_test.to_csv("data/test/global_test_set.csv", index=False)
    
    print("🏥 Ripartizione tra i nodi ospedalieri...")
    splitter = CardiacFederatedSplitter()
    splitter.split_data(df_train_federated) 
    print("✅ Dataset pronti.")

if __name__ == "__main__":
    main()