#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 05_LightGBM_federated_training.py
Descrizione: Addestramento locale dei nodi ospedalieri con LightGBM.
             Utilizza 'CardiacFederatedManager' per gestire i 5 nodi ospedalieri.
Output: 
 - Modelli locali salvati in 'results/federated/local_lightgbm_models.joblib'
"""

import os
import sys
import joblib
import lightgbm as lgb

# --- FIX PER MODULENOTFOUNDERROR (Necessario per Windows) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------------------------------------

from utils.federated_learning import CardiacFederatedManager

def main():
    print("🏥 Avvio Addestramento Federato Locale (LightGBM)...")
    
    SCALER_PATH = "results/global_scaler.joblib"
    RESULTS_DIR = "results/federated"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(SCALER_PATH):
        print("❌ Errore: Esegui prima lo script 03 per generare lo scaler!")
        return

    scaler = joblib.load(SCALER_PATH)
    manager = CardiacFederatedManager()
    hospitals = ['Ospedale_Roma', 'Ospedale_Milano', 'Ospedale_Napoli', 'Ospedale_Firenze', 'Ospedale_Rimini']

    for h in hospitals:
        path = f"data/federated/{h}_training_data.csv"
        if not os.path.exists(path):
            print(f"⚠️ Salto {h}: file non trovato.")
            continue
            
        # Parametri ottimizzati per dataset piccoli e non bilanciati
        model = lgb.LGBMClassifier(n_estimators=50, is_unbalance=True, random_state=42, verbose=-1)
        
        print(f"📡 Training su {h:16}...", end=" ", flush=True)
        manager.train_node(h, path, model, scaler=scaler)
        print("✅ Completato")

    # Salvataggio finale di tutti i modelli locali
    output_file = os.path.join(RESULTS_DIR, "local_lightgbm_models.joblib")
    manager.save_results(output_file)
    print(f"\n✨ Modelli locali salvati con successo in: {output_file}")

if __name__ == "__main__":
    main()