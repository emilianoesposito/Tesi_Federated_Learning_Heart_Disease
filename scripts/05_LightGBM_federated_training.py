#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 05_LightGBM_federated_training.py
Descrizione: Addestramento locale dei nodi ospedalieri con LightGBM.
             Utilizza 'CardiacFederatedManager' per gestire i 5 nodi ospedalieri.
Output: 
 - Modelli locali salvati in 'results/federated/local_lightgbm_models.joblib'
"""

import os, sys, joblib
import lightgbm as lgb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.federated_learning import CardiacFederatedManager

def main():
    scaler = joblib.load("results/global_scaler.joblib")
    manager = CardiacFederatedManager()
    hospitals = ['Ospedale_Roma', 'Ospedale_Milano', 'Ospedale_Napoli', 'Ospedale_Firenze', 'Ospedale_Rimini']

    for h in hospitals:
        path = f"data/federated/{h}_training_data.csv"
        model = lgb.LGBMClassifier(n_estimators=100, is_unbalance=True, random_state=42, verbose=-1)
        manager.train_node(h, path, model, scaler=scaler)
    
    os.makedirs("results/federated", exist_ok=True)
    manager.save_local_models("results/federated/local_lightgbm_models.joblib")
    print("✅ Modelli LightGBM locali salvati.")

if __name__ == "__main__":
    main()