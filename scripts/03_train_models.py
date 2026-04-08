#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 03_train_models.py
Description: Addestramento della Baseline globale con MLP e LightGBM.
             Utilizza le utility per il pre-processing e il monitoraggio risorse.
Output: 
 - Modelli e Scaler salvati in 'results/'
 - Metriche salvate in 'results/metrics_summary.csv'          
"""

import sys
import os
import joblib 
import pandas as pd
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb

# Fix per i percorsi
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.feature_engineering import prepare_data_for_training

def main():
    print("🧠 Addestramento Modelli Baseline (Globale)...")
    
    # 1. Caricamento dati
    input_path = "data/processed/Enhanced_Training_Dataset.csv"
    if not os.path.exists(input_path):
        print(f"❌ Errore: File {input_path} non trovato!")
        return
    
    df = pd.read_csv(input_path)
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_training(df)
    
    # Crea cartella risultati
    os.makedirs("results", exist_ok=True)
    
    # Salva lo scaler (fondamentale per dopo!)
    joblib.dump(scaler, "results/global_scaler.joblib")
    
    metrics_list = []

    # 2. Addestramento MLP (Rete Neurale)
    print("🚀 Training MLP...")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    
    metrics_list.append({
        'model': 'MLP',
        'accuracy': accuracy_score(y_test, y_pred_mlp),
        'f1_score': f1_score(y_test, y_pred_mlp)
    })
    joblib.dump(mlp, "results/MLP.joblib")

    # 3. Addestramento LightGBM
    print("🚀 Training LightGBM...")
    lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)
    y_pred_lgbm = lgbm.predict(X_test)
    
    metrics_list.append({
        'model': 'LightGBM',
        'accuracy': accuracy_score(y_test, y_pred_lgbm),
        'f1_score': f1_score(y_test, y_pred_lgbm)
    })
    joblib.dump(lgbm, "results/LightGBM.joblib")

    # 4. Salvataggio Metriche per lo script 04
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv("results/metrics_summary.csv", index=False)
    
    print("\n✅ Modelli salvati e 'results/metrics_summary.csv' generato!")

if __name__ == "__main__":
    main()