#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 03_train_models.py
Descrizione: Addestramento della Baseline globale. 
             CORREZIONE: Valutazione sul Test Set Globale (20%) separato nello Script 01.
"""

import sys, os, joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    train_path = "data/processed/Enhanced_Training_Dataset.csv"
    test_path = "data/test/global_test_set.csv"
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train, y_train = df_train.drop('target_label', axis=1), df_train['target_label']
    X_test, y_test = df_test.drop('target_label', axis=1), df_test['target_label']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("results", exist_ok=True)
    joblib.dump(scaler, "results/global_scaler.joblib")
    
    metrics = []
    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_test_scaled)
    metrics.append({'model': 'MLP', 'accuracy': accuracy_score(y_test, y_pred), 'f1_score': f1_score(y_test, y_pred)})
    joblib.dump(mlp, "results/MLP.joblib")

    # LightGBM
    lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    lgbm.fit(X_train_scaled, y_train)
    y_pred = lgbm.predict(X_test_scaled)
    metrics.append({'model': 'LightGBM', 'accuracy': accuracy_score(y_test, y_pred), 'f1_score': f1_score(y_test, y_pred)})
    joblib.dump(lgbm, "results/LightGBM.joblib")

    pd.DataFrame(metrics).to_csv("results/metrics_summary.csv", index=False)
    print("✅ Baseline addestrata e valutata su Test Set Globale.")

if __name__ == "__main__":
    main()