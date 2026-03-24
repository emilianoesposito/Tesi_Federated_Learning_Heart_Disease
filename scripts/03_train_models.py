#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 03_train_models.py
Description: Addestramento della Baseline con MLP e LightGBM.
             Include il bilanciamento SMOTE e il calcolo delle metriche richieste.
Output: 
 - Modelli salvati in 'results'
 - Tabella riassuntiva 'results/metrics_summary.csv'          
"""

import sys
import os
import joblib 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

# Aggiunta root progetto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

TRAINING_FILE = 'data/processed/Enhanced_Training_Dataset.csv'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_baseline():
    if not os.path.exists(TRAINING_FILE):
        print(f"❌ Errore: File {TRAINING_FILE} non trovato. Esegui lo script 01.")
        return
    
    # 1. Caricamento Dati
    print("📥 Caricamento dataset per la Baseline...")
    df = pd.read_csv(TRAINING_FILE)

    # Definizione feature (le 13 variabili cliniche) e Target
    X = df.drop(columns=['target_label', 'outcome'], errors='ignore')
    y = df['target_label']

    # 2. Split e Bilanciamento
    # Usiamo lo split standard 80/20 prima della partizione federata
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"⚖️ Applicazione SMOTE per bilanciare i {len(X_train)} record di training...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"✅ Bilanciamento completato: {len(X_res)} campioni generati.")

    # 3. Definizione Modelli (MLP e LightGBM)
    models = {
        'MLP_Baseline': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        'LightGBM_Baseline': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    }

    metrics_rows = []

    # 4. Training e Valutazione
    for name, model in models.items():
        print(f"🚀 Addestramento modello: {name}...")
        model.fit(X_res, y_res)

        # Predizioni
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calcolo Metriche 
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        print(f"📊 Risultati {name}: Acc={acc:.2f}, F1={f1:.2f}, ROC-AUC={roc:.2f}")

        metrics_rows.append({
            'model': name,
            'accuracy': round(acc, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc, 4)
        })

        # Salvataggio modello
        joblib.dump(model, os.path.join(RESULTS_DIR, f"{name}.joblib"))

    # 5. Salvataggio Report Finale
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(os.path.join(RESULTS_DIR, 'metrics_summary.csv'), index=False)
    print(f"✅ Baseline completata. Risultati salvati in {RESULTS_DIR}/metrics_summary.csv")

if __name__ == "__main__":
    train_baseline()


