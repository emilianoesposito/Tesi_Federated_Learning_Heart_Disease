# utils/feature_engineering.py
# -*- coding: utf-8 -*-
"""
Utility di Feature Engineering per il dataset Heart Disease UCI.
Include:
 - Pre-processing delle variabili cliniche.
 - Standardizzazione delle feature (StandardScaler).
 - Bilanciamento del dataset tramite SMOTE per gestire campioni ridotti. 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def prepare_data_for_training(df, target_col='target_label', test_size=0.2, random_state=42):
    """
    Prepara i dati clinici per l'addestramento della Baseline.
    Esegue lo split, la standardizzazione e applica SMOTE.
    """
    print(f"🧹 Inizio pre-processing su {len(df)} record...")

    # 1. Selezione delle Feature (le 13 variabili cliniche richieste)
    # Rimuoviamo il target e eventuali colonne di servizio come 'outcome'
    X = df.drop(columns=[target_col, 'outcome'], errors='ignore')
    y = df[target_col]

    # 2. Split Train/Test con stratificazione (mantiene le proporzioni sani/malati)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. Standardizzazione (Essenziale per il modello MLP) 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Bilanciamento tramite SMOTE
    # Utile per compensare i pochi record (circa 60 per ogni nodo futuro)
    print(f"⚖️ Applicazione di SMOTE su set di training (Originale: {len(X_train)} record)...")
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

    print(f"✅ Pre-processing completato. Training set bilanciato: {len(X_res)}record.")

    return {
        'X_train': X_res,
        'X_test': X_test_scaled,
        'y_train': y_res,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }

# Rimuoviamo le vecchie funzioni non più necessarie
# extend_candidates_dataset e extend_companies_dataset sono state eliminate