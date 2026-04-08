# utils/feature_engineering.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def prepare_data_for_training(df, target_col='target_label', test_size=0.2, apply_smote=True):
    """Esegue split, scaling e bilanciamento con gestione sicura del target."""
    
    # --- FIX SICUREZZA ---
    actual_target = None
    if target_col in df.columns:
        actual_target = target_col
    elif 'outcome' in df.columns:
        actual_target = 'outcome'
    else:
        actual_target = df.columns[-1] # Prende l'ultima colonna come fallback
    # ---------------------

    X = df.drop(columns=[actual_target])
    y = df[actual_target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        return X_train_res, X_test_scaled, y_train_res, y_test, scaler
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler