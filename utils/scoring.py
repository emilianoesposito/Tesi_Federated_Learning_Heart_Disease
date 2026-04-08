# utils/scoring.py
import pandas as pd
import numpy as np

class ClinicalRiskScorer:
    def __init__(self):
        # Mapping UNIFICATO per tutta la tesi
        self.feature_mapping = {
            'age': 'eta', 'sex': 'sesso', 'cp': 'dolore_toracico',
            'trestbps': 'pressione', 'chol': 'colesterolo',
            'fbs': 'glicemia', 'restecg': 'ecg_riposo', 'thalach': 'freq_max',
            'exang': 'angina_sforzo', 'oldpeak': 'depressione_st',
            'slope': 'pendenza_st', 'ca': 'vasi_colorati', 'thal': 'talassemia',
            'target': 'target_label'
        }

    def compute_risk_scores(self, df):
        # Rinomina
        df = df.rename(columns=self.feature_mapping)
        # Pulizia e Binarizzazione (0: sano, 1: malato)
        if 'target_label' in df.columns:
            df['target_label'] = (df['target_label'] > 0).astype(int)
        return df.fillna(df.median())