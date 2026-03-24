# utils/scoring.py
# -*- coding: utf-8 -*-
"""
Logica di scoring e adattamento per il dataset Heart Disease UCI.
Include:
 - Mapping delle variabili cliniche originali verso il sistema tesi.
 - Gestione dei valori mancanti (imputazione con mediana).
 - Binarizzazione del target clinico per screening.
"""

import pandas as pd
import numpy as np

class ClinicalRiskScorer:
    def __init__(self):
        # Mappatura tra nomi originali UCI e nomi variabili del sistema
        self.feature_mapping = {
            'age': 'eta',
            'trestbps': 'pressione',
            'chol': 'colesterolo',
            'target': 'outcome'
        }

    def compute_risk_scores(self, df):
        """
        Adatta il dataset Heart Disease UCI per la tesi.
        Effettua il mapping delle 13 variabili cliniche e normalizza il target.
        """
        print("🧪 Analisi e adattamento variabili cliniche Heart Disease UCI...")

        # 1. Rinominazione delle colonne per uniformità con il sistema originale
        df = df.rename(columns=self.feature_mapping)

        # 2. Gestione valori mancanti
        # Sostituzione dei valori nulli con la mediana per preservare i 303 record
        if df.isnull().values.any():
            df = df.fillna(df.median())

        # 3. Trasformazione del target in binario
        # Sostituzione dell'indice sintetico con l'outcome clinico reale.
        # Introduzione di un 'Health_Score' ausiliario per il feature engineering, se necessario.
        if 'outcome' in df.columns:
            # UCI: 0 = sano, 1-4: gradi di malattia. Trasformiamo in 0 vs 1.
            df['target_label'] = (df['outcome'] > 0).astype(int)

            # Esempio di Health_Score ausiliario basato su pressione e colesterolo
            df['health_score'] = 1.0 - (
                (df['pressione'] / df['pressione'].max()) * 0.5 +
                (df['colesterolo'] / df['colesterolo'].max()) * 0.5
            )
        
        return df
    
    def generate_enhanced_training_data(self, df_raw):
        """
        Metodo wrapper per mantenere compatibilità con lo script 01_generate_dataset.py
        """
        return self.compute_risk_scores(df_raw)