# utils/federated_learning.py
# -*- coding: utf-8 -*-
import os
import joblib
import pandas as pd

class CardiacFederatedManager:
    def __init__(self):
        self.local_results = []

    def train_node(self, hospital_name, data_path, model_obj, scaler=None):
        df = pd.read_csv(data_path)
        
        # Protezione: se mancano i nomi delle colonne, li forziamo
        if 'target_label' not in df.columns:
            if 'outcome' in df.columns:
                df = df.rename(columns={'outcome': 'target_label'})
            else:
                # Se il CSV è senza header (solo numeri), assegniamo i nomi corretti
                df.columns = ['eta', 'sesso', 'dolore_toracico', 'pressione', 'colesterolo', 
                             'glicemia', 'ecg_riposo', 'freq_max', 'angina_sforzo', 
                             'depressione_st', 'pendenza_st', 'vasi_colorati', 'talassemia', 'target_label']

        X = df.drop(columns=['target_label'])
        y = df['target_label']

        if scaler:
            X = scaler.transform(X)

        model_obj.fit(X, y)
        self.local_results.append({'hospital': hospital_name, 'model': model_obj, 'size': len(df)})
        return self.local_results[-1]

    def save_results(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.local_results, path)