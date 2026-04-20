# utils/federated_learning.py
# -*- coding: utf-8 -*-

import os
import joblib
import pandas as pd

class CardiacFederatedManager:
    """
    Manager per gestire l'addestramento simulato sui nodi ospedalieri.
    Memorizza i modelli locali e i metadati per la successiva aggregazione.
    """
    
    def __init__(self):
        # Lista che conterrà i dizionari con i risultati di ogni ospedale
        self.local_results = []

    def train_node(self, hospital_name, data_path, model_obj, scaler=None):
        """
        Carica i dati di un singolo ospedale, addestra il modello e salva il risultato.
        """
        # 1. Caricamento dati
        df = pd.read_csv(data_path)
        
        # 2. Protezione e uniformazione dei nomi delle colonne
        if 'target_label' in df.columns:
            target_col = 'target_label'
        elif 'outcome' in df.columns:
            df = df.rename(columns={'outcome': 'target_label'})
            target_col = 'target_label'
        else:
            # Caso estremo: se il CSV non ha header, assegniamo i nomi della tesi
            df.columns = [
                'eta', 'sesso', 'dolore_toracico', 'pressione', 'colesterolo', 
                'glicemia', 'ecg_riposo', 'freq_max', 'angina_sforzo', 
                'depressione_st', 'pendenza_st', 'vasi_colorati', 'talassemia', 'target_label'
            ]
            target_col = 'target_label'

        # 3. Definizione delle feature (X) e del target (y)
        # Ora siamo certi che target_col esista e sia coerente
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 4. Applicazione dello scaler se fornito (fondamentale per MLP)
        if scaler:
            # transform() restituisce un array numpy, perfetto per fit()
            X = scaler.transform(X)

        # 5. Addestramento del modello locale
        # Nota: usiamo y.values.ravel() per evitare warning di Scikit-Learn
        model_obj.fit(X, y.values.ravel() if hasattr(y, 'values') else y)
        
        # 6. Salvataggio nei risultati locali del manager
        self.local_results.append({
            'hospital': hospital_name, 
            'model': model_obj, 
            'size': len(df)
        })

    def save_local_models(self, output_path):
        """
        Metodo per salvare su disco tutti i modelli addestrati nei nodi.
        Risolve l'AttributeError riscontrato negli script di addestramento.
        """
        if not self.local_results:
            print(f"⚠️ Attenzione: Nessun risultato trovato per {output_path}. Esegui prima train_node().")
            return
            
        # Crea la cartella se non esiste
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Salva l'intera lista di modelli e metadati
        joblib.dump(self.local_results, output_path)
        print(f"✅ Modelli locali ({len(self.local_results)}) salvati correttamente in: {output_path}")

    def get_local_results(self):
        """Restituisce la lista dei risultati addestrati."""
        return self.local_results