# utils/federated_data_splitter.py
# -*- coding: utf-8 -*-
"""
Federated data splitter per il dataset cardiaco.
Divide il dataset in 5 ospedali simulati con proporzioni differenti.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CardiacFederatedSplitter:
    """
    Splitter per simulare la distribuzione dei dati tra 5 ospedali.
    Utilizza un partizionamento probabilistico per riflettere le diverse dimensioni dei nodi.
    """

    def __init__(self, output_dir='data/federated'):
        self.output_dir = output_dir
        self.nodes_config = {
            'Ospedale_Roma': 0.35,    # 35% dei dati
            'Ospedale_Milano': 0.20,  # 20% dei dati
            'Ospedale_Napoli': 0.20,  # 20% dei dati
            'Ospedale_Firenze': 0.15, # 15% dei dati
            'Ospedale_Rimini': 0.10   # 10% dei dati
        }

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def split_proportionally(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Divide il dataframe in base alle proporzioni definite in nodes_config. 
        """
        logger.info(f"Avvio partizionamento su {len(df)} record totali...")

        # Mischiamo i dati per garantire una distribuzione casuale tra gli ospedali
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

        hospital_datasets = {}
        start_idx = 0

        for node_name, proportion in self.nodes_config.items():
            # Calcolo numero di campioni per questo ospedale
            num_samples = int(len(df_shuffled) * proportion)
            end_idx = start_idx + num_samples

            # Estrazione del subset
            node_df = df_shuffled.iloc[start_idx:end_idx].copy()
            hospital_datasets[node_name] = node_df

            start_idx = end_idx
            logger.info(f"Assegnati {len(node_df)} record a {node_name}")
        
        return hospital_datasets
    
    def save_hospital_datasets(self, hospital_datasets: Dict[str, pd.DataFrame]):
        """Salva ogni dataset ospedaliero in un file CSV separato."""
        for hospital, df in hospital_datasets.items():
            filename = f"{hospital}_training_data.csv"
            filepath = os.path.join(self.output_dir, filename)
            df.to_csv(filepath, index=False)
            logger.debug(f"Salvato {hospital} in {filepath}")
        logger.info(f"Tutti i {len(hospital_datasets)} file CSV sono stati salvati in {self.output_dir}")

    def get_hospital_statistics(self, hospital_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Genera la tabella riassuntiva della distribuzione.
        """
        stats= []
        total_records = sum(len(df) for df in hospital_datasets.values())

        for hospital, df in hospital_datasets.items():
            count = len(df)
            percentuale = (count / total_records) * 100
            # Calcolo prevalenza patologia (target_label = 1)
            positive_cases = df['target_label'].sum() if 'target_label' in df.columns else 0
            prevalenza = (positive_cases / count) * 100 if count > 0 else 0

            stats.append({
                'Nodo (Ospedale)': hospital,
                'Record Totali': count,
                '% sul Totale': f"{percentuale:.1f}%",
                'Casi Positivi': int(positive_cases),
                'Prevalenza Malattia': f"{prevalenza:.1f}%" 
            })
        
        return pd.DataFrame(stats)

def main():
    """
    Esecuzione stand-alone per generare i dati e mostrarli in tabella.
    """
    print("\n" + "="*50)
    print("🏥 CARDIAC FEDERATED DATA SPLITTER")
    print("="*50)

    splitter = CardiacFederatedSplitter()

    # Percorso del dataset generato dallo script 01_generate_dataset.py
    training_file = 'data/processed/Enhanced_Training_Dataset.csv'

    if os.path.exists(training_file):
        df_train = pd.read_csv(training_file)

        # 1. Divisione dei dati
        hospital_datasets = splitter.split_proportionally(df_train)

        # 2. Salvataggio CSV
        splitter.save_hospital_datasets(hospital_datasets)

        # 3. Generazione e stampa tabella
        stats_df = splitter.get_hospital_statistics(hospital_datasets)

        print("\n📊 DISTRIBUZIONE RECORD PER NODO:")
        print("-" * 75)
        print(stats_df.to_string(index=False))
        print("-" * 75)
        print(f"Dataset totale processato: {len(df_train)} record\n")
    else:
        logger.error(f"File {training_file} non trovato. Esegui prima lo script 01_generate_dataset.py")

if __name__ == "__main__":
    main()
