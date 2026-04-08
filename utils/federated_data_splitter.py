# utils/federated_data_splitter.py
import pandas as pd
import numpy as np
import os

class CardiacFederatedSplitter:
    def __init__(self, output_dir='data/federated'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Nodi e relative quote percentuali definite nella tesi
        self.nodes_config = {
            'Ospedale_Roma': 0.35,
            'Ospedale_Milano': 0.20,
            'Ospedale_Napoli': 0.20,
            'Ospedale_Firenze': 0.15,
            'Ospedale_Rimini': 0.10
        }

    def split_data(self, df):
        # Mescolamento dati per garantire casualità prima dello split
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        total_rows = len(df_shuffled)
        
        start_idx = 0
        for node, percentage in self.nodes_config.items():
            # Calcolo del numero di righe per questo specifico nodo
            num_rows = int(total_rows * percentage)
            end_idx = start_idx + num_rows
            
            # L'ultimo nodo prende tutto il rimanente per evitare arrotondamenti mancanti
            if node == 'Ospedale_Rimini':
                node_df = df_shuffled.iloc[start_idx:]
            else:
                node_df = df_shuffled.iloc[start_idx:end_idx]
            
            output_path = os.path.join(self.output_dir, f"{node}_training_data.csv")
            node_df.to_csv(output_path, index=False)
            
            print(f"📦 {node}: assegnati {len(node_df)} record ({percentage*100}%).")
            start_idx = end_idx