# utils/federated_data_splitter.py
import pandas as pd
import numpy as np
import os

class CardiacFederatedSplitter:
    def __init__(self, output_dir='data/federated'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.nodes = ['Ospedale_Roma', 'Ospedale_Milano', 'Ospedale_Napoli', 'Ospedale_Firenze', 'Ospedale_Rimini']

    def split_data(self, df):
        # Mescolamento dati
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        # Divisione in 5 parti
        chunks = np.array_split(df_shuffled, len(self.nodes))
        
        for i, node in enumerate(self.nodes):
            output_path = os.path.join(self.output_dir, f"{node}_training_data.csv")
            # Garantiamo che sia un DataFrame e salviamo
            pd.DataFrame(chunks[i]).to_csv(output_path, index=False)