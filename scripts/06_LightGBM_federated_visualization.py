#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd

# Fix per i percorsi (Windows/Linux)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.visualization import visualize_federated_distribution

def main():
    DATA_DIR = "data/federated"
    OUTPUT_DIR = "results/federated"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    hospitals = ['Ospedale_Roma', 'Ospedale_Milano', 'Ospedale_Napoli', 'Ospedale_Firenze', 'Ospedale_Rimini']
    combined_data = []

    print("📊 Analisi Distribuzione Non-IID tra i nodi (Training Set)...")
    
    for h in hospitals:
        file_path = os.path.join(DATA_DIR, f"{h}_training_data.csv")
        if not os.path.exists(file_path):
            print(f"⚠️ Salto {h}: file non trovato.")
            continue
            
        df_local = pd.read_csv(file_path)
        
        # Uniformiamo il nome della colonna target per il grafico
        if 'target_label' not in df_local.columns and 'outcome' in df_local.columns:
            df_local = df_local.rename(columns={'outcome': 'target_label'})
        
        df_local['Hospital'] = h
        combined_data.append(df_local)
        
        # Log di controllo prevalenza
        if 'target_label' in df_local.columns:
            prev = (df_local['target_label'].sum() / len(df_local)) * 100
            print(f"🏥 {h:16} | Record: {len(df_local):>3} | Malati: {prev:.1f}%")

    if combined_data:
        df_all = pd.concat(combined_data, ignore_index=True)
        
        print("\n🎨 Generazione del grafico Non-IID...")
        # CORREZIONE: Aggiunto output_name="federated_non_iid_distribution.png"
        visualize_federated_distribution(
            df_all, 
            save_dir=OUTPUT_DIR, 
            output_name="federated_non_iid_distribution.png"
        )
        print(f"✅ Grafico salvato in: {OUTPUT_DIR}/federated_non_iid_distribution.png")
    else:
        print("❌ Errore: Nessun dato trovato nella cartella data/federated/")

if __name__ == "__main__":
    main()