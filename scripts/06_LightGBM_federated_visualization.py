#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 06_LightGBM_federated_visualization.py
Descrizione: Visualizzazione della distribuzione eterogenea (Non-IID) tra i nodi.
             Dimostra graficamente la variazione della prevalenza della malattia 
             tra i 5 ospedali simulati utilizzando le utility di visualizzazione.
Output: Plot PNG in 'results/federated/federated_non_iid_distribution.png'
"""

import os
import sys
import pandas as pd

# Fix per Windows
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

    print("📊 Analisi della distribuzione eterogenea (Non-IID) tra i nodi...")
    print("-" * 60)

    for h in hospitals:
        file_path = os.path.join(DATA_DIR, f"{h}_training_data.csv")
        if not os.path.exists(file_path):
            print(f"⚠️ Salto {h}: file non trovato.")
            continue
            
        df_local = pd.read_csv(file_path)
        
        # Gestione flessibile del nome colonna target
        target_col = 'target_label' if 'target_label' in df_local.columns else 'outcome'
        
        if target_col not in df_local.columns:
            print(f"❌ Errore: Colonna target non trovata in {h}. Colonne: {df_local.columns.tolist()}")
            continue

        # Rinominiamo internamente per uniformità nel grafico
        df_local = df_local.rename(columns={target_col: 'target_label'})
        df_local['Hospital'] = h
        combined_data.append(df_local)
        
        # Calcolo prevalenza per log
        prevalence = (df_local['target_label'].sum() / len(df_local)) * 100
        print(f"🏥 {h:16} | Record: {len(df_local):>3} | Prevalenza Malati: {prevalence:.1f}%")

    if combined_data:
        df_all = pd.concat(combined_data, ignore_index=True)
        visualize_federated_distribution(df_all, save_dir=OUTPUT_DIR, output_name="federated_non_iid_distribution.png")
        print("\n✅ Grafico 'federated_non_iid_distribution.png' generato con successo!")
    else:
        print("⚠️ Nessun dato trovato per la visualizzazione.")

if __name__ == "__main__":
    main()