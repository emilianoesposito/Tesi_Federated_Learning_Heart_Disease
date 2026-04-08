#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 07_mlp_federated_training.py
Descrizione: Addestramento locale dei nodi ospedalieri tramite Multi-Layer Perceptron (MLP).
             Prepara i pesi dei modelli per l'aggregazione sicura (Shamir/FedAvg).
Output: 
 - Risultati locali (modelli e pesi) in 'results/federated/local_mlp_models.joblib'
"""

import os
import sys
import joblib
from sklearn.neural_network import MLPClassifier

# Fix per i percorsi su Windows
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.federated_learning import CardiacFederatedManager

# === Configurazione Percorsi ===
DATA_DIR = "data/federated"
RESULTS_DIR = "results/federated"
GLOBAL_SCALER_PATH = "results/global_scaler.joblib"

hospitals = [
    'Ospedale_Roma', 
    'Ospedale_Milano', 
    'Ospedale_Napoli', 
    'Ospedale_Firenze', 
    'Ospedale_Rimini'
]

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Caricamento dello scaler globale
    if not os.path.exists(GLOBAL_SCALER_PATH):
        print(f"❌ Errore: Scaler non trovato in {GLOBAL_SCALER_PATH}.")
        print("💡 Esegui prima lo script '03_train_models.py'.")
        return

    scaler = joblib.load(GLOBAL_SCALER_PATH)

    # 2. Inizializzazione del Manager Federato (Senza argomenti extra)
    manager = CardiacFederatedManager()
    
    print("🧠 Inizio Training Federato MLP (Preparazione Pesi per Privacy)...")
    print("-" * 65)

    for hospital in hospitals:
        file_path = os.path.join(DATA_DIR, f"{hospital}_training_data.csv")
        
        if not os.path.exists(file_path):
            print(f"⚠️ Salto {hospital}: file non trovato.")
            continue

        # Modello MLP locale con configurazione tesi
        model_template = MLPClassifier(
            hidden_layer_sizes=(64, 32), 
            max_iter=1000, 
            random_state=42,
            alpha=0.05,
            learning_rate_init=0.01
        )

        # Addestramento tramite il manager
        print(f"🏥 Addestramento {hospital:16}...", end=" ", flush=True)
        manager.train_node(hospital, file_path, model_template, scaler=scaler)
        print("✅ Pesi estratti.")

    # 3. Salvataggio dei risultati
    output_path = os.path.join(RESULTS_DIR, "local_mlp_models.joblib")
    manager.save_results(output_path)
    
    print("-" * 65)
    print(f"🏁 Step 07 completato. {len(manager.local_results)} modelli MLP pronti per Shamir.")
    print(f"📂 File generato: {output_path}")

if __name__ == "__main__":
    main()