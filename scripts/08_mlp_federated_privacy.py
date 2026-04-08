#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 08_mlp_federated_privacy.py
Descrizione: Aggregazione Federata dei modelli MLP con protocolli di Privacy.
             Implementa Secure Aggregation (Shamir's Secret Sharing) e 
             Differential Privacy (RDP) sui pesi aggregati.
Output: 
 - Modello globale protetto in 'results/federated_privacy/global_model_privacy.joblib'
 - Report di privacy in 'results/federated_privacy/privacy_report.json'
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np
import json
from sklearn.neural_network import MLPClassifier

# Aggiunta root per importare i moduli dalla cartella utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.enhanced_shamir_privacy import (
    ShamirConfig, 
    DifferentialPrivacyConfig, 
    SecureAggregationProtocol
)

# === Configurazione Percorsi ===
INPUT_MODELS = "results/federated/local_mlp_models.joblib"
RESULTS_DIR = "results/federated_privacy"

def main():
    print("🛡️ Fase 08: Aggregazione Sicura e Privacy Differenziale (DP + SSS)...")
    
    if not os.path.exists(INPUT_MODELS):
        print(f"❌ Errore: Modelli locali non trovati in {INPUT_MODELS}.")
        print("💡 Esegui prima lo script '07_mlp_federated_training.py'.")
        return
    
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Caricamento dei modelli locali (prodotti dallo script 07)
    local_data_list = joblib.load(INPUT_MODELS)
    num_nodes = len(local_data_list)
    total_samples = sum(item['size'] for item in local_data_list)

    # 2. Configurazione Protocolli di Privacy (da utils)
    # Threshold=3 significa che servono almeno 3 ospedali onesti per ricostruire il segreto
    shamir_cfg = ShamirConfig(threshold=3, num_participants=num_nodes)
    dp_cfg = DifferentialPrivacyConfig(epsilon_total=1.0, delta=1e-5)
    
    protocol = SecureAggregationProtocol(shamir_cfg, dp_cfg)

    # 3. Estrazione e Pesatura dei Parametri (FedAvg)
    # Estraiamo i coefficienti (coefs_) e le intercettazioni (intercepts_) di ogni MLP
    weighted_params_list = []
    
    print(f"🔐 Preparazione pesi per {num_nodes} nodi (Record totali: {total_samples})...")

    for item in local_data_list:
        model = item['model']
        weight = item['size'] / total_samples  # Peso proporzionale alla dimensione del dataset
        
        # Concateniamo tutti i pesi della rete in un unico vettore piatto per il protocollo SSS
        params = model.coefs_ + model.intercepts_
        weighted_params = [p * weight for p in params]
        weighted_params_list.append(weighted_params)

    # 4. Esecuzione Aggregazione Sicura
    # Questa funzione applica Shamir, maschera i dati e aggiunge rumore DP calibrato
    print("⚖️ Esecuzione Secure Aggregation e calibrazione rumore DP...")
    global_params = protocol.aggregate(weighted_params_list)

    # 5. Ricostruzione del Modello MLP Globale
    # Usiamo un modello template con la stessa architettura (64, 32)
    global_model = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42)
    
    # Inizializzazione fittizia per creare la struttura dei pesi (13 feature UCI)
    global_model.fit(np.zeros((1, 13)), np.array([0]))
    
    # Iniezione dei parametri aggregati nel modello globale
    num_layers = len(global_model.coefs_)
    global_model.coefs_ = global_params[:num_layers]
    global_model.intercepts_ = global_params[num_layers:]

    # 6. Salvataggio Modello Finale e Report
    model_path = os.path.join(RESULTS_DIR, "global_model_privacy.joblib")
    joblib.dump(global_model, model_path)
    
    report = {
        "status": "SUCCESS",
        "nodes": num_nodes,
        "total_records": total_samples,
        "privacy_config": {
            "epsilon": dp_cfg.epsilon_total,
            "shamir_threshold": shamir_cfg.threshold
        },
        "output_model": model_path
    }
    
    with open(os.path.join(RESULTS_DIR, "privacy_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    print("-" * 60)
    print(f"✅ Aggregazione completata con successo.")
    print(f"💾 Modello globale salvato in: {model_path}")
    print(f"📑 Report di privacy disponibile in: {RESULTS_DIR}/privacy_report.json")

if __name__ == "__main__":
    main()