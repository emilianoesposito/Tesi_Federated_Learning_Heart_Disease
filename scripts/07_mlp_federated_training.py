#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 07_mlp_federated_training.py
Descrizione: Addestramento locale dei nodi ospedalieri tramite Multi-Layer Perceptron (MLP).
             Prepara i pesi dei modelli per l'aggregazione sicura (Shamir/FedAvg).
Output: 
 - Risultati locali (modelli e pesi) in 'results/federated/local_mlp_models.joblib'
"""

import os, sys, joblib
from sklearn.neural_network import MLPClassifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.federated_learning import CardiacFederatedManager

def main():
    scaler = joblib.load("results/global_scaler.joblib")
    manager = CardiacFederatedManager()
    hospitals = ['Ospedale_Roma', 'Ospedale_Milano', 'Ospedale_Napoli', 'Ospedale_Firenze', 'Ospedale_Rimini']

    for h in hospitals:
        path = f"data/federated/{h}_training_data.csv"
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        manager.train_node(h, path, model, scaler=scaler)

    manager.save_local_models("results/federated/local_mlp_models.joblib")
    print("✅ Pesi MLP locali pronti per l'aggregazione.")

if __name__ == "__main__":
    main()