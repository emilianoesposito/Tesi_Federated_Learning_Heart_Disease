import sys
import os
import joblib
import pandas as pd
import numpy as np
import json
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

# SILENZIAMO I WARNING (Corretto in filterwarnings)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Root per utility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.enhanced_shamir_privacy import SecureAggregationProtocol, ShamirConfig, DifferentialPrivacyConfig

# Percorsi
INPUT_MODELS = "results/federated/local_mlp_models.joblib"
TEST_SET_PATH = "data/test/global_test_set.csv"
SCALER_PATH = "results/global_scaler.joblib"
RESULTS_DIR = "results/federated_privacy"

def extract_params(item):
    """Estrae i pesi indipendentemente dal formato di salvataggio."""
    if hasattr(item, 'coefs_'):
        return list(item.coefs_) + list(item.intercepts_)
    if isinstance(item, dict):
        if 'model' in item and hasattr(item['model'], 'coefs_'):
            return list(item['model'].coefs_) + list(item['model'].intercepts_)
        if 'weights' in item:
            return item['weights']
    return None

def main():
    print("🛡️ Fase 08: Analisi Trade-off Privacy (DP + SSS)")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(INPUT_MODELS):
        print(f"❌ Errore: Manca {INPUT_MODELS}"); return

    # 1. Caricamento e Estrazione
    local_data = joblib.load(INPUT_MODELS)
    cleaned_weights = []
    
    items = local_data.values() if isinstance(local_data, dict) else local_data
    for item in items:
        p = extract_params(item)
        if p: cleaned_weights.append([np.array(layer) for layer in p])

    num_nodes = len(cleaned_weights)
    if num_nodes == 0:
        print("❌ Nessun parametro trovato."); return
    
    print(f"✅ Estratti parametri per {num_nodes} partecipanti.")

    # 2. Caricamento Dati Test
    df_test = pd.read_csv(TEST_SET_PATH)
    scaler = joblib.load(SCALER_PATH)
    X_test = scaler.transform(df_test.drop(columns=['target_label']))
    y_test = df_test['target_label']

    epsilons = [None, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = []

    # 3. Ciclo di Aggregazione e Test
    for eps in epsilons:
        eps_label = eps if eps is not None else "No_Privacy"
        print(f"--- Testando ε: {eps_label} ---")

        actual_eps = eps if eps is not None else 1e6
        protocol = SecureAggregationProtocol(
            shamir_cfg=ShamirConfig(num_participants=num_nodes),
            dp_cfg=DifferentialPrivacyConfig(epsilon_total=actual_eps)
        )
        
        try:
            # AGGREGAZIONE + CORREZIONE MEDIA (FedAvg reale)
            raw_sum_params = protocol.aggregate(cleaned_weights)
            global_params = [layer / num_nodes for layer in raw_sum_params]
            
            # Ricostruzione Modello
            model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1)
            model.fit(X_test[:10], y_test[:10]) 

            # Iniezione Pesi
            n_coefs = len(model.coefs_)
            for i in range(n_coefs):
                model.coefs_[i] = global_params[i]
            for i in range(len(model.intercepts_)):
                model.intercepts_[i] = global_params[n_coefs + i].ravel()

            # Valutazione
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            print(f"   📊 Risultato -> Acc: {acc:.4f} | F1: {f1:.4f}")
            results.append({"epsilon": str(eps_label), "accuracy": float(acc), "f1_score": float(f1)})
            
        except Exception as e:
            print(f"   ⚠️ Errore: {e}")

    # 4. Salvataggio
    with open(os.path.join(RESULTS_DIR, "federated_privacy_comparison.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Analisi completata. Risultati in {RESULTS_DIR}")

if __name__ == "__main__":
    main()