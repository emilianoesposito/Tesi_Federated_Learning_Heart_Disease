import os
import sys
import pandas as pd
import json
import matplotlib.pyplot as plt

def main():
    RESULTS_DIR = "results/federated_privacy"
    INPUT_PATH = os.path.join(RESULTS_DIR, "federated_privacy_comparison.json")
    OUTPUT_DIR = "results/visualizations"
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Errore: Manca il file {INPUT_PATH}. Esegui lo script 08!"); return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)

    # 1. Separazione Baseline (No_Privacy) e dati DP
    df['eps_num'] = pd.to_numeric(df['epsilon'], errors='coerce')
    df_dp = df.dropna(subset=['eps_num']).sort_values('eps_num')
    
    # Recupero valore No_Privacy per la linea orizzontale
    no_priv_data = df[df['epsilon'] == 'No_Privacy']
    
    # 2. Creazione Grafico
    plt.figure(figsize=(10, 6))
    
    # Linee per Accuratezza e F1 con Differential Privacy
    plt.plot(df_dp['eps_num'], df_dp['accuracy'], marker='o', label='Accuracy (DP)', linewidth=2, color='#1f77b4')
    plt.plot(df_dp['eps_num'], df_dp['f1_score'], marker='s', label='F1-Score (DP)', linewidth=2, linestyle='--', color='#ff7f0e')

    # Linea di riferimento Baseline (No Privacy)
    if not no_priv_data.empty:
        plt.axhline(y=no_priv_data['accuracy'].values[0], color='green', linestyle=':', alpha=0.6, label='Baseline Accuracy (No Privacy)')
        plt.axhline(y=no_priv_data['f1_score'].values[0], color='red', linestyle=':', alpha=0.6, label='Baseline F1 (No Privacy)')

    # Formattazione Assi
    plt.xscale('log')
    # Forza le etichette dell'asse X a mostrare i nostri epsilon esatti
    tick_values = df_dp['eps_num'].tolist()
    plt.xticks(tick_values, labels=[str(v) for v in tick_values])
    
    plt.xlabel('Privacy Budget (ε) - Scala Logaritmica')
    plt.ylabel('Score (0.0 - 1.0)')
    plt.title('Trade-off Privacy vs Utility (Federated MLP)')
    plt.legend(loc='lower right')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.ylim(0, 1.0) # Opzionale: fissa l'asse Y tra 0 e 1 per chiarezza tesi

    save_path = os.path.join(OUTPUT_DIR, "federated_privacy_tradeoff.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Grafico salvato con successo in: {save_path}")

if __name__ == "__main__":
    main()