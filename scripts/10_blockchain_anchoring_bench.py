#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 10_blockchain_anchoring_bench.py
Descrizione: Benchmark di scalabilità della notarizzazione Blockchain.
             Usa le utility per calcolare i tempi di hashing e l'uso di risorse.
Output: Plot PNG in 'results/blockchain/scalability_analysis.png'
"""

import time
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Aggiunta root progetto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.blockchain_data_anchoring import ClinicalBlockchainAnchor
from utils.parallel_training import SystemResourceMonitor

# === Configurazione ===
OUTPUT_DIR = "results/blockchain"
RECORD_SCALES = [100, 303, 1000, 5000, 10000] # 303 è il dataset reale UCI

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    anchor = ClinicalBlockchainAnchor()
    monitor = SystemResourceMonitor()
    
    results = []
    print("⛓️ Avvio Benchmark Notarizzazione Blockchain...")
    print("-" * 50)

    monitor.start()
    for count in RECORD_SCALES:
        start_t = time.time()
        
        # Simulazione creazione hash per N record
        hashes = []
        for i in range(count):
            dummy_record = {"id": i, "pressione": 120, "target": 1}
            hashes.append(anchor.create_record_hash(dummy_record))
        
        # Creazione Merkle Root finale
        root = anchor.build_merkle_root(hashes)
        
        elapsed = time.time() - start_t
        results.append({'Records': count, 'Time_Sec': elapsed})
        print(f"⏱️ Records: {count:>5} | Tempo: {elapsed:.4f}s | Root: {root[:8]}...")

    resource_stats = monitor.stop()

    # --- Generazione Grafico per la Tesi ---
    df_bench = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.plot(df_bench['Records'], df_bench['Time_Sec'], marker='o', color='#2c3e50', linewidth=2)
    
    # Evidenziamo il punto del dataset reale (303 record)
    uci_time = df_bench.iloc[1]['Time_Sec']
    plt.scatter(303, uci_time, color='red', s=100, zorder=5, label='Dataset UCI Reale (303)')
    
    plt.title("Performance Notarizzazione (Hashing + Merkle Root)", fontsize=14)
    plt.xlabel("Numero di Record Clinici")
    plt.ylabel("Tempo di Elaborazione (Secondi)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = os.path.join(OUTPUT_DIR, "scalability_analysis.png")
    plt.savefig(plot_path, dpi=300)

    print("-" * 50)
    print(f"✅ Benchmark completato. Grafico salvato in: {plot_path}")
    if not resource_stats.empty:
        print(f"📉 Consumo CPU medio: {resource_stats['cpu'].mean():.1f}%")

if __name__ == "__main__":
    main()