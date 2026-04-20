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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.blockchain_data_anchoring import ClinicalBlockchainAnchor

OUTPUT_DIR = "results/blockchain"
RECORD_SCALES = [100, 303, 1000, 5000, 10000]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    anchor = ClinicalBlockchainAnchor()
    results = []
    
    print("⛓️ Avvio Benchmark Notarizzazione Blockchain (Integrità Dati)...")

    for count in RECORD_SCALES:
        start_t = time.time()
        # Simulazione hashing
        hashes = [anchor.create_record_hash({"id": i, "val": 0.5}) for i in range(count)]
        root = anchor.build_merkle_root(hashes)
        elapsed = time.time() - start_t
        results.append({'Records': count, 'Time_Sec': elapsed})
        print(f"⏱️ Records: {count:>5} | Tempo: {elapsed:.4f}s")

    # Grafico di scalabilità
    df_bench = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.plot(df_bench['Records'], df_bench['Time_Sec'], marker='o', color='#2c3e50')
    plt.axvline(x=303, color='red', linestyle='--', label='Dataset UCI (303)')
    plt.title("Analisi Scalabilità Notarizzazione Blockchain")
    plt.xlabel("Numero di Record"), plt.ylabel("Secondi"), plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "scalability_analysis.png"))
    print("✅ Benchmark completato.")

if __name__ == "__main__":
    main()