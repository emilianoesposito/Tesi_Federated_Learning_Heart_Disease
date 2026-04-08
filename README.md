# Privacy-Preserving Cardiac Monitoring
_Architettura Federata con Privacy Differenziale, Shamir's Secret Sharing e Notarizzazione Blockchain_

<p align="center">
  <img src="https://img.shields.io/badge/Status-Project%20Completed-success" />
  <img src="https://img.shields.io/badge/AI-Federated%20Learning-blueviolet" />
  <img src="https://img.shields.io/badge/Privacy-Differential%20Privacy-green" />
  <img src="https://img.shields.io/badge/Security-Shamir%20SSS-orange" />
  <img src="https://img.shields.io/badge/Integrity-Blockchain-blue" />
</p>

## 🚀 Panoramica del Progetto
Questo progetto sviluppa un sistema avanzato di monitoraggio cardiaco che risolve il conflitto tra analisi dei dati (AI) e riservatezza del paziente (GDPR). Invece di centralizzare dati sensibili, il sistema utilizza il **Federated Learning** per addestrare i modelli localmente negli ospedali, proteggendo i pesi con **Privacy Differenziale** e garantendo l'integrità tramite **Blockchain**.

## 🛠️ Architettura Tecnica
Il sistema si basa su tre pilastri fondamentali:
1. **Federated Learning (Non-IID)**: Simulazione di 5 nodi ospedalieri con distribuzioni di dati eterogenee.
2. **Secure Aggregation & DP**: Protocollo basato su *Shamir's Secret Sharing* per l'aggregazione dei pesi e rumore di Laplace (Differential Privacy) per l'anonimizzazione.
3. **Blockchain Anchoring**: Notarizzazione del *Merkle Root* dei record clinici per garantire l'immutabilità dei log diagnostici.

## 📂 Struttura della Pipeline
Eseguire gli script in ordine numerico per replicare l'intero esperimento:

1. `01_generate_dataset.py`: Preparazione dati e split federato tra i nodi.
2. `02_visualize_dataset.py`: Analisi esplorativa dei dati (EDA).
3. `03_train_models.py`: Training della Baseline centralizzata (LightGBM vs MLP).
4. `05_LightGBM_federated_training.py`: Addestramento federato tramite LightGBM.
5. `06_LightGBM_federated_visualization.py`: Visualizzazione della distribuzione Non-IID tra gli ospedali.
6. `07_mlp_federated_training.py`: Preparazione dei modelli MLP per l'aggregazione sicura.
7. `08_mlp_federated_privacy.py`: **[CORE]** Aggregazione sicura con Shamir e Privacy Differenziale.
8. `09_mlp_federated_privacy_visualization.py`: Analisi del trade-off tra Utilità e Privacy.
9. `10_blockchain_anchoring_bench.py`: Benchmark di scalabilità della notarizzazione Blockchain.

## 📈 Risultati Ottenuti
I test eseguiti sul dataset UCI Heart Disease hanno prodotto i seguenti risultati:

| Scenario | Modello | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| **Centralizzato** | MLP (Baseline) | **83.6%** | **84.4%** |
| **Federato** | MLP (No Privacy) | 81.6% | 81.4% |
| **Sicuro** | MLP (DP + Shamir) | 77.6% | 76.4% |

**Nota sulla Privacy:** Il calo di circa il 6% nell'accuratezza è il "costo della privacy" (Epsilon=1.0), necessario per garantire che nessun dato individuale possa essere estratto dal modello globale.

## 🔗 Performance Blockchain
Il sistema di notarizzazione ha dimostrato un'altissima efficienza:
- **Tempo di notarizzazione (303 record)**: < 0.005s.
- **Scalabilità**: < 0.1s per 10.000 record clinici.
- **Impatto CPU**: ~0.0% (operazione estremamente leggera).

## ⚙️ Setup e Installazione
```bash
# Clone del repository
git clone [URL_DEL_TUO_REPO]

# Creazione ambiente virtuale
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installazione dipendenze
pip install -r requirements.txt