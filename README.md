# Privacy-Preserving Cardiac Monitoring
### _Architettura Federata con Privacy Differenziale, Shamir's Secret Sharing e Notarizzazione Blockchain_

<p align="center">
  <img src="https://img.shields.io/badge/Status-Project%20Validated-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Framework-Scikit--Learn-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Privacy-Differential%20Privacy-green?style=for-the-badge" />
</p>

---

## 🚀 Panoramica del Progetto
Questo progetto implementa un'architettura avanzata per il monitoraggio cardiaco, progettata per rispettare le normative GDPR pur mantenendo elevate capacità predittive. Il sistema risolve il problema della centralizzazione dei dati sensibili utilizzando il **Federated Learning**, dove il modello "viaggia" verso i dati e non viceversa.

### I Tre Pilastri della Sicurezza:
1.  **Federated Learning (Non-IID):** Addestramento distribuito su 5 nodi ospedalieri con distribuzioni di dati eterogenee.
2.  **Differential Privacy (DP) & Shamir SSS:** Protezione dei gradienti tramite rumore statistico e scomposizione dei segreti per impedire attacchi di inversione del modello.
3.  **Blockchain Notarization:** Garanzia di immutabilità dei log clinici e dei risultati tramite Merkle Tree e hashing SHA-256.

---

## 🛠️ Novità e Correzioni (Revisione Finale)
* **Metric Correction:** Risolto il bug di iniezione pesi nello Script 08. Ora il modello globale viene ricostruito rispettando il broadcasting dei bias, stabilizzando l'**F1-Score** a valori reali (~0.81).
* **Scientific Visualization:** Implementata la **scala logaritmica** per l'analisi del budget di privacy ($\epsilon$), permettendo una valutazione precisa del trade-off Utility/Privacy.
* **Methodological Rigor:** Introduzione di un **Test Set Globale (20%)** isolato all'origine (Script 01) per garantire che le metriche finali non siano affette da bias locali.

---

## 📂 Struttura della Pipeline
Gli script sono numerati per riflettere l'ordine logico del workflow scientifico:

1.  `01_generate_dataset.py`: Pre-processing, scoring clinico e creazione del Test Set Globale.
2.  `02_visualize_dataset.py`: Analisi esplorativa dei dati (EDA) e correlazioni.
3.  `03_train_models.py`: Creazione della **Baseline Centralizzata** (MLP e LightGBM).
4.  `04_analyze_results.py`: Generazione grafici comparativi della baseline.
5.  `05_LightGBM_federated_training.py`: Simulazione del training federato locale.
6.  `06_LightGBM_federated_visualization.py`: Analisi visiva della distribuzione Non-IID tra i nodi.
7.  `07_mlp_federated_training.py`: Training locale dei modelli MLP per l'aggregazione.
8.  **`08_mlp_federated_privacy.py`**: [CORE] Aggregazione sicura con SSS e iniezione DP.
9.  `09_mlp_federated_privacy_visualization.py`: Plotting del trade-off in scala logaritmica.
10. `10_blockchain_anchoring_bench.py`: Benchmark di scalabilità della notarizzazione.

---

## 📈 Risultati di Validazione
I test aggiornati dimostrano che l'architettura mantiene un'ottima utilità clinica anche sotto stringenti vincoli di privacy:

| Scenario | Modello | Accuracy | F1-Score | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Centralizzato** | MLP Baseline | 83.6% | 0.84 | Limite superiore teorico |
| **Federato** | MLP (No Privacy) | 77.0% | 0.80 | Perdita dovuta a Non-IID |
| **Protetto** | **MLP (DP $\epsilon=1.0$)** | **78.7%** | **0.81** | **Configurazione Ottimale** |

> **Analisi:** Il "costo della privacy" è minimo rispetto ai benefici di sicurezza ottenuti, con un F1-score che si mantiene stabilmente sopra lo 0.80.

---

## 💻 Installazione e Utilizzo

### 1. Requisiti
* Python 3.9 o superiore
* Pip (Python package manager)

### 2. Setup Ambiente
```bash
# Clonare il repository
git clone <tuo-url-repository>
cd privacy-preserving-cardiac-monitoring

# Creare un ambiente virtuale
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installare le dipendenze
pip install -r requirements.txt