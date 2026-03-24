# Privacy-Preserving Cardiac Monitoring: Federated Learning & Blockchain
_Progetto di Tesi di Laurea in Ingegneria Informatica (CyberSecurity)_

<p align="center">
  <img src="https://img.shields.io/badge/Status-Baseline%20Validated-success" />
  <img src="https://img.shields.io/badge/Dataset-UCI%20Heart%20Disease-blue" />
  <img src="https://img.shields.io/badge/Algorithm-LightGBM%20%7C%20MLP-orange" />
  <img src="https://img.shields.io/badge/Privacy-Federated%20Learning-purple" />
</p>

> **Executive Summary.** Questo progetto sviluppa un'architettura di intelligenza artificiale per il monitoraggio preventivo delle patologie cardiache. Il sistema affronta le sfide della privacy e dell'integrità dei dati sanitari integrando il **Federated Learning**, che permette l'addestramento distribuito senza condivisione di dati grezzi, e la **Blockchain** per la notarizzazione immutabile dei risultati diagnostici.

## 📑 Indice
- [Obiettivi del Progetto](#obiettivi-del-progetto)
- [Struttura del Repository](#struttura-del-repository)
- [Installazione e Setup](#installazione-e-setup)
- [Pipeline di Esecuzione (Baseline)](#pipeline-di-esecuzione-baseline)
- [Risultati della Baseline (Punto 2)](#risultati-della-baseline-punto-2)
- [Tecnologie Utilizzate](#tecnologie-utilizzate)

## 🎯 Obiettivi del Progetto
1.  **Analisi Clinica (Punto 1):** Mapping e binarizzazione del dataset Heart Disease UCI (Cleveland) per lo screening cardiaco preventivo.
2.  **Baseline Centralizzata (Punto 2):** Definizione del "Gold Standard" prestazionale tramite modelli MLP e LightGBM con bilanciamento SMOTE.
3.  **Partizione Federata (Punto 3):** Simulazione di 5 nodi ospedalieri con distribuzioni di dati non-IID (In corso).
4.  **Privacy & Integrità (Punto 4):** Implementazione di aggregazione sicura (Shamir Secret Sharing) e notarizzazione tramite Merkle Tree.

## 📂 Struttura del Repository
* **`data/`**: Contiene il dataset originale e i dati processati.
* **`results/`**: Output generati (metriche, grafici comparativi e modelli salvati).
* **`scripts/`**: Pipeline sperimentale (Preprocessing, Training Baseline, Analisi).
* **`utils/`**: Logica di core (Scoring clinico, Feature Engineering, Visualizzazione).
* **`venv/`**: Ambiente virtuale Python (escluso dal controllo di versione).
* **`download_dataset.py`**: Script per il recupero del dataset Heart Disease UCI.
* **`requirements.txt`**: Elenco delle dipendenze necessarie con versioni verificate.
* **`.gitignore`**: Configurazione per l'esclusione di file temporanei e dati sensibili.

## 🛠️ Installazione e Setup
1. **Clonazione del repository:**
   ```bash
   git clone <url-del-tuo-repo-github>
   cd <nome-cartella-repo>

2. **Attivazione ambiente virtuale:**
   ```bash
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate

3. **Installazione dipendenze:**
   ```bash
   pip install -r requirements.txt

## 🚀 Pipeline di Esecuzione (Baseline)
Per replicare i risultati della Baseline (Punti 1 e 2), eseguire gli script in sequenza dal terminale:
1. Download Dati: python download_dataset.py
2. Preprocessing: python scripts/01_generate_dataset.py
3. Visualizzazione: python scripts/02_visualize_dataset.py
4. Training: python scripts/03_train_models.py (Applica lo SMOTE per il bilanciamento).
5. Analisi: python scripts/04_analyze_results.py (Genera il confronto finale delle metriche).

## 📊 Risultati della Baseline (Punto 2)
Le prestazioni ottenute in modalità centralizzata rappresentano il **Gold Standard** per la successiva fase federata:
Modello,            Accuracy,  F1-Score,  ROC-AUC
LightGBM,           86.89%,    0.8667,    0.9545
MLP (Rete Neurale), 83.61%,    0.7917,    0.9048

**Nota:**Il modello **LightGBM** si conferma il più efficace per gestire dati clinici tabellari di piccole dimensioni. L'architettura **MLP**, pur avendo prestazioni leggermente inferiori, garantisce la compatibilità nativa con l'aggregazione dei pesi necessaria per il Federated Learning.

## 🔬 Tecnologie Utilizzate
- **Machine Learning:** Scikit-Learn (MLP), LightGBM.
- **Data Prep & Analysis:** Pandas, NumPy, Imbalanced-Learn (SMOTE).
- **Visualizzazione:** Matplotlib, Seaborn.
- **Sicurezza e Privacy:** Federated Learning, Shamir Secret Sharing, Blockchain Anchoring (Merkle Tree).


