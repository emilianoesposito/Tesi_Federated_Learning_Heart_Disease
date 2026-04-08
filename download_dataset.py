import pandas as pd
import os

def download_and_prepare_uci_data():
    # 1. Definizione URL e percorsi
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    raw_dir = 'data/raw'
    output_file = os.path.join(raw_dir, 'Dataset_Pazienti.csv')

    # Crea la cartella se non esiste
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
        print(f"📁 Creata cartella: {raw_dir}")

    # 2. Nomi delle colonne originali UCI
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    print("⏳ Scaricamento dataset Heart Disease UCI (Cleveland)...")
    
    try:
        # Caricamento dati (il dataset UCI usa '?' per i valori mancanti)
        df = pd.read_csv(url, names=columns, na_values='?')
        
        # Gestione valori mancanti (fondamentale per i 303 record reali)
        # Riempiamo i pochi NA con la mediana per non perdere record preziosi
        df = df.fillna(df.median())

        # 3. Rinominazione colonne per il sistema (Analisi e Adattamento - Punto 1)
        # Mappiamo i nomi originali verso quelli usati nei tuoi script 01-04
        mapping = {
            'age': 'eta',
            'trestbps': 'pressione',
            'chol': 'colesterolo',
            'target': 'outcome'
        }
        df = df.rename(columns=mapping)

        # 4. Trasformazione Target Binario (come richiesto dal Prof)
        # UCI: 0 = sano, 1-4 = gradi di malattia. Noi vogliamo 0 vs 1.
        df['outcome'] = (df['outcome'] > 0).astype(int)

        # 5. Salvataggio
        df.to_csv(output_file, index=False)
        
        print("-" * 30)
        print(f"✅ Download completato con successo!")
        print(f"📊 Record totali: {len(df)}")
        print(f"📍 File salvato in: {output_file}")
        print(f"💡 Target bilanciato (0=sani, 1=malati): \n{df['outcome'].value_counts()}")
        print("-" * 30)

    except Exception as e:
        print(f"❌ Errore durante il download: {e}")

if __name__ == "__main__":
    download_and_prepare_uci_data()