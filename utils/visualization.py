# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import os

def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.dpi'] = 300

def visualize_distribution(df, save_dir='results/visualizations'):
    setup_style()
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    
    # Trova automaticamente la colonna target
    t_col = 'target_label' if 'target_label' in df.columns else ('outcome' if 'outcome' in df.columns else df.columns[-1])
    
    ax = sns.countplot(data=df, x=t_col, palette=['#3498db', '#e74c3c'])
    plt.title("Distribuzione Classi: Sani (0) vs Malati (1)")
    plt.savefig(os.path.join(save_dir, "distribution.png"))
    plt.close()

def visualize_correlations(df, save_dir='results/visualizations'):
    setup_style()
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Matrice di Correlazione Variabili Cliniche")
    plt.savefig(os.path.join(save_dir, "correlations.png"))
    plt.close()

def visualize_metrics_comparison(df, save_dir, output_name):
    setup_style()
    # Supporta sia 'model' che 'Scenario' come richiesto dagli script 04 e 09
    id_col = 'model' if 'model' in df.columns else 'Scenario'
    df_melted = df.melt(id_vars=id_col, var_name='Metrica', value_name='Score')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x=id_col, y='Score', hue='Metrica', palette='viridis')
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(save_dir, output_name))
    plt.close()

def visualize_federated_distribution(df, save_dir, output_name):
    setup_style()
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Hospital', hue='target_label', palette=['#3498db', '#e74c3c'])
    plt.title("Distribuzione Non-IID tra Nodi Federati")
    plt.savefig(os.path.join(save_dir, output_name))
    plt.close()