import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import kagglehub

def get_file_list(directory_path: str) -> List[str]:
    """Retourne la liste des fichiers disponibles dans un dossier (fonction pure)."""
    try:
        return os.listdir(directory_path)
    except Exception as e:
        print(f"Erreur lors de la lecture du dossier : {e}")
        return []

def find_train_file(file_names: List[str]) -> Optional[str]:
    """Recherche le nom du fichier contenant 'train' dans une liste (fonction pure)."""
    for file in file_names:
        if 'train' in file:
            return file
    return None

def load_data(directory_path: str, file_name: Optional[str]) -> Optional[pd.DataFrame]:
    """Charge un fichier CSV dans un DataFrame (fonction pure)."""
    if file_name:
        return pd.read_csv(os.path.join(directory_path, file_name))
    return None

def impute_categorical_with_unknown(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Remplace les valeurs manquantes par 'Unknown' pour une liste de colonnes (fonction pure)."""
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna('Unknown')
    return df_copy

def impute_with_mode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Impute les valeurs manquantes par la valeur la plus fréquente (le mode) (fonction pure)."""
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            mode_val = df_copy[col].mode()[0]
            df_copy[col] = df_copy[col].fillna(mode_val)
    return df_copy

def clean_experience_column(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie la colonne experience en la rendant numérique et en imputant par la médiane (fonction pure)."""
    df_copy = df.copy()
    if 'experience' in df_copy.columns:
        df_copy['experience'] = df_copy['experience'].replace({'>20': '20', '<1': '0'})
        df_copy['experience'] = pd.to_numeric(df_copy['experience'], errors='coerce')
        median_val = df_copy['experience'].median()
        df_copy['experience'] = df_copy['experience'].fillna(median_val)
    return df_copy

def clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Fonction pure principale combinant toutes les étapes de nettoyage."""
    df_cleaned = df_raw.copy()
    
    cols_to_unknown = ['gender', 'company_size', 'company_type', 'major_discipline']
    df_cleaned = impute_categorical_with_unknown(df_cleaned, cols_to_unknown)
    
    df_cleaned = clean_experience_column(df_cleaned)
    
    cols_to_mode = ['enrolled_university', 'education_level', 'last_new_job']
    df_cleaned = impute_with_mode(df_cleaned, cols_to_mode)
    
    return df_cleaned

def plot_target_distribution(df: pd.DataFrame) -> None:
    """Affiche un graphique de la distribution de la variable cible (fonction pure)."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df, palette='viridis', hue='target', legend=False)
    plt.title('Distribution de la Variable Cible (Target)')
    plt.xlabel('Changement de travail (0 = Non, 1 = Oui)')
    plt.ylabel('Nombre de candidats')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_experience_distribution(df: pd.DataFrame) -> None:
    """Affiche un histogramme des années d'expérience (fonction pure)."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df['experience'], kde=True, bins=20, color='skyblue')
    plt.title("Distribution des années d'expérience")
    plt.xlabel("Années d'expérience")
    plt.ylabel('Fréquence')
    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.show()

def plot_job_change_by_education(df: pd.DataFrame) -> None:
    """Affiche le taux de changement d'emploi selon le niveau d'études (fonction pure)."""
    plt.figure(figsize=(8, 5))
    df_grouped = (df.groupby('education_level')['target']
                  .mean()
                  .reset_index()
                  .sort_values(by='target', ascending=False))
    sns.barplot(x='target', y='education_level', data=df_grouped, palette='mako', hue='education_level', legend=False)
    plt.title("Taux de changement d'emploi selon le niveau d'études")
    plt.xlabel("Proportion de candidats cherchant à changer d'emploi")
    plt.ylabel("Niveau d'études")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main() -> None:
    """Fonction principale pour exécuter tout le pipeline de données."""
    print("=== Étape 1 : Chargement des données ===")
    
    kaggle_path = '/kaggle/input/hr-analytics-job-change-of-data-scientists'
    if os.path.exists(kaggle_path):
        default_path = kaggle_path
        print(f"Environnement Kaggle détecté : {default_path}")
    else:
        print("Environnement local détecté, téléchargement via kagglehub...")
        default_path = kagglehub.dataset_download("arashnic/hr-analytics-job-change-of-data-scientists")
        print(f"Chemin des fichiers : {default_path}")
    
    files = get_file_list(default_path)
    train_file_name = find_train_file(files)
    df_raw = load_data(default_path, train_file_name)
    
    if df_raw is not None:
        print(f"Données chargées avec succès depuis : {train_file_name}")
        print(f"Dimensions initiales : {df_raw.shape}")
        
        print("\n=== Étape 2 : Nettoyage des données ===")
        df_cleaned = clean_dataset(df_raw)
        print("Nettoyage terminé.")
        print(f"Nombre de valeurs manquantes restantes : {df_cleaned.isnull().sum().sum()}")
        
        print("\n=== Étape 3 : Affichage des visualisations ===")
        plot_target_distribution(df_cleaned)
        plot_experience_distribution(df_cleaned)
        plot_job_change_by_education(df_cleaned)
        
        print("\nPipeline exécuté avec succès !")
    else:
        print(f"Erreur : Impossible de localiser ou charger les données dans '{default_path}'.")

if __name__ == '__main__':
    main()