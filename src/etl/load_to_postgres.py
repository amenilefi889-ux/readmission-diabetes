"""
Pipeline ETL — Diabetes 130-US Hospitals
==========================================
Extrait, nettoie et charge le dataset diabetic_data.csv vers PostgreSQL,
en reproduisant exactement les étapes de nettoyage validées dans le
notebook 01_exploration.ipynb (Phase 1 - EDA).

Usage:
    python src/etl/load_to_postgres.py
"""

import pandas as pd
from sqlalchemy import create_engine
import os

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------

RAW_DATA_PATH = "data/raw/diabetic_data.csv"

DB_USER = "diabetes_user"
DB_PASSWORD = "diabetes_pass"
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "diabetes_readmission"

DB_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Codes discharge_disposition_id correspondant à un décès ou une sortie en hospice
DECES_HOSPICE_CODES = [11, 13, 14, 19, 20, 21]

# Colonnes finales à garder pour la table patient_encounters
COLUMNS_TO_KEEP = [
    "encounter_id", "patient_nbr", "race", "gender", "age",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "medical_specialty",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult",
    "diabetesMed", "change",
    "readmitted", "target",
]


# ----------------------------------------------------------------------
# 2. EXTRACT
# ----------------------------------------------------------------------

def extract(path: str) -> pd.DataFrame:
    """Charge le CSV brut."""
    print(f"[EXTRACT] Lecture de {path} ...")
    df = pd.read_csv(path, na_values="?", low_memory=False)
    print(f"[EXTRACT] {len(df)} lignes chargées.")
    return df


# ----------------------------------------------------------------------
# 3. TRANSFORM (reprend exactement les étapes du notebook)
# ----------------------------------------------------------------------

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage complet : valeurs manquantes, dédup, exclusions, cible."""

    # --- Variable cible ---
    df["target"] = (df["readmitted"] == "<30").astype(int)

    # --- Suppression des colonnes trop incomplètes / peu pertinentes ---
    df = df.drop(columns=["weight", "payer_code"])

    # --- Imputation par catégorie explicite ---
    df["max_glu_serum"] = df["max_glu_serum"].fillna("Not tested")
    df["A1Cresult"] = df["A1Cresult"].fillna("Not tested")
    df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
    df["race"] = df["race"].fillna("Unknown")
    df["diag_1"] = df["diag_1"].fillna("Unknown")
    df["diag_2"] = df["diag_2"].fillna("Unknown")
    df["diag_3"] = df["diag_3"].fillna("Unknown")

    n_before = len(df)
    assert df.isnull().sum().sum() == 0, "Il reste des valeurs manquantes !"
    print(f"[TRANSFORM] Valeurs manquantes traitées ({n_before} lignes).")

    # --- Déduplication : 1 patient = 1er séjour ---
    df = df.sort_values("encounter_id").drop_duplicates(
        subset="patient_nbr", keep="first"
    )
    print(f"[TRANSFORM] Déduplication patients : {len(df)} lignes restantes.")

    # --- Exclusion décès / hospice ---
    avant = len(df)
    df = df[~df["discharge_disposition_id"].isin(DECES_HOSPICE_CODES)]
    print(f"[TRANSFORM] Décès/hospice exclus : {avant - len(df)} lignes retirées.")

    # --- Exclusion genre invalide ---
    df = df[df["gender"] != "Unknown/Invalid"]
    print(f"[TRANSFORM] Genre invalide exclu : {len(df)} lignes finales.")

    # --- Sélection des colonnes finales ---
    df = df[COLUMNS_TO_KEEP].copy()

    # --- Renommage pour correspondre au schéma SQL ---
    df = df.rename(columns={
        "A1Cresult": "a1c_result",
        "diabetesMed": "diabetes_med",
        "change": "change_med",
    })

    return df


# ----------------------------------------------------------------------
# 4. LOAD
# ----------------------------------------------------------------------

def load(df: pd.DataFrame, db_url: str, table_name: str = "patient_encounters"):
    """Charge le DataFrame nettoyé dans PostgreSQL.

    PostgreSQL limite chaque requête à 65535 paramètres au total
    (pas par ligne). Avec method="multi", pandas envoie
    chunksize * nb_colonnes paramètres par requête. Ici on a 26
    colonnes, donc chunksize doit rester sous 65535 / 26 ≈ 2520.
    On prend 2000 pour garder une marge de sécurité confortable.
    """
    print(f"[LOAD] Connexion à PostgreSQL ({DB_HOST}:{DB_PORT}/{DB_NAME}) ...")
    engine = create_engine(db_url)

    n_cols = len(df.columns)
    max_chunksize = 65535 // n_cols
    chunksize = min(2000, max_chunksize)

    df.to_sql(
        table_name,
        engine,
        if_exists="replace",   # remplace la table au lieu d'ajouter
        index=False,
        method="multi",
        chunksize=chunksize,
    )
    print(f"[LOAD] {len(df)} lignes insérées dans '{table_name}'.")


# ----------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------

def main():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Fichier introuvable : {RAW_DATA_PATH}\n"
            "Vérifie que diabetic_data.csv est bien dans data/raw/"
        )

    df_raw = extract(RAW_DATA_PATH)
    df_clean = transform(df_raw)
    load(df_clean, DB_URL)

    print("\n✅ Pipeline ETL terminé avec succès.")
    print(f"   Répartition de la cible :")
    print(df_clean["target"].value_counts(normalize=True).round(4) * 100)


if __name__ == "__main__":
    main()