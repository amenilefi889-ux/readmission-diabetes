"""
Pipeline ETL — Diabetes 130-US Hospitals
==========================================
Extrait, nettoie et charge le dataset diabetic_data.csv vers PostgreSQL,
en reproduisant exactement les étapes de nettoyage validées dans le
notebook 01_exploration.ipynb (Phase 1 - EDA).

La configuration DB est lue depuis les variables d'environnement (avec des
valeurs par défaut pour l'exécution locale hors Docker), afin que ce script
fonctionne à la fois :
  - en local (Windows/PowerShell) : localhost:5433 (port publié par Docker Compose)
  - dans le conteneur Docker : postgres:5432 (nom du service + port interne)

Usage:
    python src/etl/load_to_postgres.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.utils.logging_utils import configure_json_logger

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------

RAW_DATA_PATH = "data/raw/diabetic_data.csv"
SCHEMA_PATH = Path("src/db/schema.sql")

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5433")
DB_NAME = os.environ.get("DB_NAME", "diabetes_readmission")

DB_USER = os.environ.get("DB_USER", "diabetes_user")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "diabetes_pass")
DB_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
LOGGER = configure_json_logger(__name__)

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
    LOGGER.info("extract_start", extra={"path": path})
    df = pd.read_csv(path, na_values="?", low_memory=False)
    LOGGER.info("extract_complete", extra={"rows": len(df)})
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
    LOGGER.info("transform_missing_values_complete", extra={"rows": n_before})

    # --- Déduplication : 1 patient = 1er séjour ---
    df = df.sort_values("encounter_id").drop_duplicates(
        subset="patient_nbr", keep="first"
    )
    LOGGER.info("transform_deduplicate_complete", extra={"rows": len(df)})

    # --- Exclusion décès / hospice ---
    avant = len(df)
    df = df[~df["discharge_disposition_id"].isin(DECES_HOSPICE_CODES)]
    LOGGER.info(
        "transform_discharge_filter_complete",
        extra={"removed_rows": avant - len(df), "rows": len(df)},
    )

    # --- Exclusion genre invalide ---
    df = df[df["gender"] != "Unknown/Invalid"]
    LOGGER.info("transform_gender_filter_complete", extra={"rows": len(df)})

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
    """Charge le DataFrame nettoyé dans PostgreSQL."""
    LOGGER.info(
        "load_start",
        extra={"host": DB_HOST, "port": DB_PORT, "database": DB_NAME, "table": table_name},
    )
    engine = create_engine(db_url)
    safe_chunksize = max(1, 65535 // max(len(df.columns), 1))
    chunksize = min(1000, safe_chunksize)

    with engine.begin() as connection:
        inspector = inspect(connection)
        if table_name not in inspector.get_table_names():
            if not SCHEMA_PATH.exists():
                raise FileNotFoundError(f"Schéma introuvable : {SCHEMA_PATH}")

            schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
            for statement in schema_sql.split(";"):
                statement = statement.strip()
                if statement:
                    connection.execute(text(statement))

        connection.execute(text(f"DELETE FROM {table_name}"))

    try:
        df.to_sql(
            table_name,
            engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=chunksize,
        )
        LOGGER.info(
            "load_complete",
            extra={"rows": len(df), "table": table_name, "chunksize": chunksize},
        )
    except Exception:
        LOGGER.exception("load_failed", extra={"table": table_name})
        raise


# ----------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------

def main():
    try:
        if not os.path.exists(RAW_DATA_PATH):
            raise FileNotFoundError(
                f"Fichier introuvable : {RAW_DATA_PATH}\n"
                "Vérifie que diabetic_data.csv est bien dans data/raw/"
            )

        df_raw = extract(RAW_DATA_PATH)
        df_clean = transform(df_raw)
        load(df_clean, DB_URL)

        target_distribution = df_clean["target"].value_counts(normalize=True).round(4).to_dict()
        LOGGER.info(
            "etl_complete",
            extra={"target_distribution": target_distribution},
        )
    except Exception:
        LOGGER.exception("etl_failed")
        raise


if __name__ == "__main__":
    main()
