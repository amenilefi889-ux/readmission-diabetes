"""
Feature engineering — Diabetes 130-US Hospitals
=================================================
Lit la table `patient_encounters` depuis PostgreSQL et construit la
matrice de features prête pour l'entraînement (encodage age, regroupement
des codes diagnostics ICD-9, encodage des catégorielles).

La configuration DB est lue depuis les variables d'environnement (avec des
valeurs par défaut pour les paramètres non sensibles), afin que ce module
fonctionne à la fois :
  - en local (Windows/PowerShell) : localhost:5433
  - dans le conteneur Docker : postgres:5432
  - importé par l'app Streamlit (qui tourne aussi dans Docker)

Usage:
    from src.features.build_features import load_and_build_features
    X, y = load_and_build_features()
"""

import os
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

DB_USER = os.environ.get("DB_USER", "diabetes_user")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5433")
DB_NAME = os.environ.get("DB_NAME", "diabetes_readmission")

TABLE_NAME = "patient_encounters"

# Mapping ordinal des tranches d'âge : "[0-10)" -> 0, "[10-20)" -> 1, ...
AGE_ORDER = [f"[{i}-{i+10})" for i in range(0, 100, 10)]
AGE_MAPPING = {label: i for i, label in enumerate(AGE_ORDER)}


def _diag_to_group(code: str) -> str:
    """Regroupe un code ICD-9 (diag_1/2/3) en grande catégorie clinique.

    Reprend le regroupement classique utilisé dans la littérature sur ce
    dataset (Strack et al., 2014) : circulatoire, diabète, respiratoire,
    digestif, blessures, musculo-squelettique, génito-urinaire, néoplasmes,
    autre.
    """
    if code == "Unknown" or code is None:
        return "Unknown"

    # Codes V/E = causes externes / facteurs de santé -> "Other"
    if code.startswith("V") or code.startswith("E"):
        return "Other"

    try:
        num = float(code)
    except ValueError:
        return "Other"

    # Diabète (250.xx) en priorité, avant les plages numériques générales
    if 250 <= num < 251:
        return "Diabetes"
    if (390 <= num <= 459) or num == 785:
        return "Circulatory"
    if (460 <= num <= 519) or num == 786:
        return "Respiratory"
    if (520 <= num <= 579) or num == 787:
        return "Digestive"
    if 800 <= num <= 999:
        return "Injury"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if (580 <= num <= 629) or num == 788:
        return "Genitourinary"
    if 140 <= num <= 239:
        return "Neoplasms"
    return "Other"


def get_top_specialties(df: pd.DataFrame) -> list[str]:
    return (
        df["medical_specialty"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .nlargest(10)
        .index
        .tolist()
    )


def load_raw(table_name: str = TABLE_NAME) -> pd.DataFrame:
    """Charge la table depuis PostgreSQL."""
    if not DB_PASSWORD:
        raise RuntimeError("DB_PASSWORD n'est pas défini — crée un fichier .env")
    db_url = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)
    print(f"[FEATURES] Lecture de la table '{table_name}' ...")
    df = pd.read_sql_table(table_name, engine)
    print(f"[FEATURES] {len(df)} lignes chargées depuis PostgreSQL.")
    return df


def _prepare_feature_dataframe(df: pd.DataFrame, top_specialties=None) -> pd.DataFrame:
    """Applique les mêmes transformations que lors de l'entraînement."""
    df = df.copy()

    # --- Age ordinal ---
    df["age_ordinal"] = df["age"].map(AGE_MAPPING)

    # --- Regroupement des diagnostics ---
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_group"] = df[col].apply(_diag_to_group)

    # --- Flags binaires simples ---
    df["a1c_tested"] = (df["a1c_result"] != "Not tested").astype(int)
    df["max_glu_tested"] = (df["max_glu_serum"] != "Not tested").astype(int)
    df["diabetes_med_flag"] = (df["diabetes_med"] == "Yes").astype(int)
    df["change_med_flag"] = (df["change_med"] == "Ch").astype(int)
    df["gender_male"] = (df["gender"] == "Male").astype(int)

    # --- Réduction de cardinalité : medical_specialty a >70 valeurs,
    # on garde les 10 plus fréquentes et on regroupe le reste ---
    if top_specialties is None:
        top_specialties = get_top_specialties(df)

    specialty_values = df["medical_specialty"].fillna("Unknown").astype(str)
    df["medical_specialty"] = specialty_values.where(specialty_values.isin(top_specialties), "Other")

    # --- Features dérivées : utilisation du système de santé / intensité des soins ---
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    df["medications_per_day"] = df["num_medications"] / df["time_in_hospital"].clip(lower=1)
    df["procedures_per_day"] = df["num_procedures"] / df["time_in_hospital"].clip(lower=1)
    df["lab_procedures_per_day"] = df["num_lab_procedures"] / df["time_in_hospital"].clip(lower=1)
    df["has_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)
    df["has_prior_emergency"] = (df["number_emergency"] > 0).astype(int)

    return df


def transform_dataframe_for_model(df: pd.DataFrame, feature_columns=None, top_specialties=None) -> pd.DataFrame:
    """Transforme un DataFrame brut en matrice compatible avec le modèle."""
    df = _prepare_feature_dataframe(df, top_specialties=top_specialties)

    categorical_cols = [
        "race", "admission_type_id", "discharge_disposition_id",
        "admission_source_id", "medical_specialty",
        "diag_1_group", "diag_2_group", "diag_3_group",
        "max_glu_serum", "a1c_result",
    ]

    numeric_cols = [
        "age_ordinal", "time_in_hospital", "num_lab_procedures",
        "num_procedures", "num_medications", "number_outpatient",
        "number_emergency", "number_inpatient", "number_diagnoses",
        "a1c_tested", "max_glu_tested", "diabetes_med_flag",
        "change_med_flag", "gender_male",
        "total_prior_visits", "medications_per_day", "procedures_per_day",
        "lab_procedures_per_day", "has_prior_inpatient", "has_prior_emergency",
    ]

    df[categorical_cols] = df[categorical_cols].astype(str)

    X = pd.get_dummies(df[categorical_cols + numeric_cols], columns=categorical_cols, drop_first=True)
    X.columns = [str(c) for c in X.columns]

    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)

    return X


def build_features(df: pd.DataFrame, top_specialties=None) -> tuple[pd.DataFrame, pd.Series]:
    """Construit la matrice de features X et la cible y."""
    df = _prepare_feature_dataframe(df, top_specialties=top_specialties)

    categorical_cols = [
        "race", "admission_type_id", "discharge_disposition_id",
        "admission_source_id", "medical_specialty",
        "diag_1_group", "diag_2_group", "diag_3_group",
        "max_glu_serum", "a1c_result",
    ]

    numeric_cols = [
        "age_ordinal", "time_in_hospital", "num_lab_procedures",
        "num_procedures", "num_medications", "number_outpatient",
        "number_emergency", "number_inpatient", "number_diagnoses",
        "a1c_tested", "max_glu_tested", "diabetes_med_flag",
        "change_med_flag", "gender_male",
        "total_prior_visits", "medications_per_day", "procedures_per_day",
        "lab_procedures_per_day", "has_prior_inpatient", "has_prior_emergency",
    ]

    df[categorical_cols] = df[categorical_cols].astype(str)
    X = pd.get_dummies(df[categorical_cols + numeric_cols], columns=categorical_cols, drop_first=True)
    X.columns = [str(c) for c in X.columns]
    y = df["target"]

    print(f"[FEATURES] Matrice construite : {X.shape[0]} lignes, {X.shape[1]} colonnes.")
    return X, y


def load_and_build_features(return_metadata: bool = False):
    df = load_raw()
    top_specialties = get_top_specialties(df)
    X, y = build_features(df, top_specialties=top_specialties)
    if return_metadata:
        return X, y, top_specialties
    return X, y


if __name__ == "__main__":
    X, y = load_and_build_features()
    print("\nAperçu des features :")
    print(X.head())
    print("\nDistribution de la cible :")
    print(y.value_counts(normalize=True).round(4) * 100)
