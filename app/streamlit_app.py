import sys
import os
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import shap
import matplotlib.pyplot as plt

# --- Fix : permet d'importer le package `src` depuis app/streamlit_app.py ---
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.features.build_features import transform_dataframe_for_model

MODEL_PATH = ROOT_DIR / "models" / "xgb_model.joblib"
FEATURE_COLUMNS_PATH = ROOT_DIR / "models" / "feature_columns.joblib"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata.joblib"

# Codes discharge_disposition_id correspondant à un décès ou une sortie en
# hospice : le modèle n'a jamais été entraîné sur ces cas (exclus dans l'ETL),
# donc on les retire des options proposées au praticien.
DECES_HOSPICE_CODES = {11, 13, 14, 19, 20, 21}
VALID_DISCHARGE_IDS = [i for i in range(1, 22) if i not in DECES_HOSPICE_CODES]

EXAMPLE_PATIENT = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[70-80)",
    "admission_type_id": 3,
    "discharge_disposition_id": 6,
    "admission_source_id": 7,
    "time_in_hospital": 8,
    "medical_specialty": "InternalMedicine",
    "num_lab_procedures": 58,
    "num_procedures": 1,
    "num_medications": 24,
    "number_outpatient": 1,
    "number_emergency": 0,
    "number_inpatient": 2,
    "number_diagnoses": 9,
    "max_glu_serum": "Norm",
    "a1c_result": ">7",
    "diabetes_med": "Yes",
    "change_med": "Ch",
    "diag_1": "250",
    "diag_2": "414",
    "diag_3": "428",
}

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, feature_columns


@st.cache_resource
def load_metadata():
    """Charge les seuils calculés lors de l'entraînement (Phase 2).
    Retourne des valeurs par défaut si le fichier n'existe pas encore
    (rétrocompatibilité avec un modèle entraîné avant cet ajout)."""
    if METADATA_PATH.exists():
        return joblib.load(METADATA_PATH)
    return {"threshold_f1": 0.5, "threshold_recall60": 0.5, "auc_roc": None, "auc_pr": None}


@st.cache_resource
def init_shap_explainer():
    """Initialise l'explainer SHAP une seule fois."""
    model, _ = load_model()
    explainer = shap.TreeExplainer(model)
    return explainer


def inject_custom_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.16), transparent 24%),
                radial-gradient(circle at left center, rgba(99, 102, 241, 0.10), transparent 20%),
                linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }

        .main .block-container {

        def apply_example_patient() -> None:
            for key, value in EXAMPLE_PATIENT.items():
                st.session_state[key] = value

        def reset_patient_form() -> None:
            for key in EXAMPLE_PATIENT:
                if key in st.session_state:
                    del st.session_state[key]
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #e5e7eb;
        }
                race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Unknown"], key="race")
                gender = st.selectbox("Genre", ["Female", "Male"], key="gender")
                age = st.selectbox("Tranche d'âge", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], key="age")
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] span {
                admission_type_id = st.selectbox("Type d'admission", [1, 2, 3, 4, 5, 6, 7, 8], key="admission_type_id")
                discharge_disposition_id = st.selectbox("Disposition de sortie", VALID_DISCHARGE_IDS, key="discharge_disposition_id")
                admission_source_id = st.selectbox("Source d'admission", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], key="admission_source_id")
                time_in_hospital = st.number_input("Durée séjour (jours)", min_value=1, max_value=30, value=4, key="time_in_hospital")
                medical_specialty = st.selectbox("Spécialité médicale", ["Unknown", "Cardiology", "InternalMedicine", "Family/GeneralPractice", "Emergency/Trauma", "Surgery", "Other"], key="medical_specialty")
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 28px;
                num_lab_procedures = st.number_input("Nombre de labos", min_value=0, max_value=200, value=40, key="num_lab_procedures")
                num_procedures = st.number_input("Nombre de procédures", min_value=0, max_value=20, value=2, key="num_procedures")
                num_medications = st.number_input("Nombre de médicaments", min_value=0, max_value=100, value=15, key="num_medications")
                number_outpatient = st.number_input("Nombre de visites ambulatoires", min_value=0, max_value=50, value=0, key="number_outpatient")
                number_emergency = st.number_input("Nombre d'urgences", min_value=0, max_value=50, value=0, key="number_emergency")
                number_inpatient = st.number_input("Nombre d'hospitalisations précédentes", min_value=0, max_value=50, value=0, key="number_inpatient")
                number_diagnoses = st.number_input("Nombre de diagnostics", min_value=1, max_value=16, value=8, key="number_diagnoses")
            margin: 0;
            font-size: 2rem;
                max_glu_serum = st.selectbox("Glycémie max", ["Norm", ">200", ">300", "Not tested"], key="max_glu_serum")
                a1c_result = st.selectbox("HbA1c", [">8", ">7", "Norm", "Not tested"], key="a1c_result")
                diabetes_med = st.selectbox("Médicament diabète", ["Yes", "No"], key="diabetes_med")
                change_med = st.selectbox("Changement de traitement", ["Ch", "No"], key="change_med")
            margin: 0.55rem 0 0 0;
            color: rgba(226, 232, 240, 0.88);
                diag_1 = st.text_input("Diagnostic 1 (code ICD-9)", "250", key="diag_1")
                diag_2 = st.text_input("Diagnostic 2 (code ICD-9)", "401", key="diag_2")
                diag_3 = st.text_input("Diagnostic 3 (code ICD-9)", "428", key="diag_3")
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.34rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: white;
        }

        .section-title {
            font-size: 0.78rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #64748b;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .info-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
        }

        .info-card h3 {
            margin: 0;
            color: #0f172a;
            font-size: 1.02rem;
        }

        .info-card p {
            margin: 0.35rem 0 0 0;
            color: #475569;
            font-size: 0.93rem;
        }

        .result-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(248, 250, 252, 0.92));
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 16px 32px rgba(15, 23, 42, 0.08);
        }

        .result-grid {
            display: grid;
            grid-template-columns: 1.25fr 0.75fr;
            gap: 1rem;
        }

        .risk-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #64748b;
            font-weight: 700;
        }

        .risk-value {
            font-size: 2.6rem;
            line-height: 1;
            font-weight: 800;
            margin: 0.1rem 0 0.55rem 0;
            color: #0f172a;
        }

        .risk-note {
            margin-top: 0.45rem;
            color: #475569;
            font-size: 0.95rem;
        }

        .risk-track {
            width: 100%;
            height: 14px;
            background: #e2e8f0;
            border-radius: 999px;
            overflow: hidden;
            margin-top: 0.6rem;
        }

        .risk-fill {
            height: 100%;
            border-radius: 999px;
        }

        .mini-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.34rem 0.75rem;
            border-radius: 999px;
            background: #eff6ff;
            color: #1d4ed8;
            font-size: 0.82rem;
            font-weight: 700;
            margin-right: 0.4rem;
            margin-top: 0.4rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.18);
            padding: 1rem 1rem 0.9rem 1rem;
            border-radius: 18px;
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.05);
        }

        .shap-note {
            background: rgba(255, 255, 255, 0.84);
            border-left: 4px solid #2563eb;
            border-radius: 16px;
            padding: 0.9rem 1rem;
            color: #334155;
            margin-top: 0.4rem;
        }

        .summary-card {
            background: rgba(15, 23, 42, 0.96);
            border-radius: 22px;
            padding: 1rem 1.1rem 1.05rem 1.1rem;
            border: 1px solid rgba(148, 163, 184, 0.18);
            color: white;
            box-shadow: 0 16px 32px rgba(15, 23, 42, 0.10);
            margin-top: 0.4rem;
        }

        .summary-card h4 {
            margin: 0;
            font-size: 1rem;
            color: white;
        }

        .summary-card p {
            margin: 0.32rem 0 0 0;
            color: rgba(226, 232, 240, 0.82);
            font-size: 0.92rem;
        }

        .summary-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.9rem;
        }

        .summary-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.34rem 0.72rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.12);
            font-size: 0.82rem;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_profile(risk_percent: float) -> tuple[str, str, str]:
    if risk_percent >= 50:
        return "Élevé", "#dc2626", "Risque à surveiller de près"
    if risk_percent >= 20:
        return "Modéré", "#d97706", "Risque intermédiaire"
    return "Faible", "#16a34a", "Risque relativement contenu"


def render_header(metadata: dict, threshold_f1_value: float) -> None:
    auc_roc = metadata.get("auc_roc")
    auc_pr = metadata.get("auc_pr")
    auc_roc_text = f"{auc_roc:.3f}" if auc_roc is not None else "N/A"
    auc_pr_text = f"{auc_pr:.3f}" if auc_pr is not None else "N/A"

    st.markdown(
        """
        <div class="hero">
            <h1>Prédiction du risque de réhospitalisation</h1>
            <p>Un dashboard clinique compact pour estimer la probabilité de retour à l'hôpital dans les 30 jours et expliquer les facteurs dominants.</p>
            <div class="badge-row">
                <span class="badge">ETL PostgreSQL</span>
                <span class="badge">XGBoost</span>
                <span class="badge">SHAP</span>
                <span class="badge">Streamlit</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = st.columns(3)
    with cards[0]:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="section-title">Seuil de décision</div>
                <h3>{threshold_f1_value:.3f}</h3>
                <p>Optimisé sur validation pour équilibrer précision et rappel.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cards[1]:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="section-title">Performance</div>
                <h3>AUC-ROC {auc_roc_text} | AUC-PR {auc_pr_text}</h3>
                <p>Mesures de référence affichées pour contextualiser le score.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cards[2]:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="section-title">Couverture</div>
                <h3>69 970 séjours</h3>
                <p>Données nettoyées et chargées dans PostgreSQL pour l'entraînement.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_result(proba: float, threshold: float, risk: float) -> None:
    status, color, note = risk_profile(risk)
    predicted_readmission = proba >= threshold
    width_percent = max(0.0, min(risk, 100.0))

    left, right = st.columns([1.2, 0.8])
    with left:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="risk-label">Risque estimé</div>
                <div class="risk-value" style="color:{color};">{risk:.1f}%</div>
                <div class="risk-track"><div class="risk-fill" style="width:{width_percent}%; background:{color};"></div></div>
                <div class="risk-note">{note}. Le score est interprété avec le seuil de décision {threshold:.3f}.</div>
                <div class="mini-chip">Décision modèle: {'Réadmission probable' if predicted_readmission else 'Pas de réadmission attendue'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.metric("Niveau", status)
        st.metric("Probabilité brute", f"{proba:.4f}")


def render_patient_summary(input_data: dict) -> None:
    chips = [
        f"Âge {input_data['age']}",
        f"Séjour {input_data['time_in_hospital']} j",
        f"Médicaments {input_data['num_medications']}",
        f"HbA1c {input_data['a1c_result']}",
        f"Antécédents IP {input_data['number_inpatient']}",
        f"Diagnostics {input_data['number_diagnoses']}",
    ]

    chips_html = "".join(f'<span class="summary-chip">{chip}</span>' for chip in chips)
    st.markdown(
        f"""
        <div class="summary-card">
            <h4>Résumé patient</h4>
            <p>Vue condensée des principales variables saisies avant interprétation du score.</p>
            <div class="summary-chip-row">
                {chips_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_example_patient() -> None:
    for key, value in EXAMPLE_PATIENT.items():
        st.session_state[key] = value


def reset_patient_form() -> None:
    for key in EXAMPLE_PATIENT:
        st.session_state.pop(key, None)



st.set_page_config(page_title="Prédiction de réhospitalisation", page_icon="🏥", layout="wide")
inject_custom_styles()

model, feature_columns = load_model()
metadata = load_metadata()
threshold_f1 = metadata["threshold_f1"]

render_header(metadata, threshold_f1)

if metadata.get("auc_pr") is not None:
    st.sidebar.markdown("### Modèle entraîné")
    st.sidebar.caption(
        f"AUC-ROC={metadata['auc_roc']:.3f} | AUC-PR={metadata['auc_pr']:.3f} | seuil={threshold_f1:.3f}"
    )

st.sidebar.markdown("## Informations patient")
st.sidebar.caption("Renseignez les données cliniques principales pour obtenir un score et des explications.")

action_cols = st.sidebar.columns(2)
with action_cols[0]:
    if st.button("Charger un exemple"):
        apply_example_patient()
        st.rerun()
with action_cols[1]:
    if st.button("Réinitialiser"):
        reset_patient_form()
        st.rerun()

with st.sidebar.expander("Démographie", expanded=True):
    race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Unknown"], key="race")
    gender = st.selectbox("Genre", ["Female", "Male"], key="gender")
    age = st.selectbox("Tranche d'âge", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], key="age")

with st.sidebar.expander("Séjour hospitalier", expanded=True):
    admission_type_id = st.selectbox("Type d'admission", [1, 2, 3, 4, 5, 6, 7, 8], key="admission_type_id")
    discharge_disposition_id = st.selectbox("Disposition de sortie", VALID_DISCHARGE_IDS, key="discharge_disposition_id")
    admission_source_id = st.selectbox("Source d'admission", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], key="admission_source_id")
    time_in_hospital = st.number_input("Durée séjour (jours)", min_value=1, max_value=30, value=4, key="time_in_hospital")
    medical_specialty = st.selectbox("Spécialité médicale", ["Unknown", "Cardiology", "InternalMedicine", "Family/GeneralPractice", "Emergency/Trauma", "Surgery", "Other"], key="medical_specialty")

with st.sidebar.expander("Mesures cliniques", expanded=True):
    num_lab_procedures = st.number_input("Nombre de labos", min_value=0, max_value=200, value=40, key="num_lab_procedures")
    num_procedures = st.number_input("Nombre de procédures", min_value=0, max_value=20, value=2, key="num_procedures")
    num_medications = st.number_input("Nombre de médicaments", min_value=0, max_value=100, value=15, key="num_medications")
    number_outpatient = st.number_input("Nombre de visites ambulatoires", min_value=0, max_value=50, value=0, key="number_outpatient")
    number_emergency = st.number_input("Nombre d'urgences", min_value=0, max_value=50, value=0, key="number_emergency")
    number_inpatient = st.number_input("Nombre d'hospitalisations précédentes", min_value=0, max_value=50, value=0, key="number_inpatient")
    number_diagnoses = st.number_input("Nombre de diagnostics", min_value=1, max_value=16, value=8, key="number_diagnoses")

with st.sidebar.expander("Biologie et traitement", expanded=True):
    max_glu_serum = st.selectbox("Glycémie max", ["Norm", ">200", ">300", "Not tested"], key="max_glu_serum")
    a1c_result = st.selectbox("HbA1c", [">8", ">7", "Norm", "Not tested"], key="a1c_result")
    diabetes_med = st.selectbox("Médicament diabète", ["Yes", "No"], key="diabetes_med")
    change_med = st.selectbox("Changement de traitement", ["Ch", "No"], key="change_med")

with st.sidebar.expander("Diagnostics", expanded=False):
    diag_1 = st.text_input("Diagnostic 1 (code ICD-9)", "250", key="diag_1")
    diag_2 = st.text_input("Diagnostic 2 (code ICD-9)", "401", key="diag_2")
    diag_3 = st.text_input("Diagnostic 3 (code ICD-9)", "428", key="diag_3")

if st.sidebar.button("Prédire"):
    input_df = pd.DataFrame([{
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "medical_specialty": medical_specialty,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "max_glu_serum": max_glu_serum,
        "a1c_result": a1c_result,
        "diabetes_med": diabetes_med,
        "change_med": change_med,
        "diag_1": diag_1,
        "diag_2": diag_2,
        "diag_3": diag_3,
    }])

    X_input = transform_dataframe_for_model(input_df, feature_columns=feature_columns)
    proba = model.predict_proba(X_input)[0, 1]
    risk = round(float(proba) * 100, 1)
    predicted_readmission = proba >= threshold_f1

    st.markdown('<div class="section-title">Résultat clinique</div>', unsafe_allow_html=True)
    render_prediction_result(proba, threshold_f1, risk)
    render_patient_summary({
        "age": age,
        "time_in_hospital": time_in_hospital,
        "num_medications": num_medications,
        "a1c_result": a1c_result,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
    })

    tab_result, tab_shap = st.tabs(["Synthèse", "Explicabilité"])

    with tab_result:
        info_cols = st.columns(3)
        with info_cols[0]:
            st.metric("Seuil utilisé", f"{threshold_f1:.3f}")
        with info_cols[1]:
            st.metric("Décision", "Réadmission" if predicted_readmission else "Pas de réadmission")
        with info_cols[2]:
            st.metric("Classe", "Risque élevé" if risk >= 50 else "Risque modéré" if risk >= 20 else "Risque faible")

        st.markdown(
            """
            <div class="shap-note">
                Le score ci-dessus résume la sortie du modèle. L'onglet Explicabilité montre les variables qui ont le plus contribué à la prédiction.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_shap:
        st.markdown('<div class="section-title">Explicabilité SHAP</div>', unsafe_allow_html=True)

        try:
            with st.spinner("Calcul des explications SHAP..."):
                explainer = init_shap_explainer()
                shap_values = explainer.shap_values(X_input)

            if len(shap_values.shape) == 2:
                shap_values = shap_values[:, :, np.newaxis]

            class_idx = 1 if shap_values.shape[2] > 1 else 0
            shap_values_class = shap_values[:, :, class_idx]

            if isinstance(explainer.expected_value, (list, np.ndarray)):
                base_value = explainer.expected_value[class_idx]
            else:
                base_value = explainer.expected_value

            explanation = shap.Explanation(
                values=shap_values_class[0],
                base_values=base_value,
                data=X_input.iloc[0].values,
                feature_names=X_input.columns.tolist(),
            )

            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.waterfall(explanation, show=False, max_display=15)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.caption(
                "Le graphique montre comment chaque facteur augmente (en rouge) ou diminue (en bleu) la probabilité de réadmission par rapport à la baseline."
            )

            st.subheader("Facteurs les plus importants")
            top_n = 10
            feature_importance = pd.DataFrame({
                "Feature": X_input.columns,
                "SHAP Value": shap_values_class[0],
                "Input Value": X_input.iloc[0].values,
            })
            feature_importance["|SHAP|"] = np.abs(feature_importance["SHAP Value"])
            feature_importance = feature_importance.nlargest(top_n, "|SHAP|")
            feature_importance = feature_importance.drop("|SHAP|", axis=1)

            st.dataframe(feature_importance, use_container_width=True, hide_index=True)

        except Exception as e:
            st.warning(f"Erreur lors du calcul SHAP : {e}")
            st.info("Les explications SHAP requièrent une évaluation plus approfondie du modèle.")

else:
    st.markdown(
        """
        <div class="result-card">
            <div class="risk-label">Avant de lancer la prédiction</div>
            <div class="risk-note">Complétez les champs dans la barre latérale, puis cliquez sur <strong>Prédire</strong> pour afficher le score et les explications.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )