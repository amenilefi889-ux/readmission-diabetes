# readmission-diabetes
Prédiction de réhospitalisation des patients diabétiques — XGBoost + SHAP + Streamlit

## Installation

```bash
pip install -r requirements.txt
```

## Pipeline ETL

Pour charger les données depuis le CSV vers PostgreSQL :

```bash
python src/etl/load_to_postgres.py
```

## Entraînement du modèle

```bash
python src/model/train.py
```

## Lancer l'application Streamlit

```bash
streamlit run app/streamlit_app.py
```

L'application sera accessible à : **http://localhost:8501**

### Fonctionnalités

- **Saisie patient** : Remplissez les informations du séjour via le panneau de gauche
- **Score de risque** : Obtient une estimation du risque de réhospitalisation dans les 30 jours
- **Explicabilité SHAP** : Visualisez les facteurs qui influencent chaque prédiction
  - Graphique d'explicabilité montrant les contributions positives/négatives
  - Tableau des facteurs les plus importants pour le patient

## Stack technique

- **Python** : Pandas, scikit-learn, XGBoost, SHAP
- **PostgreSQL** : Stockage des données
- **Streamlit** : Interface web interactive
- **Docker** : Containerisation et orchestration via `docker-compose.yml`

## Architecture

```
app/
  └── streamlit_app.py           # Interface Streamlit
models/
  ├── xgb_model.joblib           # Modèle entraîné
  └── feature_columns.joblib     # Colonnes de features
src/
  ├── db/
  │   └── schema.sql             # Schéma PostgreSQL
  ├── etl/
  │   └── load_to_postgres.py    # Pipeline ETL
  ├── features/
  │   └── build_features.py      # Feature engineering
  └── model/
      └── train.py               # Entraînement du modèle
data/
  ├── raw/
  │   └── diabetic_data.csv      # Dataset brut
  └── processed/                 # Données nettoyées
```

## Dataset

Source : [Kaggle - Diabetes 130-US Hospitals](https://www.kaggle.com/datasets/brandao/diabetes)

- 101 766 séjours hospitaliers
- 130 hôpitaux américains (1999–2008)
- 50+ variables cliniques par patient
