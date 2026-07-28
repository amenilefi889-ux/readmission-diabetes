FROM python:3.12-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements-prod.txt .

# Installer les dépendances Python
# Si le fichier avec versions spécifiques échoue, utiliser les dépendances minimales
RUN pip install --no-cache-dir -r requirements-prod.txt || \
    pip install --no-cache-dir numpy pandas scikit-learn xgboost shap joblib streamlit SQLAlchemy psycopg psycopg2-binary matplotlib seaborn Pillow python-dotenv pytest requests

# Copier le code
COPY . .

# Créer le répertoire models s'il n'existe pas
RUN mkdir -p models

# Exposer le port Streamlit
EXPOSE 8501

# Lancer Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
