# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

WORKDIR /app

# Copier les requirements
COPY requirements-prod.txt .

# Installer les dépendances Python
RUN --mount=type=cache,target=/root/.cache/pip \
	pip install \
	--prefer-binary \
	--retries 10 \
	--timeout 180 \
	-r requirements-prod.txt

# Copier le code
COPY . .

# Créer le répertoire models s'il n'existe pas
RUN mkdir -p models

# Exposer le port Streamlit
EXPOSE 8501

# Lancer Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
