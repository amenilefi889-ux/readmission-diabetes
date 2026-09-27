# 🚀 Guide Déploiement et Lancement

Crée une configuration locale avant le premier démarrage :

```bash
copy .env.example .env       # Windows
cp .env.example .env         # macOS/Linux
```

Renseigne ensuite `POSTGRES_PASSWORD` et `DB_PASSWORD` dans `.env`. Le fichier
`.env` reste local et n'est pas versionné.

## ⚡ Démarrage Rapide (30 secondes)

### Pour les Évaluateurs
```bash
# 1. Terminal - Lancer les services
docker-compose up -d

# 2. Vérifier que tout fonctionne
docker-compose ps

# 3. Ouvrir dans le navigateur
# URL: http://localhost:8501
```

Après que PostgreSQL est healthy, l'application exécute automatiquement l'ETL
et l'entraînement avant de démarrer Streamlit. Suis la progression avec
`docker-compose logs -f app`, puis:
- PostgreSQL: http://localhost:5433 (utilisateur: diabetes_user)
- Streamlit: http://localhost:8501

---

## 📋 Configuration Complète (Dev/Test)

### Pré-requis
- Python 3.10+
- Docker & Docker Compose
- 2GB RAM libre minimum
- Port 5433 (PostgreSQL) et 8501 (Streamlit) disponibles

### Installation Locale (sans Docker)

```bash
# 1. Cloner le repo
cd readmission-diabetes

# 2. Créer env virtuel
python -m venv .venv
.venv\Scripts\Activate  # Windows
source .venv/bin/activate  # macOS/Linux

# 3. Installer les dépendances
pip install -r requirements-prod.txt

# 4. Configurer PostgreSQL localement
# Option A: Docker seul (PostgreSQL)
docker run -d \
  --name diabetes_postgres \
  -e POSTGRES_USER=diabetes_user \
   -e POSTGRES_PASSWORD=<mot-de-passe-local> \
  -e POSTGRES_DB=diabetes_readmission \
  -p 5433:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  -v ./src/db/schema.sql:/docker-entrypoint-initdb.d/schema.sql \
  postgres:16-alpine

# Option B: PostgreSQL natif (macOS/Linux)
psql -U postgres -f src/db/schema.sql

# 5. Charger les données
python src/etl/load_to_postgres.py

# 6. Vérifier que les modèles existent
ls models/
# Doit afficher: feature_columns.joblib, xgb_model.joblib

# 7. Lancer Streamlit
streamlit run app/streamlit_app.py
```

---

## 🐳 Déploiement Docker

### Architecture

```
┌────────────────────────────┐
│   Docker Compose Network   │
├────────────────────────────┤
│ ┌──────────────────────┐   │
│ │ PostgreSQL Service   │   │
│ │ Port: 5433:5432      │   │
│ │ Data: postgres_data  │   │
│ └──────────────────────┘   │
│           ▲                │
│           │ (internal)     │
│           │                │
│ ┌──────────────────────┐   │
│ │ Streamlit App        │   │
│ │ Port: 8501:8501      │   │
│ │ Depends: postgres    │   │
│ └──────────────────────┘   │
└────────────────────────────┘
```

### Commandes Docker

**Lancer le système complet**:
```bash
docker-compose up -d

# Suivi de la progression
docker-compose logs -f app
```

**Arrêter le système**:
```bash
docker-compose stop
```

**Redémarrer**:
```bash
docker-compose restart
```

**Arrêter et supprimer les conteneurs (garde les données)**:
```bash
docker-compose down
```

**Arrêter et supprimer tout (données incluses)**:
```bash
docker-compose down -v
```

**Vérifier le statut**:
```bash
docker-compose ps
```

**Voir les logs**:
```bash
# Tous les services
docker-compose logs -f

# App seulement
docker-compose logs -f app

# PostgreSQL seulement
docker-compose logs -f postgres

# Dernières 100 lignes
docker-compose logs --tail 100
```

**Se connecter au shell d'un conteneur**:
```bash
# Conteneur app
docker-compose exec app bash

# Conteneur PostgreSQL
docker-compose exec postgres psql -U diabetes_user -d diabetes_readmission
```

---

## ✅ Vérification du Système

### Checklist de Lancement

```bash
# 1. Services running
docker-compose ps
# ✅ postgres: healthy
# ✅ app: running

# 2. PostgreSQL accessible
psql -h localhost -p 5433 -U diabetes_user -d diabetes_readmission -c "SELECT COUNT(*) FROM patient_encounters;"
# Doit afficher: 69970 records

# 3. Modèles chargés
python -c "import joblib; m = joblib.load('models/xgb_model.joblib'); print(f'✅ Model loaded: {m}')"

# 4. Streamlit responsive
curl http://localhost:8501
# Doit retourner le HTML de l'app

# 5. Tests (optionnel)
pytest tests/test_pipeline.py -v
# ✅ 10 passed
```

---

## 🧪 Tests et Validation

### Lancer les Tests
```bash
# Tous les tests
pytest tests/test_pipeline.py -v

# Test spécifique
pytest tests/test_pipeline.py::TestDiagnosticGrouping::test_diabetes_code -v

# Avec couverture
pytest tests/test_pipeline.py --cov=src --cov-report=html
```

### Tests Manuels dans Streamlit

1. **Test 1: Patient Haut Risque**
   - Age: [60-70)
   - Medications: 20
   - Prior Inpatient: 3
   - Expected: 🔴 High risk (>50%)

2. **Test 2: Patient Bas Risque**
   - Age: [20-30)
   - Medications: 2
   - Prior Inpatient: 0
   - Expected: 🟢 Low risk (<20%)

3. **Test 3: SHAP Visualization**
   - Vérifier que le force plot s'affiche
   - Voir les features en rouge (risk) et bleu (protective)

---

## 🔍 Troubleshooting

### PostgreSQL n'est pas ready

**Symptôme**: `ERROR: Database connection failed`

**Solutions**:
```bash
# 1. Vérifier que le service est running
docker-compose ps postgres

# 2. Vérifier les logs PostgreSQL
docker-compose logs postgres

# 3. Augmenter le délai (dans docker-compose.yml)
# Augmenter `retries: 5` à `retries: 10`

# 4. Redémarrer PostgreSQL
docker-compose restart postgres

# 5. Attendre manuellement
sleep 20
```

### Port déjà utilisé

**Symptôme**: `ERROR: driver failed programming external connectivity on endpoint`

**Solutions**:
```bash
# 1. Identifier le processus
netstat -ano | grep :5433  # Windows
lsof -i :5433  # macOS/Linux

# 2. Changer le port dans docker-compose.yml
# Modifier: "5434:5432" (nouveau port)

# 3. Ou arrêter l'application qui l'utilise
docker-compose down
```

### Streamlit affiche blanc

**Symptôme**: Page blanche à http://localhost:8501

**Solutions**:
```bash
# 1. Vérifier les logs de l'app
docker-compose logs app | tail -50

# 2. Redémarrer Streamlit
docker-compose restart app

# 3. Vérifier que les modèles existent
docker-compose exec app ls models/

# 4. Nettoyer les caches Streamlit
docker-compose exec app rm -rf ~/.streamlit/cache
docker-compose restart app
```

### Erreur de connexion à PostgreSQL

**Symptôme**: `psycopg2.OperationalError: FATAL: password authentication failed`

**Solutions**:
```bash
# 1. Vérifier les credentials dans docker-compose.yml
# POSTGRES_USER: diabetes_user
# POSTGRES_PASSWORD: valeur définie dans .env
# DB_HOST: postgres (service name!)

# 2. Vérifier la chaîne de connexion
docker-compose exec app python -c "import os; print('DB_HOST=' + os.environ['DB_HOST'] + ', DB_NAME=' + os.environ['DB_NAME'])"

# 3. Réinitialiser les données
docker-compose down -v
docker-compose up -d
# Attendre 20 secondes
python src/etl/load_to_postgres.py
```

### SHAP ne s'affiche pas

**Symptôme**: "SHAP calculation failed" dans Streamlit

**Solutions**:
```bash
# 1. Vérifier que SHAP est installé
docker-compose exec app pip show shap

# 2. Relancer l'app
docker-compose restart app

# 3. Vérifier la taille du modèle
docker-compose exec app ls -lh models/xgb_model.joblib
```

---

## 📊 Monitoring en Temps Réel

### Surveiller les Ressources
```bash
# Affichage en direct des stats
docker stats

# Specific container
docker stats diabetes_streamlit
```

### Vérifier les Événements
```bash
docker-compose events
```

### Logs détaillés
```bash
# Follow app logs
docker-compose logs -f --timestamps app

# Last 50 lines
docker-compose logs --tail 50 app

# With grep filter
docker-compose logs app | grep ERROR
```

---

## 🔄 Mise à Jour et Maintenance

### Reconstruire les Images
```bash
# Rebuild Dockerfile
docker-compose build --no-cache

# Puis redémarrer
docker-compose up -d --force-recreate
```

### Mettre à Jour le Code
```bash
# Récupérer les changements
git pull

# Reconstruire l'image
docker-compose build app

# Redémarrer
docker-compose up -d
```

### Sauvegarder les Données
```bash
# Exporter les données PostgreSQL
docker-compose exec postgres pg_dump -U diabetes_user diabetes_readmission > backup.sql

# Restaurer
docker-compose exec postgres psql -U diabetes_user diabetes_readmission < backup.sql

# Sauvegarder le volume Docker
docker run --rm -v postgres_data:/data -v $(pwd):/backup busybox tar czf /backup/postgres_backup.tar.gz -C / data
```

---

## 📈 Performance

### Optimisations
- PostgreSQL: Connection pooling configuré via SQLAlchemy
- Streamlit: Caching des modèles et SHAP explainer
- XGBoost: Predict sur CPU (GPU disponible si configuré)

### Benchmark
```bash
# Temps de prédiction
time python -c "
import joblib
X = joblib.load('models/feature_columns.joblib')
model = joblib.load('models/xgb_model.joblib')
# Single prediction: ~10-50ms
"
```

---

## 🎯 Prochaines Étapes

1. **Production**:
   - Configurer SSL/TLS
   - Load balancer (Nginx)
   - Monitoring (Prometheus/Grafana)
   - Logging centralisé (ELK)

2. **Modèle**:
   - Retraining automatique
   - A/B testing
   - Drift detection

3. **Données**:
   - Backup automatique
   - Archivage
   - GDPR compliance

---

**Pour la soutenance: Utilisez `docker-compose up -d` et ouvrez http://localhost:8501** ✅
