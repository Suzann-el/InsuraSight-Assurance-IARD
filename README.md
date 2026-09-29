# 🛡️ InsuraSight — Projet Data Science Assurance IARD

Projet end-to-end de suivi de sinistralité et de modélisation actuarielle pour un portefeuille assurance IARD (auto, habitation, santé).

## 🐳 Docker Hub

Images disponibles publiquement :

docker pull suzannelhaouzi/insura-sight-api
docker pull suzannelhaouzi/insura-sight-dashboard

🔗 https://hub.docker.com/u/suzannelhaouzi

## Render 
https://insura-sight.onrender.com/docs
## streamlit 
https://insurasight-assurance-iard.streamlit.app/


## Architecture

```
insura-sight/
│
├── data/
│   ├── raw/                  # Données synthétiques générées
│   │   ├── clients.csv           (5 000 clients)
│   │   ├── polices.csv           (7 200 contrats)
│   │   ├── sinistres.csv         (3 123 sinistres)
│   │   └── exposition_mensuelle.csv
│   ├── features/             # Table de modélisation
│   └── processed/            # Segments, scores fraude, prévisions
│
├── src/
│   ├── ingestion/
│   │   └── generate_data.py      # Génération dataset synthétique
│   └── features/
│       └── build_features.py     # Feature engineering
│
├── notebooks/
│   ├── 01_EDA.py                 # Analyse exploratoire (5 figures)
│   ├── 02_modeling.py            # Modèles freq/sévérité + MLflow
│   ├── 03_clustering_fraude.py   # K-Means + Isolation Forest
│   └── 04_forecast.py            # Prévisions Prophet (3 séries)
│
├── api/
│   └── main.py                   # FastAPI — 7 endpoints
│
├── dashboard/
│   └── app.py                    # Streamlit — 5 pages
│
├── models/                   # Modèles sérialisés (.pkl)
├── reports/figures/          # 10 figures générées
├── mlflow.db                 # Tracking MLflow (SQLite)
│
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.dashboard
└── requirements.txt
```

## Stack technique

| Composant | Outils |
|-----------|--------|
| Data generation | `numpy`, `pandas`, `Faker` |
| EDA | `matplotlib`, `seaborn` |
| Feature engineering | `pandas`, `scikit-learn` |
| Modélisation fréquence | `PoissonRegressor` (GLM), `XGBRegressor` (Poisson) |
| Modélisation sévérité | `GammaRegressor` (GLM), `XGBRegressor` (Gamma) |
| Clustering | `KMeans`, `UMAP`, `HDBSCAN` |
| Détection fraude | `IsolationForest` |
| Prévision | `Prophet` (3 séries temporelles) |
| Explicabilité | `SHAP` |
| Tracking ML | `MLflow` (SQLite backend) |
| API | `FastAPI` + `Pydantic` |
| Dashboard | `Streamlit` |
| Déploiement | `Docker` + `docker-compose` |

## Installation & Lancement

### 1. Cloner et installer les dépendances

```bash
git clone https://github.com/Suzann-el/insura-sight
cd insura-sight
pip install -r requirements.txt
```

### 2. Générer les données et entraîner les modèles

```bash
python src/ingestion/generate_data.py
python src/features/build_features.py
python notebooks/01_EDA.py
python notebooks/02_modeling.py
python notebooks/03_clustering_fraude.py
python notebooks/04_forecast.py
```

### 3. Lancer l'API FastAPI

```bash
uvicorn api.main:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs
```

### 4. Lancer le dashboard Streamlit

```bash
streamlit run dashboard/app.py
# → http://localhost:8501
```

### 5. Via Docker (API + Dashboard)

```bash
docker-compose up --build
# API       → http://localhost:8000
# Dashboard → http://localhost:8501
```

## KPIs du portefeuille (2023)

| Indicateur | Valeur |
|-----------|--------|
| Primes acquises | 7,68 M€ |
| Loss Ratio (S/P) | **65%** |
| Fréquence sinistre | 0.169 sin/police/an |
| Coût moyen | 5 119 € |
| Délai moyen traitement | 28 jours |
| Taux litigieux | 8,9% |

## Résultats des modèles

### Fréquence (E[N])

| Modèle | MAE | Gini |
|--------|-----|------|
| GLM Poisson | 0.286 | 0.145 |
| **XGBoost Poisson** | **0.274** | **0.180** |

### Sévérité (E[C\|N>0])

| Modèle | MAE (€) | RMSE (€) |
|--------|---------|----------|
| GLM Gamma | 4 869 | 8 241 |
| **XGBoost Gamma** | **4 539** | **8 598** |

### Prime pure = E[N] × E[C\|N>0]
- Moyenne : **561 €/police/an**
- P90 : 1 152 € | P99 : 2 984 €

## Segmentation clients (K-Means, K=4)

| Segment | Clients | Fréquence | Profil |
|---------|---------|-----------|--------|
| 🟢 Bon risque fidèle | 1 207 (24%) | 0.000 | Aucun sinistre |
| 🔵 Risque modéré | 2 982 (60%) | 0.243 | Cœur de portefeuille |
| 🔴 Très haut risque | 645 (13%) | 1.207 | Sinistres fréquents |
| 🟠 Risque élevé | 166 (3%) | 0.847 | Taux litigieux 71% |

## Détection de fraude (Isolation Forest)

- 157 dossiers suspects détectés (5% du portefeuille)
- Coût moyen suspect : **25 491 €** vs 4 155 € (×6)
- 118 dossiers en priorité haute (ML + règles métier)
- Gain potentiel estimé : **1,2 M€**

## API Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Health check |
| GET | `/info` | Version & modèles chargés |
| POST | `/predict/frequence` | E[N] pour une police |
| POST | `/predict/severite` | E[C\|N>0] pour une police |
| POST | `/predict/prime_pure` | E[N] × E[C\|N>0] + tarification |
| POST | `/predict/fraude` | Score anomalie Isolation Forest |
| GET | `/kpis` | KPIs portefeuille en temps réel |
| GET | `/segments` | Résumé des 4 segments clients |

---

*Projet réalisé dans le cadre d'un portfolio Data Science — Saoussan*
*GitHub : [github.com/Suzann-el](https://github.com/Suzann-el)*
