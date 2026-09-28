"""
InsuraSight — API FastAPI
=========================
Endpoints :
  GET  /                    → health check
  GET  /info                → version, modèles chargés
  POST /predict/frequence   → E[N] = fréquence de sinistres prédite
  POST /predict/severite    → E[C|N>0] = coût moyen conditionnel
  POST /predict/prime_pure  → E[N] × E[C|N>0] = prime pure
  POST /predict/fraude      → score d'anomalie Isolation Forest
  GET  /kpis                → KPIs portefeuille (année de référence)
  GET  /segments            → résumé des segments clients

Lancement :
    uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime

# ── Chargement des modèles ──────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model(filename):
    path = os.path.join(BASE_DIR, "models", filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

print("⏳ Chargement des modèles...")
xgb_freq     = load_model("xgb_poisson_freq.pkl")
xgb_sev      = load_model("xgb_gamma_sev.pkl")
glm_freq     = load_model("glm_poisson_freq.pkl")
glm_sev      = load_model("glm_gamma_sev.pkl")
iso_forest   = load_model("isolation_forest_fraude.pkl")
kmeans_data  = load_model("kmeans_clients.pkl")
feature_cols = load_model("feature_cols.pkl")

MODELS_LOADED = {
    "xgb_frequence" : xgb_freq is not None,
    "xgb_severite"  : xgb_sev  is not None,
    "glm_frequence" : glm_freq  is not None,
    "glm_severite"  : glm_sev   is not None,
    "isolation_forest": iso_forest is not None,
    "kmeans"         : kmeans_data is not None,
}
print(f"   Modèles chargés : {sum(MODELS_LOADED.values())}/{len(MODELS_LOADED)}")

# ── Schémas Pydantic ────────────────────────────────────────────────────────

class PoliceInput(BaseModel):
    """Données d'une police pour le scoring de risque."""
    # Features client
    age                  : int   = Field(..., ge=18, le=100, example=38)
    anciennete_annees    : float = Field(..., ge=0,  le=30,  example=4.5)
    score_risque_init    : float = Field(..., ge=0,  le=1,   example=0.25)
    nb_sinistres_passes  : int   = Field(..., ge=0,  le=20,  example=1)

    # Features contrat
    prime_annuelle       : float = Field(..., gt=0,         example=1400.0)
    anciennete_contrat_an: float = Field(..., ge=0,  le=30, example=2.0)

    # Features historique N-1
    nb_sin_nm1           : int   = Field(0,   ge=0,  le=10)
    montant_nm1          : float = Field(0.0, ge=0)
    flag_recidiviste     : int   = Field(0,   ge=0,  le=1)
    taux_litigieux_nm1   : float = Field(0.0, ge=0,  le=1)

    # Encodages (valeurs entières correspondant au LabelEncoder)
    region_enc           : int   = Field(0,   ge=0,  le=9,   example=5)
    canal_enc            : int   = Field(0,   ge=0,  le=4,   example=0)
    statut_enc           : int   = Field(1,   ge=0,  le=2,   example=1)

    # Type de produit (one-hot)
    type_produit_Auto      : int = Field(1, ge=0, le=1)
    type_produit_Habitation: int = Field(0, ge=0, le=1)
    type_produit_Santé     : int = Field(0, ge=0, le=1)

    # Niveau de garantie (one-hot — laisser tous à 0 si non applicable)
    niveau_garantie_RC_seule      : int = Field(0, ge=0, le=1, alias="niveau_garantie_RC seule")
    niveau_garantie_Tiers_etendu  : int = Field(0, ge=0, le=1, alias="niveau_garantie_Tiers étendu")
    niveau_garantie_Tous_risques  : int = Field(1, ge=0, le=1, alias="niveau_garantie_Tous risques")
    niveau_garantie_Basique       : int = Field(0, ge=0, le=1)
    niveau_garantie_Confort       : int = Field(0, ge=0, le=1)
    niveau_garantie_Premium       : int = Field(0, ge=0, le=1)
    niveau_garantie_Essentiel     : int = Field(0, ge=0, le=1)
    niveau_garantie_Equilibre     : int = Field(0, ge=0, le=1, alias="niveau_garantie_Équilibré")
    niveau_garantie_Complet       : int = Field(0, ge=0, le=1)

    model_config = {"populate_by_name": True}


class SinistreInput(BaseModel):
    """Données d'un sinistre pour le scoring fraude."""
    montant_reclame     : float = Field(..., gt=0,  example=12500.0)
    montant_indemnise   : float = Field(..., gt=0,  example=9800.0)
    delai_declaration_j : int   = Field(..., ge=0,  le=365, example=1)
    nb_jours_traitement : int   = Field(..., ge=0,  le=365, example=7)
    prime_annuelle      : float = Field(..., gt=0,  example=1400.0)
    age                 : int   = Field(..., ge=18, le=100, example=38)
    anciennete_annees   : float = Field(..., ge=0,  le=30,  example=4.5)
    nb_sinistres_passes : int   = Field(..., ge=0,  le=20,  example=3)
    score_risque_init   : float = Field(..., ge=0,  le=1,   example=0.6)


# ── Helpers ─────────────────────────────────────────────────────────────────

def police_to_features(p: PoliceInput) -> np.ndarray:
    """Convertit un PoliceInput en vecteur de features aligné sur FEATURE_COLS."""
    raw = {
        "age"                        : p.age,
        "anciennete_annees"          : p.anciennete_annees,
        "score_risque_init"          : p.score_risque_init,
        "nb_sinistres_passes"        : p.nb_sinistres_passes,
        "age_x_risque"               : p.age * p.score_risque_init,
        "flag_nouveau_client"        : int(p.anciennete_annees < 1),
        "prime_annuelle"             : p.prime_annuelle,
        "log_prime"                  : np.log1p(p.prime_annuelle),
        "ratio_prime_marche"         : 1.0,   # valeur neutre si pas de benchmark disponible
        "anciennete_contrat_an"      : p.anciennete_contrat_an,
        "nb_sin_nm1"                 : p.nb_sin_nm1,
        "montant_nm1"                : p.montant_nm1,
        "flag_recidiviste"           : p.flag_recidiviste,
        "taux_litigieux_nm1"         : p.taux_litigieux_nm1,
        "region_enc"                 : p.region_enc,
        "canal_enc"                  : p.canal_enc,
        "statut_enc"                 : p.statut_enc,
        "type_produit_Auto"          : p.type_produit_Auto,
        "type_produit_Habitation"    : p.type_produit_Habitation,
        "type_produit_Santé"         : p.type_produit_Santé,
        "niveau_garantie_RC seule"   : p.niveau_garantie_RC_seule,
        "niveau_garantie_Tiers étendu": p.niveau_garantie_Tiers_etendu,
        "niveau_garantie_Tous risques": p.niveau_garantie_Tous_risques,
        "niveau_garantie_Basique"    : p.niveau_garantie_Basique,
        "niveau_garantie_Confort"    : p.niveau_garantie_Confort,
        "niveau_garantie_Premium"    : p.niveau_garantie_Premium,
        "niveau_garantie_Essentiel"  : p.niveau_garantie_Essentiel,
        "niveau_garantie_Équilibré"  : p.niveau_garantie_Equilibre,
        "niveau_garantie_Complet"    : p.niveau_garantie_Complet,
    }
    if feature_cols:
        arr = np.array([raw.get(col, 0) for col in feature_cols], dtype=float)
    else:
        arr = np.array(list(raw.values()), dtype=float)
    return arr.reshape(1, -1)


def sinistre_to_features(s: SinistreInput) -> np.ndarray:
    """Convertit un SinistreInput en vecteur pour l'Isolation Forest."""
    if iso_forest is None:
        raise HTTPException(status_code=503, detail="Modèle fraude non chargé")
    feats = iso_forest["features"]
    ratio_ri = s.montant_reclame / max(s.montant_indemnise, 1)
    ratio_sp = s.montant_reclame / max(s.prime_annuelle, 1)
    raw = {
        "montant_reclame"     : s.montant_reclame,
        "montant_indemnise"   : s.montant_indemnise,
        "ratio_reclame_indem" : ratio_ri,
        "ratio_sin_prime"     : ratio_sp,
        "delai_declaration_j" : s.delai_declaration_j,
        "nb_jours_traitement" : s.nb_jours_traitement,
        "age"                 : s.age,
        "anciennete_annees"   : s.anciennete_annees,
        "nb_sinistres_passes" : s.nb_sinistres_passes,
        "score_risque_init"   : s.score_risque_init,
    }
    arr = np.array([raw.get(f, 0) for f in feats], dtype=float).reshape(1, -1)
    scaler = iso_forest["scaler"]
    return scaler.transform(arr)


# ── App FastAPI ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "InsuraSight API",
    description = "API de scoring actuariel — fréquence, sévérité, prime pure, fraude",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service"   : "InsuraSight API",
        "status"    : "ok",
        "timestamp" : datetime.now().isoformat(),
    }


@app.get("/info", tags=["Health"])
def info():
    return {
        "version"       : "1.0.0",
        "modeles_charges": MODELS_LOADED,
        "features"      : feature_cols if feature_cols else [],
        "endpoints"     : [
            "POST /predict/frequence",
            "POST /predict/severite",
            "POST /predict/prime_pure",
            "POST /predict/fraude",
            "GET  /kpis",
            "GET  /segments",
        ],
    }


@app.post("/predict/frequence", tags=["Scoring"])
def predict_frequence(police: PoliceInput, modele: Literal["xgboost", "glm"] = "xgboost"):
    """
    Prédit la fréquence annuelle de sinistres (E[N]) pour une police.
    Utilise XGBoost Poisson (défaut) ou GLM Poisson.
    """
    X = police_to_features(police)

    if modele == "xgboost":
        if xgb_freq is None:
            raise HTTPException(503, "Modèle XGBoost fréquence non chargé")
        freq = float(xgb_freq.predict(X)[0])
    else:
        if glm_freq is None:
            raise HTTPException(503, "Modèle GLM fréquence non chargé")
        freq = float(glm_freq.predict(X)[0])

    # Interprétation métier
    if freq < 0.05:
        niveau = "Très faible"
    elif freq < 0.15:
        niveau = "Faible"
    elif freq < 0.30:
        niveau = "Modéré"
    elif freq < 0.50:
        niveau = "Élevé"
    else:
        niveau = "Très élevé"

    return {
        "frequence_pred"    : round(freq, 6),
        "nb_sinistres_par_an": round(freq, 3),
        "niveau_risque"     : niveau,
        "modele_utilise"    : modele,
    }


@app.post("/predict/severite", tags=["Scoring"])
def predict_severite(police: PoliceInput, modele: Literal["xgboost", "glm"] = "xgboost"):
    """
    Prédit le coût moyen conditionnel d'un sinistre E[C|N>0].
    """
    X = police_to_features(police)

    if modele == "xgboost":
        if xgb_sev is None:
            raise HTTPException(503, "Modèle XGBoost sévérité non chargé")
        sev = float(xgb_sev.predict(X)[0])
    else:
        if glm_sev is None:
            raise HTTPException(503, "Modèle GLM sévérité non chargé")
        sev = float(glm_sev.predict(X)[0])

    return {
        "severite_pred_eur" : round(max(sev, 0), 2),
        "modele_utilise"    : modele,
    }


@app.post("/predict/prime_pure", tags=["Scoring"])
def predict_prime_pure(police: PoliceInput, modele: Literal["xgboost", "glm"] = "xgboost"):
    """
    Calcule la prime pure prédite : E[N] × E[C|N>0].
    Retourne aussi le ratio prime pure / prime actuelle (sous/sur-tarification).
    """
    X = police_to_features(police)

    if modele == "xgboost":
        if xgb_freq is None or xgb_sev is None:
            raise HTTPException(503, "Modèles XGBoost non chargés")
        freq = float(xgb_freq.predict(X)[0])
        sev  = float(xgb_sev.predict(X)[0])
    else:
        if glm_freq is None or glm_sev is None:
            raise HTTPException(503, "Modèles GLM non chargés")
        freq = float(glm_freq.predict(X)[0])
        sev  = float(glm_sev.predict(X)[0])

    prime_pure = max(freq * sev, 0)
    ratio_tarif = prime_pure / max(police.prime_annuelle, 1)

    if ratio_tarif < 0.5:
        tarification = "Fortement sur-tarifée"
    elif ratio_tarif < 0.8:
        tarification = "Sur-tarifée"
    elif ratio_tarif < 1.2:
        tarification = "Bien calibrée"
    elif ratio_tarif < 1.5:
        tarification = "Sous-tarifée"
    else:
        tarification = "Fortement sous-tarifée"

    return {
        "frequence_pred"      : round(freq, 6),
        "severite_pred_eur"   : round(max(sev, 0), 2),
        "prime_pure_pred_eur" : round(prime_pure, 2),
        "prime_actuelle_eur"  : police.prime_annuelle,
        "ratio_pure_actuelle" : round(ratio_tarif, 4),
        "tarification"        : tarification,
        "modele_utilise"      : modele,
    }


@app.post("/predict/fraude", tags=["Scoring"])
def predict_fraude(sinistre: SinistreInput):
    """
    Calcule le score d'anomalie d'un sinistre (Isolation Forest).
    Score > seuil → dossier à investiguer.
    """
    if iso_forest is None:
        raise HTTPException(503, "Modèle Isolation Forest non chargé")

    X_scaled = sinistre_to_features(sinistre)
    model    = iso_forest["model"]

    anomaly_score  = float(-model.score_samples(X_scaled)[0])
    anomaly_label  = int(model.predict(X_scaled)[0])   # -1 = suspect, 1 = normal
    is_suspect     = anomaly_label == -1

    # Règles métier complémentaires
    rules_score = 0
    rules_triggered = []
    if sinistre.montant_reclame > 30000:
        rules_score += 3; rules_triggered.append("Montant très élevé (>30k€)")
    if sinistre.delai_declaration_j <= 1:
        rules_score += 2; rules_triggered.append("Déclaration immédiate (≤1j)")
    if sinistre.nb_sinistres_passes >= 3:
        rules_score += 2; rules_triggered.append("Récidiviste (≥3 sinistres passés)")
    if sinistre.montant_reclame / max(sinistre.montant_indemnise, 1) > 1.5:
        rules_score += 3; rules_triggered.append("Ratio réclamé/indemnisé élevé (>1.5)")

    priorite = "HAUTE" if is_suspect and rules_score >= 3 else \
               "MOYENNE" if is_suspect or rules_score >= 3 else "BASSE"

    return {
        "anomaly_score"    : round(anomaly_score, 4),
        "is_suspect"       : is_suspect,
        "priorite"         : priorite,
        "score_regles"     : rules_score,
        "regles_declenchees": rules_triggered,
        "recommandation"   : (
            "⚠️ Dossier à investiguer en priorité" if priorite == "HAUTE"
            else "🔍 Surveillance recommandée" if priorite == "MOYENNE"
            else "✅ Dossier dans la norme"
        ),
    }


@app.get("/kpis", tags=["Reporting"])
def get_kpis():
    """
    Retourne les KPIs clés du portefeuille (année 2023).
    """
    try:
        df_sin  = pd.read_csv(os.path.join(BASE_DIR, "data/raw/sinistres.csv"),
                               parse_dates=["date_survenance"])
        df_pol  = pd.read_csv(os.path.join(BASE_DIR, "data/raw/polices.csv"))
        df_expo = pd.read_csv(os.path.join(BASE_DIR, "data/raw/exposition_mensuelle.csv"))
        df_expo["mois_dt"] = pd.to_datetime(df_expo["mois"])

        df_2023  = df_sin[df_sin["date_survenance"].dt.year == 2023]
        exp_2023 = df_expo[df_expo["mois_dt"].dt.year == 2023]

        primes   = exp_2023["prime_acquise"].sum()
        sin_pays = df_2023["montant_indemnise"].sum()
        freq     = len(df_2023) / len(df_pol[df_pol["statut"] == "Active"])
        cout_moy = df_2023["montant_indemnise"].mean()
        loss_ratio = sin_pays / primes
        delai_moy  = df_2023["nb_jours_traitement"].mean()
        taux_ouvert    = (df_2023["statut"] == "Ouvert").mean()
        taux_litigieux = (df_2023["statut"] == "Litigieux").mean()
        nb_fraudes     = df_2023["fraude_potentielle"].sum()

        return {
            "annee_reference"      : 2023,
            "primes_acquises_eur"  : float(round(primes, 0)),
            "sinistres_payes_eur"  : float(round(sin_pays, 0)),
            "loss_ratio"           : float(round(loss_ratio, 4)),
            "frequence_sinistre"   : float(round(freq, 4)),
            "cout_moyen_eur"       : float(round(cout_moy, 0)),
            "delai_moyen_jours"    : float(round(delai_moy, 1)),
            "taux_dossiers_ouverts": float(round(taux_ouvert, 4)),
            "taux_litigieux"       : float(round(taux_litigieux, 4)),
            "nb_fraudes_detectees" : int(nb_fraudes),
            "nb_polices_actives"   : int(len(df_pol[df_pol["statut"] == "Active"])),
            "nb_sinistres_total"   : int(len(df_2023)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/segments", tags=["Reporting"])
def get_segments():
    """
    Retourne le résumé des segments clients K-Means.
    """
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "data/processed/profil_clients_segments.csv"))
        seg = (df.groupby("segment")
               .agg(
                   nb_clients          = ("client_id", "count"),
                   age_moyen           = ("age", "mean"),
                   score_risque_moyen  = ("score_risque_init", "mean"),
                   freq_sinistre_moy   = ("freq_sin_par_police", "mean"),
                   montant_sinistre_moy= ("montant_moy", "mean"),
                   ratio_SP_moyen      = ("ratio_sinistre_prime", "mean"),
                   prime_moyenne       = ("prime_moy", "mean"),
               )
               .round(3)
               .reset_index()
               .to_dict(orient="records"))
        return {"nb_segments": len(seg), "segments": seg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
