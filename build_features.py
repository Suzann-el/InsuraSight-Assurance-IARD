"""
InsuraSight — Feature Engineering
==================================
Construit la table analytique principale (une ligne par police) :
  - Features client (âge, ancienneté, score risque, sinistres passés)
  - Features contrat (produit, garantie, prime, ancienneté contrat)
  - Features géographiques (région encodée)
  - Target fréquence  : nb_sinistres (count, loi de Poisson)
  - Target sévérité   : montant_indemnise_moy (continu, loi Gamma)
  - Exposition        : nb_annees (offset pour GLM)

Usage :
    python src/features/build_features.py
    → data/features/table_modelisation.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

os.makedirs("data/features", exist_ok=True)

# ── Chargement ──────────────────────────────────────────────────────────────
print("📂 Chargement...")
df_clients   = pd.read_csv("data/raw/clients.csv")
df_polices   = pd.read_csv("data/raw/polices.csv")
df_sinistres = pd.read_csv("data/raw/sinistres.csv", parse_dates=["date_survenance"])
df_expo      = pd.read_csv("data/raw/exposition_mensuelle.csv")

ANNEE_REF = 2023   # année sur laquelle on construit les targets

# ── Exposition en années (offset GLM) ──────────────────────────────────────
expo_an = (
    df_expo[df_expo["mois"].str.startswith(str(ANNEE_REF))]
    .groupby("police_id")["prime_acquise"]
    .count()                          # nb de mois exposés
    .div(12)                          # → années
    .reset_index(name="exposition_annees")
)

# ── Agrégation sinistres N (targets) ───────────────────────────────────────
sin_ref = df_sinistres[df_sinistres["date_survenance"].dt.year == ANNEE_REF]

# TARGET 1 : fréquence (nb sinistres par police)
freq_target = (
    sin_ref.groupby("police_id")
    .agg(
        nb_sinistres        = ("sinistre_id", "count"),
        montant_total_indem = ("montant_indemnise", "sum"),
        montant_moy_indem   = ("montant_indemnise", "mean"),
        delai_moyen_decl    = ("delai_declaration_j", "mean"),
    )
    .reset_index()
)

# ── Sinistres N-1 (features comportementales) ──────────────────────────────
sin_nm1 = df_sinistres[df_sinistres["date_survenance"].dt.year == ANNEE_REF - 1]
hist_nm1 = (
    sin_nm1.groupby("police_id")
    .agg(
        nb_sin_nm1       = ("sinistre_id", "count"),
        montant_nm1      = ("montant_indemnise", "sum"),
        nb_litigieux_nm1 = ("statut", lambda x: (x == "Litigieux").sum()),
    )
    .reset_index()
)

# ── Construction de la table principale ────────────────────────────────────
print("🔧 Construction des features...")

# region existe dans polices ET clients → suffixes → on garde region_pol (contrat)
base = df_polices.merge(df_clients, on="client_id", suffixes=("_pol", "_cli"))
base = base.rename(columns={"region_pol": "region", "statut": "statut"})
base = base.merge(expo_an, on="police_id", how="left")
base = base.merge(freq_target, on="police_id", how="left")
base = base.merge(hist_nm1, on="police_id", how="left")

# Remplissage des NaN (polices sans sinistres)
base["nb_sinistres"]         = base["nb_sinistres"].fillna(0).astype(int)
base["montant_total_indem"]  = base["montant_total_indem"].fillna(0)
base["montant_moy_indem"]    = base["montant_moy_indem"].fillna(np.nan)   # garde NaN pour sévérité
base["nb_sin_nm1"]           = base["nb_sin_nm1"].fillna(0).astype(int)
base["montant_nm1"]          = base["montant_nm1"].fillna(0)
base["nb_litigieux_nm1"]     = base["nb_litigieux_nm1"].fillna(0).astype(int)
base["exposition_annees"]    = base["exposition_annees"].fillna(0)

# ── Feature engineering ────────────────────────────────────────────────────

# 1. Ancienneté du contrat à fin ANNEE_REF
base["date_souscription"] = pd.to_datetime(base["date_souscription"])
base["anciennete_contrat_an"] = (
    (pd.Timestamp(f"{ANNEE_REF}-12-31") - base["date_souscription"])
    .dt.days / 365
).clip(lower=0).round(2)

# 2. Ratio prime / benchmark marché par produit
prime_moy_produit = base.groupby("type_produit")["prime_annuelle"].transform("mean")
base["ratio_prime_marche"] = (base["prime_annuelle"] / prime_moy_produit).round(4)

# 3. Interaction âge × score risque
base["age_x_risque"] = (base["age"] * base["score_risque_init"]).round(4)

# 4. Log de la prime (feature pour le modèle sévérité)
base["log_prime"] = np.log1p(base["prime_annuelle"]).round(4)

# 5. Flag nouveau client (ancienneté < 1 an)
base["flag_nouveau_client"] = (base["anciennete_annees"] < 1).astype(int)

# 6. Flag récidiviste (≥ 2 sinistres N-1)
base["flag_recidiviste"] = (base["nb_sin_nm1"] >= 2).astype(int)

# 7. Taux de litigiosité N-1
base["taux_litigieux_nm1"] = np.where(
    base["nb_sin_nm1"] > 0,
    base["nb_litigieux_nm1"] / base["nb_sin_nm1"],
    0
).round(4)

# ── Encodages catégoriels ──────────────────────────────────────────────────

# One-hot encoding : type_produit, niveau_garantie
base = pd.get_dummies(base, columns=["type_produit", "niveau_garantie"], drop_first=False)

# Label encoding : région
le_region = LabelEncoder()
base["region_enc"] = le_region.fit_transform(base["region"])

# Label encoding : canal
le_canal = LabelEncoder()
base["canal_enc"] = le_canal.fit_transform(base["canal_souscription"])

# Label encoding : statut police
le_statut = LabelEncoder()
base["statut_enc"] = le_statut.fit_transform(base["statut"])

# ── Sélection et export ────────────────────────────────────────────────────

# Features finales
FEATURES_FREQ = [
    # Client
    "age", "anciennete_annees", "score_risque_init", "nb_sinistres_passes",
    "age_x_risque", "flag_nouveau_client",
    # Contrat
    "prime_annuelle", "log_prime", "ratio_prime_marche",
    "anciennete_contrat_an",
    # Historique
    "nb_sin_nm1", "montant_nm1", "flag_recidiviste", "taux_litigieux_nm1",
    # Encodages
    "region_enc", "canal_enc", "statut_enc",
    # One-hot produit
    "type_produit_Auto", "type_produit_Habitation", "type_produit_Santé",
    # One-hot garantie
    "niveau_garantie_RC seule", "niveau_garantie_Tiers étendu", "niveau_garantie_Tous risques",
    "niveau_garantie_Basique", "niveau_garantie_Confort", "niveau_garantie_Premium",
    "niveau_garantie_Essentiel", "niveau_garantie_Équilibré", "niveau_garantie_Complet",
]

TARGETS = ["nb_sinistres", "montant_moy_indem", "montant_total_indem"]
META    = ["police_id", "client_id", "exposition_annees"]

# Garder uniquement les colonnes disponibles (certains one-hot peuvent manquer)
features_dispo = [f for f in FEATURES_FREQ if f in base.columns]
cols_export = META + features_dispo + TARGETS

df_model = base[cols_export].copy()

# Filtrer les polices avec exposition > 0
df_model = df_model[df_model["exposition_annees"] > 0].reset_index(drop=True)

df_model.to_csv("data/features/table_modelisation.csv", index=False)

print(f"\n✅ Table de modélisation construite")
print(f"   Shape          : {df_model.shape}")
print(f"   Polices        : {len(df_model):,}")
print(f"   Features       : {len(features_dispo)}")
print(f"   % avec sinistre: {(df_model['nb_sinistres'] > 0).mean():.2%}")
print(f"   Target freq    : min={df_model['nb_sinistres'].min()}, "
      f"max={df_model['nb_sinistres'].max()}, "
      f"mean={df_model['nb_sinistres'].mean():.4f}")
print(f"   Target sévérité: mean={df_model['montant_moy_indem'].mean():.0f} €  "
      f"(sur {df_model['montant_moy_indem'].notna().sum()} polices sinistrées)")
print(f"\n   Fichier : data/features/table_modelisation.csv")
