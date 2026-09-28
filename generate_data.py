"""
InsuraSight — Génération du dataset synthétique
================================================
Simule un portefeuille d'assurance IARD (auto, habitation, santé)
avec des sinistres réalistes, des profils clients et des tendances temporelles.

Usage :
    python src/ingestion/generate_data.py
    → Écrit 4 fichiers CSV dans data/raw/
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import random
import os

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N_CLIENTS   = 5_000
N_POLICES   = 7_200   # certains clients ont plusieurs contrats
START_DATE  = date(2021, 1, 1)
END_DATE    = date(2024, 12, 31)
OUTPUT_DIR  = "data/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def weighted_choice(choices: list, weights: list) -> str:
    return random.choices(choices, weights=weights, k=1)[0]


# ─────────────────────────────────────────
# 1. TABLE CLIENTS
# ─────────────────────────────────────────

print("⏳ Génération des clients...")

regions = [
    "Île-de-France", "Auvergne-Rhône-Alpes", "Nouvelle-Aquitaine",
    "Occitanie", "Hauts-de-France", "Provence-Alpes-Côte d'Azur",
    "Bretagne", "Normandie", "Grand Est", "Centre-Val de Loire"
]

canaux = ["Web", "Courtier", "Agent", "Téléphone", "Partenaire banque"]

clients = []
for i in range(N_CLIENTS):
    age = int(np.clip(np.random.normal(42, 14), 18, 85))
    anciennete = int(np.clip(np.random.exponential(4), 0, 20))   # années
    nb_sin_passes = np.random.poisson(0.6)                        # sinistres antérieurs
    # score de risque initial : corrélé à l'âge, l'ancienneté et les sinistres passés
    score_base = (
        0.3 * min(nb_sin_passes / 3, 1)
        + 0.2 * max((25 - age) / 25, 0)            # jeunes conducteurs plus risqués
        + 0.1 * max((1 - anciennete / 10), 0)       # nouveaux clients plus risqués
        + np.random.normal(0, 0.1)
    )
    score_risque = float(np.clip(score_base, 0.01, 0.99))

    clients.append({
        "client_id"         : f"CLT{i+1:05d}",
        "age"               : age,
        "region"            : random.choice(regions),
        "anciennete_annees" : anciennete,
        "nb_sinistres_passes": nb_sin_passes,
        "canal_souscription": weighted_choice(canaux, [35, 25, 20, 12, 8]),
        "score_risque_init" : round(score_risque, 4),
        "date_creation"     : random_date(START_DATE - timedelta(days=365*5), START_DATE).isoformat(),
    })

df_clients = pd.DataFrame(clients)
df_clients.to_csv(f"{OUTPUT_DIR}/clients.csv", index=False)
print(f"   ✅ {len(df_clients)} clients générés")


# ─────────────────────────────────────────
# 2. TABLE POLICES
# ─────────────────────────────────────────

print("⏳ Génération des polices...")

produits = ["Auto", "Habitation", "Santé"]
produit_weights = [45, 35, 20]

garanties_map = {
    "Auto"       : ["RC seule", "Tiers étendu", "Tous risques"],
    "Habitation" : ["Basique", "Confort", "Premium"],
    "Santé"      : ["Essentiel", "Équilibré", "Complet"],
}

prime_params = {
    # (moyenne, std) en euros/an — calibrées pour un Loss Ratio cible ~72%
    # Formule : prime = coût_moyen_type * freq_type / LR_cible
    "Auto"       : {"RC seule": (900, 120),  "Tiers étendu": (1400, 200), "Tous risques": (2200, 350)},
    "Habitation" : {"Basique": (1200, 150),  "Confort": (1800, 250),      "Premium": (2800, 400)},
    "Santé"      : {"Essentiel": (600, 100), "Équilibré": (980, 150),     "Complet": (1500, 250)},
}

statuts_police = ["Active", "Active", "Active", "Active", "Résiliée", "Suspendue"]

polices = []
client_ids = df_clients["client_id"].tolist()

for i in range(N_POLICES):
    client_id   = random.choice(client_ids)
    produit     = weighted_choice(produits, produit_weights)
    garanties   = garanties_map[produit]
    garantie    = weighted_choice(garanties, [40, 35, 25])
    mu, sigma   = prime_params[produit][garantie]

    # récupérer le score risque du client pour ajuster la prime
    score = df_clients.loc[df_clients["client_id"] == client_id, "score_risque_init"].values[0]
    prime = max(100, int(np.random.normal(mu, sigma) * (1 + 0.3 * score)))

    date_souscription = random_date(START_DATE, END_DATE - timedelta(days=180))
    statut = weighted_choice(statuts_police, [1, 1, 1, 1, 0.8, 0.2])

    polices.append({
        "police_id"        : f"POL{i+1:06d}",
        "client_id"        : client_id,
        "type_produit"     : produit,
        "niveau_garantie"  : garantie,
        "date_souscription": date_souscription.isoformat(),
        "prime_annuelle"   : prime,
        "statut"           : statut,
        "region"           : df_clients.loc[df_clients["client_id"] == client_id, "region"].values[0],
    })

df_polices = pd.DataFrame(polices)
df_polices.to_csv(f"{OUTPUT_DIR}/polices.csv", index=False)
print(f"   ✅ {len(df_polices)} polices générées")


# ─────────────────────────────────────────
# 3. TABLE SINISTRES
# ─────────────────────────────────────────

print("⏳ Génération des sinistres...")

types_sinistre = {
    "Auto"       : ["Accident matériel", "Accident corporel", "Vol", "Bris de glace", "Incendie"],
    "Habitation" : ["Dégât des eaux", "Incendie", "Vol/Cambriolage", "Catastrophe naturelle", "Bris de glace"],
    "Santé"      : ["Hospitalisation", "Soins courants", "Optique", "Dentaire", "Maternité"],
}

type_weights = {
    "Auto"       : [45, 15, 15, 20, 5],
    "Habitation" : [40, 15, 20, 15, 10],
    "Santé"      : [10, 35, 20, 25, 10],
}

# Coûts moyens par type de sinistre (montant_reclame)
cout_params = {
    "Accident matériel"   : (3500, 2000),
    "Accident corporel"   : (18000, 12000),
    "Vol"                 : (6000, 3000),
    "Bris de glace"       : (450, 150),
    "Incendie"            : (25000, 15000),
    "Dégât des eaux"      : (4500, 2500),
    "Vol/Cambriolage"     : (7000, 4000),
    "Catastrophe naturelle": (15000, 8000),
    "Hospitalisation"     : (5000, 3000),
    "Soins courants"      : (250, 100),
    "Optique"             : (350, 120),
    "Dentaire"            : (800, 400),
    "Maternité"           : (1200, 500),
}

statuts_sinistre = ["Fermé", "Fermé", "Fermé", "Ouvert", "Litigieux"]

sinistres = []
sin_id = 1

for _, pol in df_polices.iterrows():
    produit     = pol["type_produit"]
    score       = df_clients.loc[df_clients["client_id"] == pol["client_id"], "score_risque_init"].values[0]
    date_souscr = date.fromisoformat(pol["date_souscription"])

    # Fréquence annuelle de sinistres : loi de Poisson, modulée par score risque
    # Calibré pour Loss Ratio cible ~70% sur portefeuille IARD
    lambda_sin = 0.13 + 0.35 * score
    if pol["statut"] == "Résiliée":
        lambda_sin *= 1.3   # biais de sélection : les mauvais risques résili

    nb_sin = np.random.poisson(lambda_sin * max(1, (END_DATE - date_souscr).days / 365))

    for _ in range(nb_sin):
        type_sin  = weighted_choice(types_sinistre[produit], type_weights[produit])
        mu, sigma = cout_params[type_sin]
        montant_reclame = max(50, int(np.random.lognormal(np.log(mu), 0.5)))

        # taux d'indemnisation selon type (franchise, plafond, etc.)
        taux_indem = np.random.uniform(0.55, 0.95)
        montant_indemnise = int(montant_reclame * taux_indem)

        date_surv  = random_date(date_souscr, END_DATE)
        delai_decl = random.randint(0, 30)              # délai déclaration
        date_decl  = date_surv + timedelta(days=delai_decl)

        statut = weighted_choice(statuts_sinistre, [3, 3, 3, 2, 1])
        if date_decl > END_DATE - timedelta(days=90):
            statut = "Ouvert"   # sinistres récents encore ouverts

        nb_jours_trait = (
            random.randint(5, 30)   if statut == "Fermé"
            else random.randint(30, 180) if statut == "Litigieux"
            else random.randint(1, 60)
        )

        # flag fraude potentielle (rare)
        fraude_potentielle = (
            montant_reclame > 2 * mu and delai_decl < 3 and
            df_clients.loc[df_clients["client_id"] == pol["client_id"], "nb_sinistres_passes"].values[0] > 2
        )

        sinistres.append({
            "sinistre_id"       : f"SIN{sin_id:07d}",
            "police_id"         : pol["police_id"],
            "client_id"         : pol["client_id"],
            "type_produit"      : produit,
            "type_sinistre"     : type_sin,
            "date_survenance"   : date_surv.isoformat(),
            "date_declaration"  : min(date_decl, END_DATE).isoformat(),
            "montant_reclame"   : montant_reclame,
            "montant_indemnise" : montant_indemnise,
            "statut"            : statut,
            "nb_jours_traitement": nb_jours_trait,
            "delai_declaration_j": delai_decl,
            "fraude_potentielle": int(fraude_potentielle),
        })
        sin_id += 1

df_sinistres = pd.DataFrame(sinistres)
df_sinistres.to_csv(f"{OUTPUT_DIR}/sinistres.csv", index=False)
print(f"   ✅ {len(df_sinistres)} sinistres générés")


# ─────────────────────────────────────────
# 4. TABLE PRIMES ACQUISES (expositions)
# ─────────────────────────────────────────

print("⏳ Génération de l'exposition (primes acquises par mois)...")

# Pour chaque police active, on calcule la prime acquise par mois (pro rata temporis)
expositions = []
months = pd.date_range(start=START_DATE.isoformat(), end=END_DATE.isoformat(), freq="MS")

for _, pol in df_polices.iterrows():
    date_souscr = pd.Timestamp(pol["date_souscription"])
    prime_mensuelle = round(pol["prime_annuelle"] / 12, 2)

    for m in months:
        if m >= date_souscr and m <= pd.Timestamp(END_DATE):
            expositions.append({
                "police_id"       : pol["police_id"],
                "client_id"       : pol["client_id"],
                "type_produit"    : pol["type_produit"],
                "region"          : pol["region"],
                "mois"            : m.strftime("%Y-%m"),
                "prime_acquise"   : prime_mensuelle,
                "statut_police"   : pol["statut"],
            })

df_expo = pd.DataFrame(expositions)
df_expo.to_csv(f"{OUTPUT_DIR}/exposition_mensuelle.csv", index=False)
print(f"   ✅ {len(df_expo)} lignes d'exposition générées")


# ─────────────────────────────────────────
# RÉSUMÉ
# ─────────────────────────────────────────

print("\n" + "="*50)
print("📦 DATASET GÉNÉRÉ — RÉSUMÉ")
print("="*50)
print(f"  Clients          : {len(df_clients):>8,}")
print(f"  Polices          : {len(df_polices):>8,}")
print(f"  Sinistres        : {len(df_sinistres):>8,}")
print(f"  Lignes exposition: {len(df_expo):>8,}")
print(f"\n  Loss Ratio estimé (global) : "
      f"{df_sinistres['montant_indemnise'].sum() / df_polices['prime_annuelle'].sum():.2%}")
print(f"  Coût moyen sinistre        : {df_sinistres['montant_indemnise'].mean():>8,.0f} €")
print(f"  Fréquence sinistre         : {len(df_sinistres)/len(df_polices):.3f} sin/police")
print(f"  Sinistres suspects fraude  : {df_sinistres['fraude_potentielle'].sum():>8,}")
print(f"\n  Fichiers sauvegardés dans  : {OUTPUT_DIR}/")
