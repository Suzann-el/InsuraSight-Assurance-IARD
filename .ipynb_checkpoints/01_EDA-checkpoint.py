"""
InsuraSight — Notebook 01 : Exploratory Data Analysis
======================================================
Analyse complète du portefeuille :
  - Profil du portefeuille (clients, polices)
  - Analyse des sinistres (fréquence, sévérité, temporalité)
  - KPIs sinistralité (S/P, Loss Ratio, Combined Ratio)
  - Corrélations et axes de risque

Exécution : python notebooks/01_EDA.py
Les figures sont sauvegardées dans reports/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("reports/figures", exist_ok=True)

# ── Palette cohérente avec l'identité InsuraSight ──────────────────────────
COLORS = {
    "primary"   : "#1B4F72",   # bleu marine
    "secondary" : "#2E86C1",   # bleu moyen
    "accent"    : "#E74C3C",   # rouge sinistre
    "success"   : "#27AE60",   # vert OK
    "warn"      : "#F39C12",   # orange alerte
    "light"     : "#D6EAF8",   # fond clair
    "neutral"   : "#5D6D7E",   # gris
}
PALETTE3 = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]

plt.rcParams.update({
    "figure.dpi"       : 120,
    "figure.facecolor" : "white",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.titlesize"   : 13,
    "axes.titleweight" : "bold",
    "axes.labelsize"   : 11,
    "font.family"      : "DejaVu Sans",
})

# ─────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────

print("📂 Chargement des données...")
df_clients  = pd.read_csv("data/raw/clients.csv")
df_polices  = pd.read_csv("data/raw/polices.csv")
df_sinistres = pd.read_csv("data/raw/sinistres.csv", parse_dates=["date_survenance", "date_declaration"])
df_expo     = pd.read_csv("data/raw/exposition_mensuelle.csv")

df_expo["mois_dt"] = pd.to_datetime(df_expo["mois"])
df_sinistres["annee"] = df_sinistres["date_survenance"].dt.year
df_sinistres["mois_dt"] = df_sinistres["date_survenance"].dt.to_period("M").dt.to_timestamp()

print(f"   Clients   : {len(df_clients):,}")
print(f"   Polices   : {len(df_polices):,}")
print(f"   Sinistres : {len(df_sinistres):,}")
print(f"   Expo      : {len(df_expo):,} lignes\n")


# ─────────────────────────────────────────
# 0. RÉSUMÉ STATISTIQUE
# ─────────────────────────────────────────

print("=" * 55)
print("  RÉSUMÉ STATISTIQUE — SINISTRES")
print("=" * 55)
desc = df_sinistres[["montant_reclame", "montant_indemnise",
                      "nb_jours_traitement", "delai_declaration_j"]].describe().round(0)
print(desc.to_string())
print()

# Distribution des statuts
print("Statuts sinistres :")
print(df_sinistres["statut"].value_counts().to_string())
print()

# ─────────────────────────────────────────
# FIGURE 1 — PROFIL PORTEFEUILLE
# ─────────────────────────────────────────

print("📊 Figure 1 — Profil portefeuille...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("InsuraSight — Profil du portefeuille", fontsize=15, fontweight="bold", y=1.01)

# 1a — Répartition des polices par produit
ax = axes[0, 0]
vc = df_polices["type_produit"].value_counts()
wedges, texts, autotexts = ax.pie(
    vc.values, labels=vc.index, autopct="%1.1f%%",
    colors=PALETTE3, startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
for t in autotexts: t.set_fontsize(10)
ax.set_title("Répartition des polices par produit")

# 1b — Distribution des âges des clients
ax = axes[0, 1]
ax.hist(df_clients["age"], bins=30, color=COLORS["secondary"],
        edgecolor="white", linewidth=0.5)
ax.axvline(df_clients["age"].mean(), color=COLORS["accent"], linestyle="--", linewidth=1.5,
           label=f'Moyenne : {df_clients["age"].mean():.0f} ans')
ax.set_title("Distribution des âges clients")
ax.set_xlabel("Âge")
ax.set_ylabel("Nombre de clients")
ax.legend(fontsize=9)

# 1c — Canal de souscription
ax = axes[0, 2]
canal = df_clients["canal_souscription"].value_counts()
bars = ax.barh(canal.index, canal.values, color=COLORS["primary"], edgecolor="white")
for bar, val in zip(bars, canal.values):
    ax.text(val + 20, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=9)
ax.set_title("Canal de souscription")
ax.set_xlabel("Nombre de clients")

# 1d — Prime annuelle par produit (boxplot)
ax = axes[1, 0]
produits = df_polices["type_produit"].unique()
data_bp = [df_polices[df_polices["type_produit"] == p]["prime_annuelle"].values for p in produits]
bp = ax.boxplot(data_bp, patch_artist=True, labels=produits,
                medianprops={"color": "white", "linewidth": 2})
for patch, color in zip(bp["boxes"], PALETTE3):
    patch.set_facecolor(color)
ax.set_title("Prime annuelle par produit (€)")
ax.set_ylabel("Prime (€)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# 1e — Ancienneté des clients
ax = axes[1, 1]
ax.hist(df_clients["anciennete_annees"], bins=20, color=COLORS["warn"],
        edgecolor="white", linewidth=0.5)
ax.set_title("Ancienneté des clients (années)")
ax.set_xlabel("Ancienneté")
ax.set_ylabel("Nombre de clients")

# 1f — Score de risque initial
ax = axes[1, 2]
ax.hist(df_clients["score_risque_init"], bins=40, color=COLORS["accent"],
        edgecolor="white", linewidth=0.5, alpha=0.85)
ax.set_title("Distribution du score de risque initial")
ax.set_xlabel("Score de risque [0–1]")
ax.set_ylabel("Nombre de clients")

plt.tight_layout()
plt.savefig("reports/figures/01_profil_portefeuille.png", bbox_inches="tight")
plt.close()
print("   ✅ Sauvegardé : 01_profil_portefeuille.png")


# ─────────────────────────────────────────
# FIGURE 2 — ANALYSE SINISTRES
# ─────────────────────────────────────────

print("📊 Figure 2 — Analyse des sinistres...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("InsuraSight — Analyse des sinistres", fontsize=15, fontweight="bold", y=1.01)

# 2a — Distribution montant indemnisé (log scale)
ax = axes[0, 0]
log_vals = np.log1p(df_sinistres["montant_indemnise"])
ax.hist(log_vals, bins=40, color=COLORS["secondary"], edgecolor="white", linewidth=0.5)
ax.set_title("Distribution montant indemnisé (log)")
ax.set_xlabel("log(montant + 1)")
ax.set_ylabel("Fréquence")

# 2b — Coût moyen par type de sinistre
ax = axes[0, 1]
cout_type = (df_sinistres.groupby("type_sinistre")["montant_indemnise"]
             .mean().sort_values(ascending=True))
bars = ax.barh(cout_type.index, cout_type.values, color=COLORS["primary"], edgecolor="white")
for bar, val in zip(bars, cout_type.values):
    ax.text(val + 100, bar.get_y() + bar.get_height()/2,
            f"{val:,.0f} €", va="center", fontsize=8)
ax.set_title("Coût moyen indemnisé par type")
ax.set_xlabel("Montant moyen (€)")

# 2c — Nombre de sinistres par type
ax = axes[0, 2]
nb_type = df_sinistres["type_sinistre"].value_counts()
ax.bar(range(len(nb_type)), nb_type.values, color=COLORS["secondary"], edgecolor="white")
ax.set_xticks(range(len(nb_type)))
ax.set_xticklabels(nb_type.index, rotation=45, ha="right", fontsize=8)
ax.set_title("Volume sinistres par type")
ax.set_ylabel("Nombre de sinistres")

# 2d — Statut des sinistres
ax = axes[1, 0]
statut = df_sinistres["statut"].value_counts()
colors_s = [COLORS["success"], COLORS["warn"], COLORS["accent"]]
ax.pie(statut.values, labels=statut.index, autopct="%1.1f%%",
       colors=colors_s[:len(statut)], startangle=90,
       wedgeprops={"edgecolor": "white", "linewidth": 2})
ax.set_title("Statut des dossiers sinistres")

# 2e — Délai de traitement par statut (violinplot)
ax = axes[1, 1]
statuts_order = ["Fermé", "Ouvert", "Litigieux"]
data_violin = [df_sinistres[df_sinistres["statut"] == s]["nb_jours_traitement"].values
               for s in statuts_order if s in df_sinistres["statut"].values]
labels_v = [s for s in statuts_order if s in df_sinistres["statut"].values]
vp = ax.violinplot(data_violin, positions=range(len(labels_v)), showmedians=True)
for body, color in zip(vp["bodies"], [COLORS["success"], COLORS["warn"], COLORS["accent"]]):
    body.set_facecolor(color)
    body.set_alpha(0.7)
ax.set_xticks(range(len(labels_v)))
ax.set_xticklabels(labels_v)
ax.set_title("Délai de traitement par statut (jours)")
ax.set_ylabel("Jours")

# 2f — Taux de réclamation vs indemnisation
ax = axes[1, 2]
sample = df_sinistres.sample(min(500, len(df_sinistres)), random_state=42)
sc = ax.scatter(sample["montant_reclame"], sample["montant_indemnise"],
                alpha=0.4, c=COLORS["secondary"], edgecolors="none", s=20)
max_val = max(sample["montant_reclame"].max(), sample["montant_indemnise"].max())
ax.plot([0, max_val], [0, max_val], "--", color=COLORS["accent"],
        linewidth=1, label="Réclamé = Indemnisé")
ax.set_title("Montant réclamé vs indemnisé (€)")
ax.set_xlabel("Montant réclamé")
ax.set_ylabel("Montant indemnisé")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

plt.tight_layout()
plt.savefig("reports/figures/02_analyse_sinistres.png", bbox_inches="tight")
plt.close()
print("   ✅ Sauvegardé : 02_analyse_sinistres.png")


# ─────────────────────────────────────────
# FIGURE 3 — KPIs SINISTRALITÉ
# ─────────────────────────────────────────

print("📊 Figure 3 — KPIs sinistralité...")

# Calcul S/P mensuel
sin_mensuel = (df_sinistres.groupby("mois_dt")["montant_indemnise"]
               .sum().reset_index(name="sinistres_payes"))
expo_mensuel = (df_expo.groupby("mois_dt")["prime_acquise"]
                .sum().reset_index(name="primes_acquises"))
kpi_mensuel = sin_mensuel.merge(expo_mensuel, on="mois_dt", how="inner")
kpi_mensuel["loss_ratio"] = kpi_mensuel["sinistres_payes"] / kpi_mensuel["primes_acquises"]
kpi_mensuel = kpi_mensuel.sort_values("mois_dt")

# S/P par produit et année
sin_prod_an = (df_sinistres.groupby(["annee", "type_produit"])["montant_indemnise"]
               .sum().reset_index(name="sinistres"))
expo_prod_an = (df_expo.assign(annee=df_expo["mois_dt"].dt.year)
                .groupby(["annee", "type_produit"])["prime_acquise"]
                .sum().reset_index(name="primes"))
kpi_prod_an = sin_prod_an.merge(expo_prod_an, on=["annee", "type_produit"])
kpi_prod_an["loss_ratio"] = kpi_prod_an["sinistres"] / kpi_prod_an["primes"]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("InsuraSight — KPIs Sinistralité", fontsize=15, fontweight="bold", y=1.01)

# 3a — Loss Ratio mensuel (courbe lissée)
ax = axes[0, 0]
ax.plot(kpi_mensuel["mois_dt"], kpi_mensuel["loss_ratio"],
        color=COLORS["neutral"], linewidth=0.8, alpha=0.5, label="Mensuel brut")
rolling = kpi_mensuel["loss_ratio"].rolling(3, center=True).mean()
ax.plot(kpi_mensuel["mois_dt"], rolling,
        color=COLORS["primary"], linewidth=2, label="Moyenne glissante 3M")
ax.axhline(0.70, color=COLORS["warn"], linestyle="--", linewidth=1.2, label="Seuil alerte 70%")
ax.axhline(0.80, color=COLORS["accent"], linestyle="--", linewidth=1.2, label="Seuil critique 80%")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_title("Loss Ratio mensuel (S/P)")
ax.set_xlabel("")
ax.set_ylabel("S/P")
ax.legend(fontsize=8)

# 3b — S/P par produit et année (grouped bar)
ax = axes[0, 1]
annees = sorted(kpi_prod_an["annee"].unique())
produits_list = ["Auto", "Habitation", "Santé"]
x = np.arange(len(produits_list))
w = 0.2
for i, an in enumerate(annees):
    vals = [kpi_prod_an[(kpi_prod_an["annee"] == an) &
                         (kpi_prod_an["type_produit"] == p)]["loss_ratio"].values
            for p in produits_list]
    vals = [v[0] if len(v) > 0 else 0 for v in vals]
    bars = ax.bar(x + i * w, vals, w, label=str(an),
                  color=plt.cm.Blues(0.4 + i * 0.15), edgecolor="white")
ax.axhline(0.70, color=COLORS["warn"], linestyle="--", linewidth=1.2)
ax.set_xticks(x + w * (len(annees) - 1) / 2)
ax.set_xticklabels(produits_list)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_title("Loss Ratio par produit et année")
ax.legend(fontsize=8, title="Année")

# 3c — Fréquence sinistre mensuelle
ax = axes[1, 0]
nb_sin_mensuel = df_sinistres.groupby("mois_dt").size().reset_index(name="nb_sinistres")
polices_actives = len(df_polices[df_polices["statut"] == "Active"])
nb_sin_mensuel["freq"] = nb_sin_mensuel["nb_sinistres"] / polices_actives
ax.bar(nb_sin_mensuel["mois_dt"], nb_sin_mensuel["nb_sinistres"],
       width=20, color=COLORS["secondary"], edgecolor="none", alpha=0.7)
ax.set_title(f"Volume mensuel de sinistres déclarés")
ax.set_ylabel("Nombre de sinistres")
# Mise en évidence des pics
top3 = nb_sin_mensuel.nlargest(3, "nb_sinistres")
for _, row in top3.iterrows():
    ax.annotate(f'{row["nb_sinistres"]}',
                xy=(row["mois_dt"], row["nb_sinistres"]),
                xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=8, color=COLORS["accent"], fontweight="bold")

# 3d — Coût moyen et fréquence par région (scatter = pure premium)
ax = axes[1, 1]
sin_region = df_sinistres.merge(df_polices[["police_id", "region"]], on="police_id", how="left")
region_freq = sin_region.groupby("region").size() / df_polices.groupby("region").size()
region_cout = sin_region.groupby("region")["montant_indemnise"].mean()
regions = region_freq.index.intersection(region_cout.index)
ax.scatter(region_freq[regions], region_cout[regions],
           s=120, color=COLORS["primary"], zorder=5)
for reg in regions:
    ax.annotate(reg.replace("-", "\n"), (region_freq[reg], region_cout[reg]),
                fontsize=7, ha="center", va="bottom",
                xytext=(0, 6), textcoords="offset points")
ax.set_title("Fréquence vs Coût moyen par région")
ax.set_xlabel("Fréquence sinistre (sin/police)")
ax.set_ylabel("Coût moyen indemnisé (€)")

plt.tight_layout()
plt.savefig("reports/figures/03_kpis_sinistralite.png", bbox_inches="tight")
plt.close()
print("   ✅ Sauvegardé : 03_kpis_sinistralite.png")


# ─────────────────────────────────────────
# FIGURE 4 — CORRÉLATIONS & RISQUE
# ─────────────────────────────────────────

print("📊 Figure 4 — Corrélations et axes de risque...")

# Tableau enrichi : une ligne par sinistre + features client
df_merged = (df_sinistres
             .merge(df_polices[["police_id", "prime_annuelle",
                                  "niveau_garantie"]], on="police_id")
             .merge(df_clients[["client_id", "age", "anciennete_annees",
                                  "nb_sinistres_passes", "score_risque_init"]], on="client_id"))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("InsuraSight — Corrélations & axes de risque", fontsize=15, fontweight="bold", y=1.01)

# 4a — Heatmap des corrélations
ax = axes[0, 0]
num_cols = ["montant_indemnise", "prime_annuelle", "age",
            "anciennete_annees", "nb_sinistres_passes", "score_risque_init",
            "nb_jours_traitement", "delai_declaration_j"]
corr = df_merged[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 7},
            xticklabels=[c.replace("_", "\n") for c in num_cols],
            yticklabels=[c.replace("_", "\n") for c in num_cols])
ax.set_title("Matrice de corrélations")
ax.tick_params(axis="both", labelsize=7)

# 4b — Score de risque vs montant indemnisé (scatter par produit)
ax = axes[0, 1]
for prod, color in zip(["Auto", "Habitation", "Santé"], PALETTE3):
    sub = df_merged[df_merged["type_produit"] == prod].sample(
        min(200, len(df_merged[df_merged["type_produit"] == prod])), random_state=42)
    ax.scatter(sub["score_risque_init"], sub["montant_indemnise"],
               alpha=0.4, color=color, label=prod, s=18, edgecolors="none")
ax.set_title("Score risque vs montant indemnisé")
ax.set_xlabel("Score de risque initial")
ax.set_ylabel("Montant indemnisé (€)")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# 4c — Sinistres passés vs fréquence future (barplot)
ax = axes[1, 0]
df_merged["tranche_sin_passes"] = pd.cut(df_merged["nb_sinistres_passes"],
                                          bins=[-1, 0, 1, 2, 10],
                                          labels=["0", "1", "2", "3+"])
freq_sin = df_merged.groupby("tranche_sin_passes", observed=True).size()
cout_sin = df_merged.groupby("tranche_sin_passes", observed=True)["montant_indemnise"].mean()

ax2 = ax.twinx()
ax.bar(freq_sin.index, freq_sin.values, color=COLORS["primary"], alpha=0.7, label="Nb sinistres")
ax2.plot(cout_sin.index, cout_sin.values, "o-", color=COLORS["accent"],
         linewidth=2, markersize=6, label="Coût moyen")
ax.set_title("Sinistres passés → volume et coût actuels")
ax.set_xlabel("Nb sinistres antérieurs")
ax.set_ylabel("Nombre de sinistres actuels", color=COLORS["primary"])
ax2.set_ylabel("Coût moyen indemnisé (€)", color=COLORS["accent"])
ax.legend(loc="upper left", fontsize=8)
ax2.legend(loc="upper right", fontsize=8)

# 4d — Âge vs coût moyen par tranche (barplot)
ax = axes[1, 1]
df_merged["tranche_age"] = pd.cut(df_merged["age"],
                                   bins=[17, 25, 35, 45, 55, 65, 85],
                                   labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"])
age_cout = df_merged.groupby("tranche_age", observed=True)["montant_indemnise"].mean()
age_nb   = df_merged.groupby("tranche_age", observed=True).size()
colors_age = [COLORS["accent"] if v > age_cout.mean() else COLORS["secondary"]
              for v in age_cout.values]
bars = ax.bar(age_cout.index, age_cout.values, color=colors_age, edgecolor="white")
ax.axhline(age_cout.mean(), color=COLORS["neutral"], linestyle="--",
           linewidth=1.2, label=f"Moyenne : {age_cout.mean():,.0f} €")
for bar, nb in zip(bars, age_nb.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"n={nb}", ha="center", va="bottom", fontsize=8)
ax.set_title("Coût moyen indemnisé par tranche d'âge")
ax.set_xlabel("Tranche d'âge")
ax.set_ylabel("Coût moyen (€)")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("reports/figures/04_correlations_risque.png", bbox_inches="tight")
plt.close()
print("   ✅ Sauvegardé : 04_correlations_risque.png")


# ─────────────────────────────────────────
# FIGURE 5 — ANALYSE TEMPORELLE
# ─────────────────────────────────────────

print("📊 Figure 5 — Tendances temporelles...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("InsuraSight — Tendances temporelles", fontsize=15, fontweight="bold", y=1.01)

# 5a — Évolution mensuelle : sinistres et primes (dual axis)
ax = axes[0, 0]
kpi = kpi_mensuel.copy()
ax2 = ax.twinx()
ax.fill_between(kpi["mois_dt"], kpi["primes_acquises"] / 1000,
                alpha=0.3, color=COLORS["success"], label="Primes (k€)")
ax.plot(kpi["mois_dt"], kpi["primes_acquises"] / 1000,
        color=COLORS["success"], linewidth=1.5)
ax2.plot(kpi["mois_dt"], kpi["sinistres_payes"] / 1000,
         color=COLORS["accent"], linewidth=1.5, label="Sinistres (k€)")
ax.set_title("Primes acquises vs Sinistres payés")
ax.set_ylabel("Primes acquises (k€)", color=COLORS["success"])
ax2.set_ylabel("Sinistres payés (k€)", color=COLORS["accent"])
ax.legend(loc="upper left", fontsize=8)
ax2.legend(loc="upper right", fontsize=8)

# 5b — Saisonnalité : sinistres par mois de l'année
ax = axes[0, 1]
df_sinistres["mois_annee"] = df_sinistres["date_survenance"].dt.month
sin_mois = df_sinistres.groupby("mois_annee").agg(
    nb=("sinistre_id", "count"),
    cout_moy=("montant_indemnise", "mean")
)
noms_mois = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
ax.bar(range(1, 13), sin_mois["nb"], color=COLORS["secondary"], edgecolor="white", alpha=0.8)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(noms_mois, fontsize=9)
ax.set_title("Saisonnalité : volume sinistres par mois")
ax.set_ylabel("Nombre moyen de sinistres")

# 5c — Croissance du portefeuille (polices actives par mois)
ax = axes[1, 0]
pol_actives = (df_expo[df_expo["statut_police"] == "Active"]
               .groupby("mois_dt")["police_id"].nunique())
ax.plot(pol_actives.index, pol_actives.values,
        color=COLORS["primary"], linewidth=2)
ax.fill_between(pol_actives.index, pol_actives.values,
                alpha=0.15, color=COLORS["primary"])
ax.set_title("Croissance du portefeuille (polices actives)")
ax.set_ylabel("Nombre de polices actives")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# 5d — Heatmap S/P mois × produit
ax = axes[1, 1]
sin_mp = (df_sinistres.assign(mois=df_sinistres["date_survenance"].dt.to_period("M").astype(str))
          .groupby(["mois", "type_produit"])["montant_indemnise"].sum())
expo_mp = (df_expo.assign(mois=df_expo["mois"])
           .groupby(["mois", "type_produit"])["prime_acquise"].sum())
lr_mp = (sin_mp / expo_mp).unstack("type_produit").fillna(0)
# Garder les 24 derniers mois pour lisibilité
lr_mp = lr_mp.tail(24)
sns.heatmap(lr_mp.T, ax=ax, cmap="RdYlGn_r", center=0.7,
            vmin=0, vmax=1.5, linewidths=0.3, linecolor="white",
            fmt=".0%", annot=True, annot_kws={"size": 7},
            xticklabels=[m[-5:] for m in lr_mp.index],
            cbar_kws={"label": "Loss Ratio"})
ax.set_title("Heatmap S/P — 24 derniers mois")
ax.set_xlabel("Mois")
ax.set_ylabel("")
ax.tick_params(axis="x", rotation=45, labelsize=7)

plt.tight_layout()
plt.savefig("reports/figures/05_tendances_temporelles.png", bbox_inches="tight")
plt.close()
print("   ✅ Sauvegardé : 05_tendances_temporelles.png")


# ─────────────────────────────────────────
# RAPPORT KPI CONSOLE
# ─────────────────────────────────────────

print("\n" + "=" * 55)
print("  TABLEAU DE BORD KPIs — ANNÉE 2023")
print("=" * 55)
df_2023_sin = df_sinistres[df_sinistres["annee"] == 2023]
df_2023_exp = df_expo[df_expo["mois_dt"].dt.year == 2023]

primes_2023    = df_2023_exp["prime_acquise"].sum()
sinistres_2023 = df_2023_sin["montant_indemnise"].sum()
loss_ratio     = sinistres_2023 / primes_2023
freq_sin       = len(df_2023_sin) / len(df_polices[df_polices["statut"] == "Active"])
cout_moyen     = df_2023_sin["montant_indemnise"].mean()
delai_moyen    = df_2023_sin["nb_jours_traitement"].mean()
taux_ouvert    = (df_2023_sin["statut"] == "Ouvert").mean()
taux_litigieux = (df_2023_sin["statut"] == "Litigieux").mean()

print(f"  {'Primes acquises':<30}: {primes_2023:>12,.0f} €")
print(f"  {'Sinistres payés':<30}: {sinistres_2023:>12,.0f} €")
print(f"  {'Loss Ratio (S/P)':<30}: {loss_ratio:>12.2%}")
print(f"  {'Fréquence sinistre':<30}: {freq_sin:>12.4f} sin/police")
print(f"  {'Coût moyen indemnisé':<30}: {cout_moyen:>12,.0f} €")
print(f"  {'Délai moyen traitement':<30}: {delai_moyen:>12.1f} jours")
print(f"  {'Taux dossiers ouverts':<30}: {taux_ouvert:>12.2%}")
print(f"  {'Taux dossiers litigieux':<30}: {taux_litigieux:>12.2%}")
print("=" * 55)
print("\n✅ EDA terminée — 5 figures sauvegardées dans reports/figures/")
