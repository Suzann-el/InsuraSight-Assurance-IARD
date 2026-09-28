"""
InsuraSight — Notebook 03 : Clustering clients & Détection de fraude
=====================================================================
PARTIE A — Segmentation clients
  → K-Means (4 segments) sur comportement sinistralité + profil risque
  → UMAP pour visualisation 2D des clusters
  → Profil moyen par segment + recommandations métier

PARTIE B — Détection de fraude
  → Isolation Forest sur les sinistres (anomaly detection)
  → Score d'anomalie par dossier
  → Règles métier combinées au score ML

Usage :
    python notebooks/03_clustering_fraude.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
import pickle

from sklearn.preprocessing   import StandardScaler
from sklearn.cluster         import KMeans
from sklearn.metrics         import silhouette_score, davies_bouldin_score
from sklearn.ensemble        import IsolationForest
from sklearn.decomposition   import PCA
import umap

warnings.filterwarnings("ignore")
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

SEED = 42
np.random.seed(SEED)

COLORS = {
    "primary"  : "#1B4F72",
    "secondary": "#2E86C1",
    "accent"   : "#E74C3C",
    "success"  : "#27AE60",
    "warn"     : "#F39C12",
    "neutral"  : "#5D6D7E",
    "purple"   : "#7D3C98",
}
CLUSTER_COLORS = [COLORS["primary"], COLORS["success"], COLORS["warn"], COLORS["accent"]]

plt.rcParams.update({
    "figure.dpi": 120, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10,
})

# ─────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────

print("📂 Chargement des données...")
df_clients   = pd.read_csv("data/raw/clients.csv")
df_polices   = pd.read_csv("data/raw/polices.csv")
df_sinistres = pd.read_csv("data/raw/sinistres.csv", parse_dates=["date_survenance"])
df_scores    = pd.read_csv("data/features/table_scores.csv")

print(f"   Clients   : {len(df_clients):,}")
print(f"   Sinistres : {len(df_sinistres):,}\n")

# ═══════════════════════════════════════════════════════════
# PARTIE A — SEGMENTATION CLIENTS
# ═══════════════════════════════════════════════════════════

print("=" * 55)
print("  PARTIE A — SEGMENTATION CLIENTS")
print("=" * 55)

# ── Construction du profil client agrégé ────────────────────

# Par client : agréger toutes les polices + sinistres
sin_cli = (
    df_sinistres
    .groupby("client_id")
    .agg(
        nb_sinistres_total  = ("sinistre_id", "count"),
        montant_total       = ("montant_indemnise", "sum"),
        montant_moy         = ("montant_indemnise", "mean"),
        nb_litigieux        = ("statut", lambda x: (x == "Litigieux").sum()),
        delai_decl_moy      = ("delai_declaration_j", "mean"),
        nb_fraude_flag      = ("fraude_potentielle", "sum"),
    )
    .reset_index()
)

pol_cli = (
    df_polices
    .groupby("client_id")
    .agg(
        nb_polices          = ("police_id", "count"),
        prime_totale        = ("prime_annuelle", "sum"),
        prime_moy           = ("prime_annuelle", "mean"),
        nb_produits_diff    = ("type_produit", "nunique"),
    )
    .reset_index()
)

profil = (
    df_clients
    .merge(pol_cli,  on="client_id", how="left")
    .merge(sin_cli,  on="client_id", how="left")
)

# Remplissage des clients sans sinistres
for col in ["nb_sinistres_total", "montant_total", "montant_moy",
            "nb_litigieux", "delai_decl_moy", "nb_fraude_flag"]:
    profil[col] = profil[col].fillna(0)

# Features enrichies
profil["freq_sin_par_police"]   = profil["nb_sinistres_total"] / profil["nb_polices"].clip(lower=1)
profil["taux_litigieux"]        = profil["nb_litigieux"] / profil["nb_sinistres_total"].clip(lower=1)
profil["ratio_sinistre_prime"]  = profil["montant_total"] / profil["prime_totale"].clip(lower=1)
profil["flag_zero_sinistre"]    = (profil["nb_sinistres_total"] == 0).astype(int)

# ── Features pour clustering ────────────────────────────────

CLUSTER_FEATURES = [
    "score_risque_init",
    "age",
    "anciennete_annees",
    "nb_sinistres_passes",
    "nb_polices",
    "nb_produits_diff",
    "prime_moy",
    "freq_sin_par_police",
    "montant_moy",
    "taux_litigieux",
    "ratio_sinistre_prime",
]

X_clust = profil[CLUSTER_FEATURES].fillna(0)
scaler_clust = StandardScaler()
X_scaled = scaler_clust.fit_transform(X_clust)

# ── Choix optimal de K (méthode elbow + silhouette) ─────────

print("🔍 Recherche du K optimal...")
k_range = range(2, 9)
inertias, silhouettes, db_scores = [], [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels, sample_size=2000))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

# K optimal = 4 (bon compromis métier)
K_OPTIMAL = 4
print(f"   K choisi : {K_OPTIMAL}  |  Silhouette : {silhouettes[K_OPTIMAL-2]:.4f}\n")

# ── K-Means final ───────────────────────────────────────────

print(f"🔧 K-Means (K={K_OPTIMAL})...")
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=SEED, n_init=20, max_iter=500)
profil["cluster"] = kmeans.fit_predict(X_scaled)

# Labels métier (basés sur le profil moyen observé)
cluster_profiles = profil.groupby("cluster")[CLUSTER_FEATURES].mean()
# Trier par score de risque + fréquence pour nommer les segments
risk_rank = (
    cluster_profiles["score_risque_init"]
    + cluster_profiles["freq_sin_par_police"]
).argsort()

CLUSTER_LABELS = {}
labels_ordered = ["🟢 Bon risque fidèle", "🔵 Risque modéré", "🟠 Risque élevé", "🔴 Très haut risque"]
for rank, cluster_id in enumerate(risk_rank.values):
    CLUSTER_LABELS[cluster_id] = labels_ordered[rank]

profil["segment"] = profil["cluster"].map(CLUSTER_LABELS)

# Statistiques par segment
print("\n  Profil des segments :")
seg_stats = (
    profil.groupby("segment")
    .agg(
        nb_clients      = ("client_id", "count"),
        age_moy         = ("age", "mean"),
        score_moy       = ("score_risque_init", "mean"),
        freq_sin        = ("freq_sin_par_police", "mean"),
        montant_moy_sin = ("montant_moy", "mean"),
        ratio_SP        = ("ratio_sinistre_prime", "mean"),
        taux_litig      = ("taux_litigieux", "mean"),
        prime_moy       = ("prime_moy", "mean"),
    )
    .round(3)
)
print(seg_stats.to_string())
print()

# Sauvegarde
profil.to_csv("data/processed/profil_clients_segments.csv", index=False)
with open("models/kmeans_clients.pkl", "wb") as f:
    pickle.dump({"kmeans": kmeans, "scaler": scaler_clust,
                 "features": CLUSTER_FEATURES, "labels": CLUSTER_LABELS}, f)

# ── UMAP 2D ─────────────────────────────────────────────────

print("🗺️  Projection UMAP 2D...")
reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED, n_jobs=1)
X_2d = reducer.fit_transform(X_scaled)
profil["umap_x"] = X_2d[:, 0]
profil["umap_y"] = X_2d[:, 1]


# ═══════════════════════════════════════════════════════════
# PARTIE B — DÉTECTION DE FRAUDE
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  PARTIE B — DÉTECTION DE FRAUDE")
print("=" * 55)

# ── Features anti-fraude ────────────────────────────────────

# Une ligne par sinistre avec features enrichies
df_sin_feat = df_sinistres.merge(
    df_clients[["client_id", "age", "anciennete_annees",
                "nb_sinistres_passes", "score_risque_init"]], on="client_id"
).merge(
    df_polices[["police_id", "prime_annuelle", "type_produit"]], on="police_id"
)

# Ratio réclamé/indemnisé (fort écart = suspect)
df_sin_feat["ratio_reclame_indem"]  = (
    df_sin_feat["montant_reclame"] / df_sin_feat["montant_indemnise"].clip(lower=1)
)
# Montant réclamé vs prime annuelle
df_sin_feat["ratio_sin_prime"]      = (
    df_sin_feat["montant_reclame"] / df_sin_feat["prime_annuelle"].clip(lower=1)
)
# Sinistres passés du client
df_sin_feat["score_risque_init"]    = df_sin_feat["score_risque_init"]

FRAUD_FEATURES = [
    "montant_reclame",
    "montant_indemnise",
    "ratio_reclame_indem",
    "ratio_sin_prime",
    "delai_declaration_j",
    "nb_jours_traitement",
    "age",
    "anciennete_annees",
    "nb_sinistres_passes",
    "score_risque_init",
]

X_fraud = df_sin_feat[FRAUD_FEATURES].fillna(0)
scaler_fraud = StandardScaler()
X_fraud_scaled = scaler_fraud.fit_transform(X_fraud)

# ── Isolation Forest ────────────────────────────────────────

print("🔧 Isolation Forest...")
iso_forest = IsolationForest(
    n_estimators    = 200,
    contamination   = 0.05,   # ~5% de dossiers suspects (conservateur)
    max_samples     = "auto",
    random_state    = SEED,
    n_jobs          = -1,
)
iso_forest.fit(X_fraud_scaled)

df_sin_feat["anomaly_score"]   = -iso_forest.score_samples(X_fraud_scaled)  # + = plus suspect
df_sin_feat["anomaly_label"]   = iso_forest.predict(X_fraud_scaled)          # -1 = anomalie
df_sin_feat["flag_fraude_ml"]  = (df_sin_feat["anomaly_label"] == -1).astype(int)

# ── Règles métier combinées ──────────────────────────────────

df_sin_feat["regle_1_montant_eleve"]    = (
    df_sin_feat["montant_reclame"] > df_sin_feat["montant_reclame"].quantile(0.95)
).astype(int)
df_sin_feat["regle_2_decl_rapide"]      = (df_sin_feat["delai_declaration_j"] <= 1).astype(int)
df_sin_feat["regle_3_recidiviste"]      = (df_sin_feat["nb_sinistres_passes"] >= 3).astype(int)
df_sin_feat["regle_4_ratio_eleve"]      = (df_sin_feat["ratio_reclame_indem"] > 1.5).astype(int)

df_sin_feat["score_regles_metier"] = (
    df_sin_feat["regle_1_montant_eleve"] * 3 +
    df_sin_feat["regle_2_decl_rapide"]   * 2 +
    df_sin_feat["regle_3_recidiviste"]   * 2 +
    df_sin_feat["regle_4_ratio_eleve"]   * 3
)
df_sin_feat["flag_priorite_haute"] = (
    (df_sin_feat["flag_fraude_ml"] == 1) &
    (df_sin_feat["score_regles_metier"] >= 3)
).astype(int)

# Résultats
n_suspects_ml    = df_sin_feat["flag_fraude_ml"].sum()
n_priorite_haute = df_sin_feat["flag_priorite_haute"].sum()
pct_suspects     = n_suspects_ml / len(df_sin_feat)

print(f"\n   Total sinistres analysés    : {len(df_sin_feat):,}")
print(f"   Suspects (Isolation Forest) : {n_suspects_ml:,}  ({pct_suspects:.2%})")
print(f"   Priorité haute (ML + règles): {n_priorite_haute:,}")
print(f"\n   Coût moyen dossier normal   : "
      f"{df_sin_feat[df_sin_feat['flag_fraude_ml']==0]['montant_indemnise'].mean():,.0f} €")
print(f"   Coût moyen dossier suspect  : "
      f"{df_sin_feat[df_sin_feat['flag_fraude_ml']==1]['montant_indemnise'].mean():,.0f} €")

df_sin_feat.to_csv("data/processed/sinistres_scores_fraude.csv", index=False)
with open("models/isolation_forest_fraude.pkl", "wb") as f:
    pickle.dump({"model": iso_forest, "scaler": scaler_fraud,
                 "features": FRAUD_FEATURES}, f)


# ─────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────

print("\n📊 Génération des figures...")

# ── FIGURE 8 — Clustering ───────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("InsuraSight — Segmentation clients", fontsize=15, fontweight="bold")

# 8a — Elbow + Silhouette
ax = axes[0, 0]
ax2 = ax.twinx()
ax.plot(list(k_range), inertias, "o-", color=COLORS["primary"],
        linewidth=2, markersize=6, label="Inertie")
ax2.plot(list(k_range), silhouettes, "s--", color=COLORS["accent"],
         linewidth=2, markersize=6, label="Silhouette")
ax.axvline(K_OPTIMAL, color=COLORS["warn"], linestyle=":", linewidth=2,
           label=f"K={K_OPTIMAL}")
ax.set_title("Choix de K — Elbow & Silhouette")
ax.set_xlabel("Nombre de clusters K")
ax.set_ylabel("Inertie intra-cluster", color=COLORS["primary"])
ax2.set_ylabel("Score Silhouette", color=COLORS["accent"])
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

# 8b — UMAP 2D
ax = axes[0, 1]
for i, (cluster_id, label) in enumerate(CLUSTER_LABELS.items()):
    mask = profil["cluster"] == cluster_id
    ax.scatter(
        profil.loc[mask, "umap_x"], profil.loc[mask, "umap_y"],
        s=8, alpha=0.5, color=CLUSTER_COLORS[i], label=label.split(" ", 1)[1],
        edgecolors="none"
    )
ax.set_title(f"Projection UMAP — {K_OPTIMAL} segments")
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.legend(fontsize=7, markerscale=3, loc="upper right")

# 8c — Taille des segments
ax = axes[0, 2]
seg_count = profil["segment"].value_counts()
short_labels = [s.split(" ", 1)[1] for s in seg_count.index]
wedges, texts, autotexts = ax.pie(
    seg_count.values, labels=short_labels, autopct="%1.1f%%",
    colors=CLUSTER_COLORS[:len(seg_count)], startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
for t in autotexts: t.set_fontsize(9)
ax.set_title("Répartition des segments")

# 8d — Score risque par segment (violin)
ax = axes[1, 0]
segments_order = sorted(CLUSTER_LABELS.values())
data_v = [profil[profil["segment"] == s]["score_risque_init"].values
          for s in segments_order]
vp = ax.violinplot(data_v, positions=range(len(segments_order)), showmedians=True)
for body, color in zip(vp["bodies"], CLUSTER_COLORS):
    body.set_facecolor(color); body.set_alpha(0.7)
ax.set_xticks(range(len(segments_order)))
ax.set_xticklabels([s.split(" ", 1)[1] for s in segments_order], fontsize=7, rotation=15)
ax.set_title("Score de risque par segment")
ax.set_ylabel("Score de risque [0–1]")

# 8e — Radar / spider chart des profils
ax = axes[1, 1]
radar_features = ["score_risque_init", "freq_sin_par_police",
                  "ratio_sinistre_prime", "taux_litigieux", "nb_polices"]
radar_labels   = ["Score\nrisque", "Fréq.\nsinistre",
                  "Ratio\nS/P", "Taux\nlitigieux", "Nb\npolices"]

angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
angles += angles[:1]

# Normaliser les features pour le radar
mins = profil[radar_features].min()
maxs = profil[radar_features].max()
profil_norm = (profil[radar_features] - mins) / (maxs - mins + 1e-9)
profil_norm["segment"] = profil["segment"]
profil_norm["cluster"] = profil["cluster"]

cluster_means = profil_norm.groupby("cluster")[radar_features].mean()

for i, (cluster_id, label) in enumerate(CLUSTER_LABELS.items()):
    if cluster_id not in cluster_means.index:
        continue
    values = cluster_means.loc[cluster_id].values.tolist()
    values += values[:1]
    ax.plot(angles, values, color=CLUSTER_COLORS[i], linewidth=2,
            label=label.split(" ", 1)[1])
    ax.fill(angles, values, color=CLUSTER_COLORS[i], alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=8)
ax.set_ylim(0, 1)
ax.set_title("Radar — Profil normalisé par segment")
ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.35, 1.1))

# 8f — Montant moyen sinistre par segment
ax = axes[1, 2]
seg_montant = profil.groupby("segment")["montant_moy"].mean().sort_values()
short_labels2 = [s.split(" ", 1)[1] for s in seg_montant.index]
color_map = {label: CLUSTER_COLORS[i] for i, label in enumerate(CLUSTER_LABELS.values())}
colors_bar = [color_map.get(s, COLORS["neutral"]) for s in seg_montant.index]
bars = ax.barh(short_labels2, seg_montant.values,
               color=colors_bar, edgecolor="white")
for bar, val in zip(bars, seg_montant.values):
    ax.text(val + 20, bar.get_y() + bar.get_height()/2,
            f"{val:,.0f} €", va="center", fontsize=9)
ax.set_title("Coût moyen sinistre par segment")
ax.set_xlabel("Montant moyen indemnisé (€)")

plt.tight_layout()
plt.savefig("reports/figures/08_clustering_clients.png", bbox_inches="tight")
plt.close()
print("   ✅ Sauvegardé : 08_clustering_clients.png")

# ── FIGURE 9 — Fraude ───────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("InsuraSight — Détection de fraude (Isolation Forest)", fontsize=15, fontweight="bold")

# 9a — Distribution du score d'anomalie
ax = axes[0, 0]
normal  = df_sin_feat[df_sin_feat["flag_fraude_ml"] == 0]["anomaly_score"]
suspect = df_sin_feat[df_sin_feat["flag_fraude_ml"] == 1]["anomaly_score"]
ax.hist(normal,  bins=40, alpha=0.7, color=COLORS["success"],  label="Normal",  density=True)
ax.hist(suspect, bins=40, alpha=0.7, color=COLORS["accent"],   label="Suspect", density=True)
ax.set_title("Distribution score d'anomalie")
ax.set_xlabel("Score anomalie (↑ = plus suspect)")
ax.set_ylabel("Densité")
ax.legend(fontsize=9)

# 9b — PCA 2D (visualisation dossiers suspects)
ax = axes[0, 1]
pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_fraud_scaled)
normal_mask  = df_sin_feat["flag_fraude_ml"] == 0
suspect_mask = df_sin_feat["flag_fraude_ml"] == 1
ax.scatter(X_pca[normal_mask,  0], X_pca[normal_mask,  1],
           s=6, alpha=0.3, color=COLORS["secondary"], label="Normal", edgecolors="none")
ax.scatter(X_pca[suspect_mask, 0], X_pca[suspect_mask, 1],
           s=25, alpha=0.8, color=COLORS["accent"],   label="Suspect ⚠️",
           edgecolors="darkred", linewidths=0.5, zorder=5)
ax.set_title(f"PCA 2D — Dossiers suspects ({n_suspects_ml} détectés)")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.legend(fontsize=8)

# 9c — Montant réclamé : normal vs suspect
ax = axes[0, 2]
data_box = [
    df_sin_feat[df_sin_feat["flag_fraude_ml"]==0]["montant_reclame"].values,
    df_sin_feat[df_sin_feat["flag_fraude_ml"]==1]["montant_reclame"].values,
]
bp = ax.boxplot(data_box, patch_artist=True, labels=["Normal", "Suspect"],
                medianprops={"color": "white", "linewidth": 2},
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.3})
bp["boxes"][0].set_facecolor(COLORS["success"])
bp["boxes"][1].set_facecolor(COLORS["accent"])
ax.set_title("Montant réclamé : normal vs suspect")
ax.set_ylabel("Montant réclamé (€)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# 9d — Délai de déclaration : normal vs suspect
ax = axes[1, 0]
bins_decl = range(0, 31, 2)
ax.hist(df_sin_feat[df_sin_feat["flag_fraude_ml"]==0]["delai_declaration_j"],
        bins=bins_decl, alpha=0.7, color=COLORS["success"], label="Normal", density=True)
ax.hist(df_sin_feat[df_sin_feat["flag_fraude_ml"]==1]["delai_declaration_j"],
        bins=bins_decl, alpha=0.7, color=COLORS["accent"],  label="Suspect", density=True)
ax.set_title("Délai de déclaration (jours)")
ax.set_xlabel("Délai (jours)")
ax.set_ylabel("Densité")
ax.legend(fontsize=9)

# 9e — Score règles métier
ax = axes[1, 1]
rule_counts = df_sin_feat["score_regles_metier"].value_counts().sort_index()
colors_rule = [COLORS["success"] if i < 3 else COLORS["warn"] if i < 5 else COLORS["accent"]
               for i in rule_counts.index]
bars = ax.bar(rule_counts.index, rule_counts.values,
              color=colors_rule, edgecolor="white")
ax.axvline(3, color=COLORS["accent"], linestyle="--", linewidth=1.5, label="Seuil alerte")
ax.set_title("Distribution du score règles métier")
ax.set_xlabel("Score (0 = ok, 8 = très suspect)")
ax.set_ylabel("Nombre de dossiers")
ax.legend(fontsize=8)
for bar, val in zip(bars, rule_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(val), ha="center", fontsize=8)

# 9f — Top 20 dossiers priorité haute
ax = axes[1, 2]
top_suspects = (
    df_sin_feat[df_sin_feat["flag_priorite_haute"] == 1]
    .nlargest(20, "anomaly_score")
    [["sinistre_id", "montant_reclame", "anomaly_score", "score_regles_metier"]]
)
ax.axis("off")
ax.set_title(f"Top dossiers priorité haute ({n_priorite_haute} total)")
if len(top_suspects) > 0:
    table_data = [["Dossier", "Montant (€)", "Score ML", "Score règles"]]
    for _, row in top_suspects.head(10).iterrows():
        table_data.append([
            row["sinistre_id"],
            f"{row['montant_reclame']:,.0f}",
            f"{row['anomaly_score']:.3f}",
            f"{int(row['score_regles_metier'])}",
        ])
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.1, 1.5)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLORS["primary"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f4f8")

plt.tight_layout()
plt.savefig("reports/figures/09_detection_fraude.png", bbox_inches="tight")
plt.close()
print("   ✅ Sauvegardé : 09_detection_fraude.png")


# ─────────────────────────────────────────
# RÉSUMÉ FINAL
# ─────────────────────────────────────────

print("\n" + "=" * 55)
print("  RÉSUMÉ PARTIE A — SEGMENTATION")
print("=" * 55)
for cluster_id, label in sorted(CLUSTER_LABELS.items()):
    n = (profil["cluster"] == cluster_id).sum()
    pct = n / len(profil) * 100
    moy_score = profil[profil["cluster"] == cluster_id]["score_risque_init"].mean()
    moy_freq  = profil[profil["cluster"] == cluster_id]["freq_sin_par_police"].mean()
    print(f"  {label:<30} : {n:>5} clients ({pct:.1f}%)  "
          f"| score={moy_score:.3f}  | freq={moy_freq:.3f}")

print(f"\n  Silhouette score : {silhouettes[K_OPTIMAL-2]:.4f}")
print(f"  Davies-Bouldin   : {db_scores[K_OPTIMAL-2]:.4f}")

print("\n" + "=" * 55)
print("  RÉSUMÉ PARTIE B — DÉTECTION FRAUDE")
print("=" * 55)
print(f"  Sinistres analysés          : {len(df_sin_feat):,}")
print(f"  Suspects Isolation Forest   : {n_suspects_ml:,}  ({pct_suspects:.2%})")
print(f"  Dossiers priorité haute     : {n_priorite_haute:,}")
print(f"  Gain potentiel estimé       : "
      f"{(df_sin_feat[df_sin_feat['flag_fraude_ml']==1]['montant_indemnise'].sum() * 0.30):,.0f} €")
print(f"  (hypothèse : 30% des montants suspects récupérables)")
print(f"\n✅ Partie 3 terminée | Fichiers dans data/processed/ et models/")
