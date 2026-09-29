"""
InsuraSight — Dashboard Streamlit
==================================
5 pages :
  📊 Vue Portefeuille    — KPIs globaux + évolution S/P
  🔍 Analyse Sinistres   — explorer les dossiers
  🎯 Score de Risque     — saisir une police et obtenir la prime pure
  👥 Segmentation        — profil des 4 segments clients
  🚨 Détection Fraude    — dossiers suspects + score ML

Lancement :
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle
import os
import sys

# ── Config page ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "InsuraSight",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Palette ──────────────────────────────────────────────────────────────────
C = {
    "primary"  : "#1B4F72",
    "secondary": "#2E86C1",
    "accent"   : "#E74C3C",
    "success"  : "#27AE60",
    "warn"     : "#F39C12",
    "neutral"  : "#5D6D7E",
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 11,
    "axes.titleweight": "bold",
})

# ── Chargement données & modèles (cache) ────────────────────────────────────

@st.cache_data
def load_data():
    base = BASE_DIR
    d = {}
    d["clients"]   = pd.read_csv(f"{base}/data/raw/clients.csv")
    d["polices"]   = pd.read_csv(f"{base}/data/raw/polices.csv")
    d["sinistres"] = pd.read_csv(f"{base}/data/raw/sinistres.csv",
                                  parse_dates=["date_survenance", "date_declaration"])
    d["expo"]      = pd.read_csv(f"{base}/data/raw/exposition_mensuelle.csv")
    d["expo"]["mois_dt"] = pd.to_datetime(d["expo"]["mois"])
    d["sinistres"]["annee"]   = d["sinistres"]["date_survenance"].dt.year
    d["sinistres"]["mois_dt"] = d["sinistres"]["date_survenance"].dt.to_period("M").dt.to_timestamp()

    if os.path.exists(f"{base}/data/processed/profil_clients_segments.csv"):
        d["segments"] = pd.read_csv(f"{base}/data/processed/profil_clients_segments.csv")
    if os.path.exists(f"{base}/data/processed/sinistres_scores_fraude.csv"):
        d["fraude"]   = pd.read_csv(f"{base}/data/processed/sinistres_scores_fraude.csv")
    if os.path.exists(f"{base}/data/processed/previsions_6mois.csv"):
        d["previsions"] = pd.read_csv(f"{base}/data/processed/previsions_6mois.csv",
                                       parse_dates=["mois"])
    return d

@st.cache_resource
def load_models():
    base = BASE_DIR
    def lm(f):
        p = f"{base}/models/{f}"
        if not os.path.exists(p): return None
        with open(p, "rb") as fp: return pickle.load(fp)
    return {
        "xgb_freq"    : lm("xgb_poisson_freq.pkl"),
        "xgb_sev"     : lm("xgb_gamma_sev.pkl"),
        "iso_forest"  : lm("isolation_forest_fraude.pkl"),
        "feature_cols": lm("feature_cols.pkl"),
    }

data   = load_data()
models = load_models()

# ── Sidebar navigation ───────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/fluency/96/insurance.png", width=60)
st.sidebar.title("InsuraSight")
st.sidebar.caption("Tableau de bord sinistralité")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Vue Portefeuille", "🔍 Analyse Sinistres",
     "🎯 Score de Risque", "👥 Segmentation", "🚨 Détection Fraude"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(f"📅 Données 2021–2024")
st.sidebar.caption(f"🏛️ Portefeuille IARD (auto / hab / santé)")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — VUE PORTEFEUILLE
# ════════════════════════════════════════════════════════════════════════════

if page == "📊 Vue Portefeuille":
    st.title("📊 Vue Portefeuille")

    annee = st.selectbox("Année de référence", [2021, 2022, 2023, 2024], index=2)

    df_sin  = data["sinistres"]
    df_expo = data["expo"]
    df_pol  = data["polices"]

    sin_an  = df_sin[df_sin["annee"] == annee]
    exp_an  = df_expo[df_expo["mois_dt"].dt.year == annee]

    primes      = exp_an["prime_acquise"].sum()
    sin_pays    = sin_an["montant_indemnise"].sum()
    loss_ratio  = sin_pays / max(primes, 1)
    freq        = len(sin_an) / max(len(df_pol[df_pol["statut"] == "Active"]), 1)
    cout_moy    = sin_an["montant_indemnise"].mean() if len(sin_an) > 0 else 0
    delai_moy   = sin_an["nb_jours_traitement"].mean() if len(sin_an) > 0 else 0

    # KPI cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("💰 Primes acquises",  f"{primes/1e6:.2f} M€")
    col2.metric("📉 Sinistres payés",  f"{sin_pays/1e6:.2f} M€")
    col3.metric("📊 Loss Ratio (S/P)", f"{loss_ratio:.1%}",
                delta=f"{'⚠️ >70%' if loss_ratio > 0.70 else '✅ OK'}",
                delta_color="off")
    col4.metric("🔁 Fréquence",        f"{freq:.3f}")
    col5.metric("💸 Coût moyen",       f"{cout_moy:,.0f} €")
    col6.metric("⏱️ Délai traitement", f"{delai_moy:.0f} j")

    st.markdown("---")

    # Graphiques
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Loss Ratio mensuel")
        sin_m = df_sin.groupby("mois_dt")["montant_indemnise"].sum().reset_index()
        exp_m = df_expo.groupby("mois_dt")["prime_acquise"].sum().reset_index()
        kpi_m = sin_m.merge(exp_m, on="mois_dt")
        kpi_m["lr"] = kpi_m["montant_indemnise"] / kpi_m["prime_acquise"].clip(lower=1)
        kpi_m = kpi_m.sort_values("mois_dt")
        kpi_m["lr_roll"] = kpi_m["lr"].rolling(3, center=True).mean()

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(kpi_m["mois_dt"], kpi_m["lr_roll"], color=C["primary"], linewidth=2)
        ax.fill_between(kpi_m["mois_dt"], kpi_m["lr_roll"], alpha=0.1, color=C["primary"])
        ax.axhline(0.70, color=C["warn"],  linestyle="--", linewidth=1, label="Alerte 70%")
        ax.axhline(0.80, color=C["accent"],linestyle="--", linewidth=1, label="Critique 80%")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_ylabel("S/P (moy. mob. 3M)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.subheader("Sinistres par produit")
        prod_stats = sin_an.groupby("type_produit").agg(
            nb=("sinistre_id","count"), cout=("montant_indemnise","sum")).reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        colors = [C["primary"], C["secondary"], C["warn"]]
        axes[0].bar(prod_stats["type_produit"], prod_stats["nb"],
                    color=colors[:len(prod_stats)], edgecolor="white")
        axes[0].set_title("Volume"); axes[0].set_ylabel("Nb sinistres")
        axes[1].bar(prod_stats["type_produit"], prod_stats["cout"]/1000,
                    color=colors[:len(prod_stats)], edgecolor="white")
        axes[1].set_title("Coût total"); axes[1].set_ylabel("k€")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Répartition des statuts")
        statut = sin_an["statut"].value_counts()
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.pie(statut.values, labels=statut.index, autopct="%1.1f%%",
               colors=[C["success"], C["warn"], C["accent"]][:len(statut)],
               wedgeprops={"edgecolor":"white","linewidth":2})
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_d:
        st.subheader("Top 5 types de sinistres (coût moyen)")
        top5 = (sin_an.groupby("type_sinistre")["montant_indemnise"]
                .mean().nlargest(5).sort_values())
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.barh(top5.index, top5.values, color=C["secondary"], edgecolor="white")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1000:.0f}k"))
        ax.set_xlabel("Coût moyen (€)")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    if "previsions" in data:
        st.markdown("---")
        st.subheader("🔮 Prévisions — 6 prochains mois")
        prev = data["previsions"]
        cols = st.columns(len(prev))
        for i, (_, row) in enumerate(prev.iterrows()):
            mois_label = row["mois"].strftime("%b %Y") if hasattr(row["mois"], "strftime") else str(row["mois"])[:7]
            lr = row.get("loss_ratio_pred", float("nan"))
            nb = row.get("nb_sinistres_pred", float("nan"))
            icon = "⚠️" if pd.notna(lr) and lr > 0.70 else "✅"
            cols[i].metric(mois_label,
                           f"{int(nb)} sin." if pd.notna(nb) else "—",
                           f"{lr:.1%} S/P {icon}" if pd.notna(lr) else "—",
                           delta_color="off")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE SINISTRES
# ════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Analyse Sinistres":
    st.title("🔍 Analyse des Sinistres")

    df = data["sinistres"].merge(
        data["polices"][["police_id","type_produit","prime_annuelle"]], on="police_id", how="left",
        suffixes=("","_pol"))

    # Filtres sidebar
    with st.sidebar:
        st.markdown("**Filtres**")
        annees = sorted(df["annee"].dropna().unique().astype(int))
        annee_f = st.multiselect("Année", annees, default=[2023, 2024])
        produit_f = st.multiselect("Produit", df["type_produit"].unique(),
                                    default=df["type_produit"].unique().tolist())
        statut_f = st.multiselect("Statut", df["statut"].unique(),
                                   default=df["statut"].unique().tolist())

    mask = (df["annee"].isin(annee_f) &
            df["type_produit"].isin(produit_f) &
            df["statut"].isin(statut_f))
    df_f = df[mask]

    col1, col2, col3 = st.columns(3)
    col1.metric("Dossiers filtrés",   f"{len(df_f):,}")
    col2.metric("Montant total",       f"{df_f['montant_indemnise'].sum()/1e6:.2f} M€")
    col3.metric("Coût moyen",          f"{df_f['montant_indemnise'].mean():,.0f} €")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribution des montants indemnisés")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        log_v = np.log1p(df_f["montant_indemnise"].clip(lower=0))
        ax.hist(log_v, bins=35, color=C["secondary"], edgecolor="white", linewidth=0.5)
        ax.set_xlabel("log(montant + 1)")
        ax.set_ylabel("Fréquence")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.subheader("Délai de traitement par statut")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for statut, color in zip(["Fermé","Ouvert","Litigieux"],
                                   [C["success"],C["warn"],C["accent"]]):
            sub = df_f[df_f["statut"] == statut]["nb_jours_traitement"].dropna()
            if len(sub): ax.hist(sub, bins=25, alpha=0.6, color=color,
                                  label=statut, density=True)
        ax.set_xlabel("Jours de traitement")
        ax.set_ylabel("Densité")
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.subheader("📋 Tableau des sinistres")
    cols_show = ["sinistre_id","police_id","type_produit","type_sinistre",
                 "date_survenance","montant_reclame","montant_indemnise","statut","nb_jours_traitement"]
    st.dataframe(
        df_f[cols_show].sort_values("montant_indemnise", ascending=False).head(200),
        use_container_width=True,
        height=350,
    )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SCORE DE RISQUE
# ════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Score de Risque":
    st.title("🎯 Score de Risque & Prime Pure")
    st.caption("Saisir les caractéristiques d'une police pour obtenir la prime pure prédite (E[N] × E[C|N>0])")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.subheader("Caractéristiques de la police")
        with st.form("scoring_form"):
            age              = st.slider("Âge de l'assuré", 18, 85, 38)
            score_risque     = st.slider("Score de risque initial", 0.0, 1.0, 0.25, 0.01)
            anciennete_cli   = st.slider("Ancienneté client (années)", 0, 20, 4)
            nb_sin_passes    = st.number_input("Sinistres passés", 0, 10, 1)
            produit          = st.selectbox("Produit", ["Auto", "Habitation", "Santé"])
            garantie_auto    = st.selectbox("Garantie Auto", ["RC seule", "Tiers étendu", "Tous risques"]) if produit == "Auto" else None
            garantie_hab     = st.selectbox("Garantie Hab.", ["Basique", "Confort", "Premium"]) if produit == "Habitation" else None
            garantie_san     = st.selectbox("Garantie Santé", ["Essentiel", "Équilibré", "Complet"]) if produit == "Santé" else None
            prime_actuelle   = st.number_input("Prime actuelle (€/an)", 100, 5000, 1400, 50)
            anciennete_cont  = st.slider("Ancienneté contrat (années)", 0, 20, 2)
            nb_sin_nm1       = st.number_input("Sinistres N-1", 0, 5, 0)
            submitted        = st.form_submit_button("🔍 Calculer le score", use_container_width=True)

    with col_result:
        if submitted:
            st.subheader("Résultats")

            # Construire le vecteur de features
            feature_cols = models["feature_cols"] or []
            garantie = garantie_auto or garantie_hab or garantie_san or ""

            raw = {
                "age": age, "anciennete_annees": anciennete_cli,
                "score_risque_init": score_risque,
                "nb_sinistres_passes": nb_sin_passes,
                "age_x_risque": age * score_risque,
                "flag_nouveau_client": int(anciennete_cli < 1),
                "prime_annuelle": prime_actuelle,
                "log_prime": np.log1p(prime_actuelle),
                "ratio_prime_marche": 1.0,
                "anciennete_contrat_an": anciennete_cont,
                "nb_sin_nm1": nb_sin_nm1,
                "montant_nm1": 0.0,
                "flag_recidiviste": int(nb_sin_nm1 >= 2),
                "taux_litigieux_nm1": 0.0,
                "region_enc": 5, "canal_enc": 0, "statut_enc": 1,
                "type_produit_Auto"       : int(produit == "Auto"),
                "type_produit_Habitation" : int(produit == "Habitation"),
                "type_produit_Santé"      : int(produit == "Santé"),
                "niveau_garantie_RC seule"   : int(garantie == "RC seule"),
                "niveau_garantie_Tiers étendu": int(garantie == "Tiers étendu"),
                "niveau_garantie_Tous risques": int(garantie == "Tous risques"),
                "niveau_garantie_Basique"     : int(garantie == "Basique"),
                "niveau_garantie_Confort"     : int(garantie == "Confort"),
                "niveau_garantie_Premium"     : int(garantie == "Premium"),
                "niveau_garantie_Essentiel"   : int(garantie == "Essentiel"),
                "niveau_garantie_Équilibré"   : int(garantie == "Équilibré"),
                "niveau_garantie_Complet"     : int(garantie == "Complet"),
            }

            X = np.array([raw.get(c, 0) for c in feature_cols], dtype=float).reshape(1, -1)

            freq = float(models["xgb_freq"].predict(X)[0])
            sev  = float(models["xgb_sev"].predict(X)[0])
            pp   = max(freq * sev, 0)
            ratio = pp / max(prime_actuelle, 1)

            if ratio < 0.5:    tarif_label, tarif_color = "Fortement sur-tarifée", "green"
            elif ratio < 0.8:  tarif_label, tarif_color = "Sur-tarifée",           "green"
            elif ratio < 1.2:  tarif_label, tarif_color = "Bien calibrée ✅",       "blue"
            elif ratio < 1.5:  tarif_label, tarif_color = "Sous-tarifée ⚠️",        "orange"
            else:              tarif_label, tarif_color = "Fortement sous-tarifée 🔴","red"

            st.metric("Fréquence prédite E[N]", f"{freq:.4f} sin/an")
            st.metric("Sévérité prédite E[C|N>0]", f"{sev:,.0f} €")
            st.metric("**Prime pure prédite**", f"{pp:,.0f} €")
            st.metric("Prime actuelle", f"{prime_actuelle:,.0f} €")

            st.markdown(f"**Tarification : :{tarif_color}[{tarif_label}]**")
            st.markdown(f"Ratio prime pure / prime actuelle : **{ratio:.2f}**")

            # Jauge visuelle
            fig, ax = plt.subplots(figsize=(5, 1.5))
            bar_color = C["success"] if ratio < 0.8 else C["warn"] if ratio < 1.2 else C["accent"]
            ax.barh(["Ratio PP/Prime"], [min(ratio, 2.5)], color=bar_color, height=0.4)
            ax.axvline(1.0, color="black", linewidth=1.5, linestyle="--")
            ax.set_xlim(0, 2.5)
            ax.set_xlabel("Ratio prime pure / prime actuelle (1.0 = équilibre)")
            ax.set_yticks([])
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        else:
            st.info("👈 Renseignez les caractéristiques de la police et cliquez sur **Calculer le score**.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SEGMENTATION
# ════════════════════════════════════════════════════════════════════════════

elif page == "👥 Segmentation":
    st.title("👥 Segmentation Clients")

    if "segments" not in data:
        st.warning("Fichier de segmentation non trouvé. Exécutez d'abord `03_clustering_fraude.py`.")
        st.stop()

    df_seg = data["segments"]

    seg_count = df_seg["segment"].value_counts()
    cols = st.columns(len(seg_count))
    for col, (seg, n) in zip(cols, seg_count.items()):
        pct = n / len(df_seg) * 100
        col.metric(seg, f"{n:,}", f"{pct:.1f}%", delta_color="off")

    st.markdown("---")

    seg_stats = (df_seg.groupby("segment")
                 .agg(nb_clients=("client_id","count"),
                      age_moy=("age","mean"),
                      score_moy=("score_risque_init","mean"),
                      freq_sin=("freq_sin_par_police","mean"),
                      cout_moy=("montant_moy","mean"),
                      prime_moy=("prime_moy","mean"),
                      ratio_sp=("ratio_sinistre_prime","mean"))
                 .round(3).reset_index())

    st.subheader("Profil moyen par segment")
    st.dataframe(seg_stats.rename(columns={
        "segment":"Segment","nb_clients":"Clients","age_moy":"Âge moy.",
        "score_moy":"Score risque","freq_sin":"Fréq. sin.","cout_moy":"Coût moy. (€)",
        "prime_moy":"Prime moy. (€)","ratio_sp":"Ratio S/P"
    }), use_container_width=True, hide_index=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Score de risque par segment")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        segments = sorted(df_seg["segment"].unique())
        data_v = [df_seg[df_seg["segment"]==s]["score_risque_init"].values for s in segments]
        colors = [C["primary"], C["success"], C["warn"], C["accent"]]
        vp = ax.violinplot(data_v, positions=range(len(segments)), showmedians=True)
        for body, color in zip(vp["bodies"], colors[:len(segments)]):
            body.set_facecolor(color); body.set_alpha(0.7)
        ax.set_xticks(range(len(segments)))
        ax.set_xticklabels([s.split(" ",1)[1] if " " in s else s
                             for s in segments], fontsize=8, rotation=15)
        ax.set_ylabel("Score de risque")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.subheader("Fréquence sinistre vs coût moyen")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for seg, color in zip(seg_stats["segment"], colors[:len(seg_stats)]):
            row = seg_stats[seg_stats["segment"] == seg].iloc[0]
            ax.scatter(row["freq_sin"], row["cout_moy"], s=row["nb_clients"]/5,
                       color=color, alpha=0.8, edgecolors="white", linewidths=1.5, zorder=5)
            ax.annotate(seg.split(" ",1)[1] if " " in seg else seg,
                        (row["freq_sin"], row["cout_moy"]),
                        fontsize=8, ha="center", va="bottom",
                        xytext=(0, 8), textcoords="offset points")
        ax.set_xlabel("Fréquence sinistre (sin/police)")
        ax.set_ylabel("Coût moyen sinistre (€)")
        ax.set_title("Taille des bulles = nb clients")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    seg_sel = st.selectbox("Détail d'un segment", sorted(df_seg["segment"].unique()))
    sub = df_seg[df_seg["segment"] == seg_sel]
    col_x, col_y, col_z = st.columns(3)
    col_x.metric("Clients", f"{len(sub):,}")
    col_y.metric("Âge moyen", f"{sub['age'].mean():.0f} ans")
    col_z.metric("Score risque moy.", f"{sub['score_risque_init'].mean():.3f}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DÉTECTION FRAUDE
# ════════════════════════════════════════════════════════════════════════════

elif page == "🚨 Détection Fraude":
    st.title("🚨 Détection de Fraude")

    if "fraude" not in data:
        st.warning("Fichier de fraude non trouvé. Exécutez d'abord `03_clustering_fraude.py`.")
        st.stop()

    df_fr = data["fraude"]
    n_sus = df_fr["flag_fraude_ml"].sum()
    n_pri = df_fr["flag_priorite_haute"].sum()
    gain  = df_fr[df_fr["flag_fraude_ml"]==1]["montant_indemnise"].sum() * 0.30

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total dossiers",     f"{len(df_fr):,}")
    col2.metric("Suspects ML",        f"{n_sus:,}", f"{n_sus/len(df_fr):.1%}", delta_color="off")
    col3.metric("Priorité haute",     f"{n_pri:,}")
    col4.metric("Gain potentiel",     f"{gain/1e6:.2f} M€", "30% récupération", delta_color="off")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribution du score d'anomalie")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df_fr[df_fr["flag_fraude_ml"]==0]["anomaly_score"],
                bins=35, alpha=0.7, color=C["success"], label="Normal", density=True)
        ax.hist(df_fr[df_fr["flag_fraude_ml"]==1]["anomaly_score"],
                bins=35, alpha=0.7, color=C["accent"],  label="Suspect", density=True)
        ax.set_xlabel("Score d'anomalie (↑ = plus suspect)")
        ax.set_ylabel("Densité")
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.subheader("Montant réclamé : normal vs suspect")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        data_bp = [df_fr[df_fr["flag_fraude_ml"]==0]["montant_reclame"].values,
                   df_fr[df_fr["flag_fraude_ml"]==1]["montant_reclame"].values]
        bp = ax.boxplot(data_bp, patch_artist=True, labels=["Normal","Suspect"],
                        medianprops={"color":"white","linewidth":2},
                        flierprops={"marker":".","markersize":3,"alpha":0.3})
        bp["boxes"][0].set_facecolor(C["success"])
        bp["boxes"][1].set_facecolor(C["accent"])
        ax.set_ylabel("Montant réclamé (€)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1000:.0f}k"))
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("📋 Dossiers priorité haute")

    priorite_filter = st.checkbox("Afficher uniquement priorité haute", value=True)
    df_show = df_fr[df_fr["flag_priorite_haute"]==1] if priorite_filter else df_fr[df_fr["flag_fraude_ml"]==1]
    df_show = df_show.sort_values("anomaly_score", ascending=False)

    cols_show = ["sinistre_id", "type_sinistre", "montant_reclame", "montant_indemnise",
                 "delai_declaration_j", "anomaly_score", "score_regles_metier",
                 "flag_priorite_haute", "statut"]
    cols_ok = [c for c in cols_show if c in df_show.columns]

    st.dataframe(
        df_show[cols_ok].head(100).style.background_gradient(
            subset=["anomaly_score"], cmap="Reds"),
        use_container_width=True,
        height=400,
    )

    if st.button("📥 Exporter les dossiers suspects (CSV)"):
        csv = df_show[cols_ok].to_csv(index=False)
        st.download_button("Télécharger", csv, "dossiers_suspects.csv", "text/csv")
