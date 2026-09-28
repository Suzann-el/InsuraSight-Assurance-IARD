"""
InsuraSight — Notebook 02 : Modélisation actuarielle
======================================================
Double modélisation fréquence × sévérité :

  FRÉQUENCE  : nombre de sinistres par police/an
    → GLM Poisson (baseline actuariel)
    → XGBoost Poisson (challenger)

  SÉVÉRITÉ   : coût moyen conditionnel (si sinistre)
    → GLM Gamma (log-link, baseline)
    → XGBoost régression (challenger)

  PRIME PURE : E[N] × E[C | N>0]
    → Score de risque final utilisé pour la tarification

Toutes les expériences sont trackées dans MLflow.

Usage :
    python notebooks/02_modeling.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
import pickle

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (mean_absolute_error, mean_squared_error,
                                     mean_poisson_deviance)
from sklearn.linear_model    import PoissonRegressor, GammaRegressor
from sklearn.pipeline        import Pipeline
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")
os.makedirs("reports/figures", exist_ok=True)
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
}

plt.rcParams.update({
    "figure.dpi": 120, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold",
})

# ─────────────────────────────────────────
# CHARGEMENT & PRÉPARATION
# ─────────────────────────────────────────

print("📂 Chargement de la table de modélisation...")
df = pd.read_csv("data/features/table_modelisation.csv")

# Features communes aux deux modèles
FEATURE_COLS = [c for c in df.columns if c not in
                ["police_id", "client_id", "nb_sinistres",
                 "montant_moy_indem", "montant_total_indem", "exposition_annees"]]

print(f"   Shape : {df.shape}  |  Features : {len(FEATURE_COLS)}")
print(f"   Polices avec sinistre : {(df['nb_sinistres'] > 0).sum()} "
      f"({(df['nb_sinistres'] > 0).mean():.2%})\n")

# ── Split train / test ──────────────────────────────────────────────────────
X = df[FEATURE_COLS].fillna(0)
y_freq = df["nb_sinistres"]
y_sev  = df["montant_moy_indem"]
expo   = df["exposition_annees"]

X_train, X_test, yf_train, yf_test, ys_train, ys_test, e_train, e_test = (
    train_test_split(X, y_freq, y_sev, expo, test_size=0.2, random_state=SEED)
)

# Pour le modèle sévérité : garder uniquement les polices sinistrées
mask_sin_train = yf_train > 0
mask_sin_test  = yf_test  > 0

X_sev_train = X_train[mask_sin_train]
X_sev_test  = X_test[mask_sin_test]
ys_train_clean = ys_train[mask_sin_train].fillna(ys_train.median())
ys_test_clean  = ys_test[mask_sin_test].fillna(ys_test.median())

print(f"Train : {len(X_train)} polices  |  Test : {len(X_test)} polices")
print(f"Train sévérité : {len(X_sev_train)} polices sinistrées\n")

# ─────────────────────────────────────────
# MLflow setup
# ─────────────────────────────────────────

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("InsuraSight_Sinistralite")

def log_metrics_freq(y_true, y_pred, prefix=""):
    """Log des métriques fréquence."""
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    # Poisson deviance (nécessite y_pred > 0)
    y_pred_clipped = np.clip(y_pred, 1e-6, None)
    dev   = mean_poisson_deviance(y_true, y_pred_clipped)
    y_true_arr = np.array(y_true)
    gini  = 1 - 2 * np.sum(
        np.cumsum(y_true_arr[np.argsort(y_pred)]) / y_true_arr.sum()
    ) / len(y_true_arr)
    mlflow.log_metrics({
        f"{prefix}mae": round(mae, 6),
        f"{prefix}rmse": round(rmse, 6),
        f"{prefix}poisson_deviance": round(dev, 6),
        f"{prefix}gini": round(gini, 4),
    })
    return {"MAE": mae, "RMSE": rmse, "Poisson Dev.": dev, "Gini": gini}

def log_metrics_sev(y_true, y_pred, prefix=""):
    """Log des métriques sévérité."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
    mlflow.log_metrics({
        f"{prefix}mae": round(mae, 2),
        f"{prefix}rmse": round(rmse, 2),
        f"{prefix}mape": round(mape, 2),
    })
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}

# ─────────────────────────────────────────
# MODÈLE 1 — GLM POISSON (FRÉQUENCE)
# ─────────────────────────────────────────

print("=" * 55)
print("  MODÈLE 1 — GLM Poisson (Fréquence)")
print("=" * 55)

with mlflow.start_run(run_name="GLM_Poisson_Frequence"):
    mlflow.set_tags({"model_type": "GLM", "target": "frequence", "framework": "sklearn"})
    mlflow.log_params({
        "alpha": 1e-4, "max_iter": 300,
        "features": len(FEATURE_COLS), "train_size": len(X_train),
    })

    pipe_glm_freq = Pipeline([
        ("scaler", StandardScaler()),
        ("glm",    PoissonRegressor(alpha=1e-4, max_iter=300)),
    ])
    pipe_glm_freq.fit(X_train, yf_train, glm__sample_weight=e_train)

    pred_glm_freq_train = pipe_glm_freq.predict(X_train)
    pred_glm_freq_test  = pipe_glm_freq.predict(X_test)

    metrics_glm_freq = log_metrics_freq(yf_test, pred_glm_freq_test, "test_")
    mlflow.sklearn.log_model(pipe_glm_freq, "model")

print(f"   Test MAE         : {metrics_glm_freq['MAE']:.5f}")
print(f"   Test RMSE        : {metrics_glm_freq['RMSE']:.5f}")
print(f"   Test Poisson Dev : {metrics_glm_freq['Poisson Dev.']:.5f}")
print(f"   Test Gini        : {metrics_glm_freq['Gini']:.4f}\n")

# ─────────────────────────────────────────
# MODÈLE 2 — XGBOOST POISSON (FRÉQUENCE)
# ─────────────────────────────────────────

print("=" * 55)
print("  MODÈLE 2 — XGBoost Poisson (Fréquence)")
print("=" * 55)

XGB_FREQ_PARAMS = {
    "objective"        : "count:poisson",
    "n_estimators"     : 300,
    "max_depth"        : 4,
    "learning_rate"    : 0.05,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "min_child_weight" : 10,   # régularisation importante en assurance
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 1.0,
    "seed"             : SEED,
    "n_jobs"           : -1,
}

with mlflow.start_run(run_name="XGBoost_Poisson_Frequence"):
    mlflow.set_tags({"model_type": "XGBoost", "target": "frequence", "framework": "xgboost"})
    mlflow.log_params(XGB_FREQ_PARAMS)

    xgb_freq = xgb.XGBRegressor(**XGB_FREQ_PARAMS)
    xgb_freq.fit(
        X_train, yf_train,
        sample_weight=e_train,
        eval_set=[(X_test, yf_test)],
        verbose=False,
    )

    pred_xgb_freq_train = xgb_freq.predict(X_train)
    pred_xgb_freq_test  = xgb_freq.predict(X_test)

    metrics_xgb_freq = log_metrics_freq(yf_test, pred_xgb_freq_test, "test_")
    mlflow.xgboost.log_model(xgb_freq, "model")

print(f"   Test MAE         : {metrics_xgb_freq['MAE']:.5f}")
print(f"   Test RMSE        : {metrics_xgb_freq['RMSE']:.5f}")
print(f"   Test Poisson Dev : {metrics_xgb_freq['Poisson Dev.']:.5f}")
print(f"   Test Gini        : {metrics_xgb_freq['Gini']:.4f}\n")

# ─────────────────────────────────────────
# MODÈLE 3 — GLM GAMMA (SÉVÉRITÉ)
# ─────────────────────────────────────────

print("=" * 55)
print("  MODÈLE 3 — GLM Gamma (Sévérité)")
print("=" * 55)

with mlflow.start_run(run_name="GLM_Gamma_Severite"):
    mlflow.set_tags({"model_type": "GLM", "target": "severite", "framework": "sklearn"})
    mlflow.log_params({"alpha": 1e-4, "max_iter": 300,
                       "train_size_sev": len(X_sev_train)})

    pipe_glm_sev = Pipeline([
        ("scaler", StandardScaler()),
        ("glm",    GammaRegressor(alpha=1e-4, max_iter=300)),
    ])
    pipe_glm_sev.fit(X_sev_train, ys_train_clean)

    pred_glm_sev_test  = pipe_glm_sev.predict(X_sev_test)
    metrics_glm_sev = log_metrics_sev(ys_test_clean, pred_glm_sev_test, "test_")
    mlflow.sklearn.log_model(pipe_glm_sev, "model")

print(f"   Test MAE  : {metrics_glm_sev['MAE']:,.0f} €")
print(f"   Test RMSE : {metrics_glm_sev['RMSE']:,.0f} €")
print(f"   Test MAPE : {metrics_glm_sev['MAPE%']:.1f} %\n")

# ─────────────────────────────────────────
# MODÈLE 4 — XGBOOST GAMMA (SÉVÉRITÉ)
# ─────────────────────────────────────────

print("=" * 55)
print("  MODÈLE 4 — XGBoost Gamma (Sévérité)")
print("=" * 55)

XGB_SEV_PARAMS = {
    "objective"        : "reg:gamma",
    "n_estimators"     : 300,
    "max_depth"        : 4,
    "learning_rate"    : 0.05,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "min_child_weight" : 5,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 1.0,
    "seed"             : SEED,
    "n_jobs"           : -1,
}

with mlflow.start_run(run_name="XGBoost_Gamma_Severite"):
    mlflow.set_tags({"model_type": "XGBoost", "target": "severite", "framework": "xgboost"})
    mlflow.log_params(XGB_SEV_PARAMS)

    xgb_sev = xgb.XGBRegressor(**XGB_SEV_PARAMS)
    xgb_sev.fit(
        X_sev_train, ys_train_clean,
        eval_set=[(X_sev_test, ys_test_clean)],
        verbose=False,
    )

    pred_xgb_sev_test  = xgb_sev.predict(X_sev_test)
    metrics_xgb_sev = log_metrics_sev(ys_test_clean, pred_xgb_sev_test, "test_")
    mlflow.xgboost.log_model(xgb_sev, "model")

print(f"   Test MAE  : {metrics_xgb_sev['MAE']:,.0f} €")
print(f"   Test RMSE : {metrics_xgb_sev['RMSE']:,.0f} €")
print(f"   Test MAPE : {metrics_xgb_sev['MAPE%']:.1f} %\n")

# ─────────────────────────────────────────
# PRIME PURE = E[N] × E[C | N>0]
# ─────────────────────────────────────────

print("=" * 55)
print("  PRIME PURE PRÉDITE (meilleurs modèles)")
print("=" * 55)

# On choisit XGBoost pour les deux (challenger gagne généralement)
freq_pred_all = xgb_freq.predict(X)
sev_pred_all  = xgb_sev.predict(X)
pure_premium  = freq_pred_all * sev_pred_all

df["freq_pred"]        = freq_pred_all
df["sev_pred"]         = sev_pred_all
df["pure_premium_pred"] = pure_premium

df.to_csv("data/features/table_scores.csv", index=False)

print(f"   Prime pure moyenne : {pure_premium.mean():,.0f} €/police/an")
print(f"   Prime pure médiane : {np.median(pure_premium):,.0f} €/police/an")
print(f"   Percentile 90      : {np.percentile(pure_premium, 90):,.0f} €")
print(f"   Percentile 99      : {np.percentile(pure_premium, 99):,.0f} €\n")

# ─────────────────────────────────────────
# FIGURE 6 — RÉSULTATS MODÈLES
# ─────────────────────────────────────────

print("📊 Génération des figures de résultats...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle("InsuraSight — Résultats de modélisation", fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Ligne 1 : Comparaison modèles fréquence ─────────────────────────────────

# 6a — Comparaison MAE / Gini
ax = fig.add_subplot(gs[0, 0])
models_names = ["GLM\nPoisson", "XGBoost\nPoisson"]
maes  = [metrics_glm_freq["MAE"], metrics_xgb_freq["MAE"]]
ginis = [metrics_glm_freq["Gini"], metrics_xgb_freq["Gini"]]
x_pos = np.arange(len(models_names))
bars = ax.bar(x_pos, maes, color=[COLORS["secondary"], COLORS["primary"]],
              edgecolor="white", width=0.5)
for bar, val in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x_pos); ax.set_xticklabels(models_names)
ax.set_title("MAE — Fréquence (↓ meilleur)")
ax.set_ylabel("MAE")

# 6b — Observed vs Predicted (fréquence, XGBoost)
ax = fig.add_subplot(gs[0, 1])
freq_bins = [0, 0, 1, 1, 2, 2, 3]
obs_by_score = pd.DataFrame({
    "score": pred_xgb_freq_test,
    "obs"  : yf_test.values,
}).sort_values("score")
obs_by_score["decile"] = pd.qcut(obs_by_score["score"], 10, labels=False, duplicates="drop")
lift = obs_by_score.groupby("decile").agg(obs=("obs","mean"), pred=("score","mean"))
ax.plot(lift["pred"], lift["obs"], "o-", color=COLORS["primary"],
        linewidth=2, markersize=5, label="Observé")
ax.plot([lift["pred"].min(), lift["pred"].max()],
        [lift["pred"].min(), lift["pred"].max()],
        "--", color=COLORS["accent"], linewidth=1.5, label="Parfait")
ax.set_title("Lift chart — Fréquence (XGBoost)")
ax.set_xlabel("Fréquence prédite (décile)")
ax.set_ylabel("Fréquence observée")
ax.legend(fontsize=8)

# 6c — Feature importance fréquence (XGBoost)
ax = fig.add_subplot(gs[0, 2])
imp_freq = pd.Series(xgb_freq.feature_importances_, index=FEATURE_COLS).nlargest(10)
ax.barh(range(len(imp_freq)), imp_freq.values[::-1],
        color=COLORS["primary"], edgecolor="white")
ax.set_yticks(range(len(imp_freq)))
ax.set_yticklabels(imp_freq.index[::-1], fontsize=8)
ax.set_title("Top 10 features — Fréquence")
ax.set_xlabel("Importance (gain)")

# ── Ligne 2 : Comparaison modèles sévérité ──────────────────────────────────

# 6d — MAE sévérité
ax = fig.add_subplot(gs[1, 0])
maes_sev = [metrics_glm_sev["MAE"], metrics_xgb_sev["MAE"]]
bars = ax.bar(["GLM\nGamma", "XGBoost\nGamma"], maes_sev,
              color=[COLORS["secondary"], COLORS["primary"]],
              edgecolor="white", width=0.5)
for bar, val in zip(bars, maes_sev):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f"{val:,.0f} €", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_title("MAE — Sévérité (↓ meilleur)")
ax.set_ylabel("MAE (€)")

# 6e — Residuals sévérité (XGBoost)
ax = fig.add_subplot(gs[1, 1])
residuals = ys_test_clean.values - pred_xgb_sev_test
ax.scatter(pred_xgb_sev_test, residuals, alpha=0.3, s=12,
           color=COLORS["secondary"], edgecolors="none")
ax.axhline(0, color=COLORS["accent"], linestyle="--", linewidth=1.5)
ax.set_title("Résidus — Sévérité (XGBoost)")
ax.set_xlabel("Prédiction (€)")
ax.set_ylabel("Résidu (€)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# 6f — Feature importance sévérité (XGBoost)
ax = fig.add_subplot(gs[1, 2])
imp_sev = pd.Series(xgb_sev.feature_importances_, index=FEATURE_COLS).nlargest(10)
ax.barh(range(len(imp_sev)), imp_sev.values[::-1],
        color=COLORS["accent"], edgecolor="white")
ax.set_yticks(range(len(imp_sev)))
ax.set_yticklabels(imp_sev.index[::-1], fontsize=8)
ax.set_title("Top 10 features — Sévérité")
ax.set_xlabel("Importance (gain)")

# ── Ligne 3 : Prime pure ─────────────────────────────────────────────────────

# 6g — Distribution de la prime pure prédite
ax = fig.add_subplot(gs[2, 0])
ax.hist(np.clip(pure_premium, 0, np.percentile(pure_premium, 99)),
        bins=50, color=COLORS["warn"], edgecolor="white", linewidth=0.5)
ax.axvline(pure_premium.mean(), color=COLORS["accent"], linestyle="--",
           linewidth=1.5, label=f"Moyenne : {pure_premium.mean():,.0f} €")
ax.set_title("Distribution Prime Pure prédite")
ax.set_xlabel("Prime pure (€/an)")
ax.set_ylabel("Nombre de polices")
ax.legend(fontsize=8)

# 6h — Prime pure par produit (box)
ax = fig.add_subplot(gs[2, 1])
# Récupérer le produit depuis le df scores
produit_cols = [c for c in df.columns if c.startswith("type_produit_")]
df["produit_label"] = df[produit_cols].idxmax(axis=1).str.replace("type_produit_", "")
data_box = [df[df["produit_label"] == p]["pure_premium_pred"].values
            for p in ["Auto", "Habitation", "Santé"]]
bp = ax.boxplot(data_box, patch_artist=True, labels=["Auto", "Habitation", "Santé"],
                medianprops={"color": "white", "linewidth": 2},
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.3})
for patch, color in zip(bp["boxes"], [COLORS["primary"], COLORS["secondary"], COLORS["warn"]]):
    patch.set_facecolor(color)
ax.set_title("Prime pure par produit")
ax.set_ylabel("Prime pure (€/an)")

# 6i — Tableau récapitulatif des métriques
ax = fig.add_subplot(gs[2, 2])
ax.axis("off")
table_data = [
    ["", "GLM", "XGBoost"],
    ["─ FRÉQUENCE ─", "", ""],
    ["MAE",       f"{metrics_glm_freq['MAE']:.5f}", f"{metrics_xgb_freq['MAE']:.5f}"],
    ["Gini",      f"{metrics_glm_freq['Gini']:.4f}", f"{metrics_xgb_freq['Gini']:.4f}"],
    ["Poisson D.",f"{metrics_glm_freq['Poisson Dev.']:.4f}", f"{metrics_xgb_freq['Poisson Dev.']:.4f}"],
    ["─ SÉVÉRITÉ ─", "", ""],
    ["MAE (€)",   f"{metrics_glm_sev['MAE']:,.0f}", f"{metrics_xgb_sev['MAE']:,.0f}"],
    ["RMSE (€)",  f"{metrics_glm_sev['RMSE']:,.0f}", f"{metrics_xgb_sev['RMSE']:,.0f}"],
    ["MAPE",      f"{metrics_glm_sev['MAPE%']:.1f}%", f"{metrics_xgb_sev['MAPE%']:.1f}%"],
]
tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.6)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor(COLORS["primary"])
        cell.set_text_props(color="white", fontweight="bold")
    elif table_data[row][0].startswith("─"):
        cell.set_facecolor(COLORS["light"] if hasattr(COLORS, "light") else "#D6EAF8")
        cell.set_text_props(fontweight="bold")
ax.set_title("Résumé des métriques", fontsize=11, fontweight="bold", pad=10)

plt.savefig("reports/figures/06_resultats_modelisation.png", bbox_inches="tight")
plt.close()
print("   ✅ Sauvegardé : 06_resultats_modelisation.png")

# ─────────────────────────────────────────
# SHAP — Explicabilité XGBoost Fréquence
# ─────────────────────────────────────────

print("📊 Calcul SHAP values (fréquence)...")

explainer   = shap.TreeExplainer(xgb_freq)
shap_values = explainer.shap_values(X_test.iloc[:300])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("InsuraSight — Explicabilité SHAP (Fréquence)", fontsize=14, fontweight="bold")

# SHAP summary bar
plt.sca(axes[0])
shap.summary_plot(shap_values, X_test.iloc[:300], plot_type="bar",
                  feature_names=FEATURE_COLS, show=False, max_display=12)
axes[0].set_title("Importance globale (|SHAP| moyen)")

# SHAP beeswarm
plt.sca(axes[1])
shap.summary_plot(shap_values, X_test.iloc[:300],
                  feature_names=FEATURE_COLS, show=False, max_display=12)
axes[1].set_title("Impact directionnel des features")

plt.tight_layout()
plt.savefig("reports/figures/07_shap_frequence.png", bbox_inches="tight")
plt.close()
print("   ✅ Sauvegardé : 07_shap_frequence.png")

# ─────────────────────────────────────────
# SAUVEGARDE DES MODÈLES
# ─────────────────────────────────────────

print("\n💾 Sauvegarde des modèles...")
with open("models/glm_poisson_freq.pkl", "wb") as f:
    pickle.dump(pipe_glm_freq, f)
with open("models/xgb_poisson_freq.pkl", "wb") as f:
    pickle.dump(xgb_freq, f)
with open("models/glm_gamma_sev.pkl", "wb") as f:
    pickle.dump(pipe_glm_sev, f)
with open("models/xgb_gamma_sev.pkl", "wb") as f:
    pickle.dump(xgb_sev, f)
with open("models/feature_cols.pkl", "wb") as f:
    pickle.dump(FEATURE_COLS, f)

print("   ✅ 4 modèles + feature list sauvegardés dans models/")

# ─────────────────────────────────────────
# RÉSUMÉ FINAL
# ─────────────────────────────────────────

print("\n" + "=" * 55)
print("  RÉCAPITULATIF MODÉLISATION")
print("=" * 55)
print(f"\n  FRÉQUENCE")
print(f"  {'Modèle':<22} {'MAE':>10} {'Gini':>8} {'Dev.':>8}")
print(f"  {'-'*50}")
print(f"  {'GLM Poisson':<22} {metrics_glm_freq['MAE']:>10.5f} "
      f"{metrics_glm_freq['Gini']:>8.4f} {metrics_glm_freq['Poisson Dev.']:>8.4f}")
print(f"  {'XGBoost Poisson':<22} {metrics_xgb_freq['MAE']:>10.5f} "
      f"{metrics_xgb_freq['Gini']:>8.4f} {metrics_xgb_freq['Poisson Dev.']:>8.4f}")
print(f"\n  SÉVÉRITÉ")
print(f"  {'Modèle':<22} {'MAE (€)':>10} {'RMSE (€)':>10} {'MAPE':>8}")
print(f"  {'-'*50}")
print(f"  {'GLM Gamma':<22} {metrics_glm_sev['MAE']:>10,.0f} "
      f"{metrics_glm_sev['RMSE']:>10,.0f} {metrics_glm_sev['MAPE%']:>7.1f}%")
print(f"  {'XGBoost Gamma':<22} {metrics_xgb_sev['MAE']:>10,.0f} "
      f"{metrics_xgb_sev['RMSE']:>10,.0f} {metrics_xgb_sev['MAPE%']:>7.1f}%")
print(f"\n  Prime pure (XGBoost×XGBoost)")
print(f"  Moyenne : {pure_premium.mean():,.0f} €   Médiane : {np.median(pure_premium):,.0f} €")
print(f"\n✅ Modélisation terminée | Runs trackés dans MLflow (mlruns/)")
