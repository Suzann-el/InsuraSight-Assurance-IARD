"""
InsuraSight — Notebook 04 : Prévision temporelle (Prophet)
===========================================================
Trois séries prévisionnelles :
  1. Volume mensuel de sinistres déclarés
  2. Montant total mensuel des indemnisations
  3. Loss Ratio mensuel (S/P)

Approche :
  - Baseline : moyenne glissante (benchmark naïf)
  - Modèle 1  : Prophet univarié avec saisonnalité annuelle
  - Modèle 2  : Prophet avec régresseur exogène (nb polices actives)
  - Backtesting : validation sur les 6 derniers mois observés
  - Forecast   : 6 mois dans le futur (Jan–Juin 2025)

Usage :
    python notebooks/04_forecast.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import os
import pickle

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

SEED = 42
np.random.seed(SEED)

COLORS = {
    "primary"  : "#1B4F72",
    "secondary": "#2E86C1",
    "accent"   : "#E74C3C",
    "success"  : "#27AE60",
    "warn"     : "#F39C12",
    "neutral"  : "#5D6D7E",
    "forecast" : "#8E44AD",
}

plt.rcParams.update({
    "figure.dpi": 120, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10,
})

# ─────────────────────────────────────────
# CHARGEMENT & AGRÉGATION MENSUELLE
# ─────────────────────────────────────────

print("📂 Chargement et préparation des séries temporelles...")

df_sinistres = pd.read_csv("data/raw/sinistres.csv",
                            parse_dates=["date_survenance", "date_declaration"])
df_expo      = pd.read_csv("data/raw/exposition_mensuelle.csv")
df_polices   = pd.read_csv("data/raw/polices.csv")

df_expo["mois_dt"] = pd.to_datetime(df_expo["mois"])

# Série 1 : Volume mensuel de sinistres
df_sinistres["mois_dt"] = df_sinistres["date_survenance"].dt.to_period("M").dt.to_timestamp()
sin_mensuel = (df_sinistres
               .groupby("mois_dt")
               .agg(nb_sinistres     = ("sinistre_id", "count"),
                    montant_indem    = ("montant_indemnise", "sum"))
               .reset_index()
               .rename(columns={"mois_dt": "ds"}))

# Série 2 : Primes acquises mensuelles
primes_mensuel = (df_expo
                  .groupby("mois_dt")["prime_acquise"]
                  .sum()
                  .reset_index()
                  .rename(columns={"mois_dt": "ds", "prime_acquise": "primes"}))

# Série 3 : Polices actives mensuelles (régresseur exogène)
pol_actives = (df_expo[df_expo["statut_police"] == "Active"]
               .groupby("mois_dt")["police_id"]
               .nunique()
               .reset_index()
               .rename(columns={"mois_dt": "ds", "police_id": "nb_polices_actives"}))

# Fusion
ts = (sin_mensuel
      .merge(primes_mensuel, on="ds", how="inner")
      .merge(pol_actives,    on="ds", how="left")
      .sort_values("ds")
      .reset_index(drop=True))

ts["loss_ratio"] = ts["montant_indem"] / ts["primes"].clip(lower=1)

print(f"   Série temporelle : {len(ts)} mois "
      f"({ts['ds'].min().strftime('%Y-%m')} → {ts['ds'].max().strftime('%Y-%m')})")
print(f"   Volume moyen mensuel  : {ts['nb_sinistres'].mean():.1f} sinistres/mois")
print(f"   Montant moyen mensuel : {ts['montant_indem'].mean():,.0f} €/mois")
print(f"   Loss Ratio moyen      : {ts['loss_ratio'].mean():.2%}\n")

# ─────────────────────────────────────────
# SPLIT TRAIN / VALIDATION
# ─────────────────────────────────────────

HORIZON_VALID = 6    # mois de validation (backtesting)
HORIZON_PRED  = 6    # mois de prévision future

cutoff_date = ts["ds"].max() - pd.DateOffset(months=HORIZON_VALID)
ts_train = ts[ts["ds"] <= cutoff_date].copy()
ts_valid = ts[ts["ds"] >  cutoff_date].copy()

print(f"   Train : {len(ts_train)} mois (jusqu'à {cutoff_date.strftime('%Y-%m')})")
print(f"   Valid : {len(ts_valid)} mois\n")

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def evaluate(y_true, y_pred, name=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None))) * 100
    print(f"   {name:<30} MAE={mae:>8.2f}  RMSE={rmse:>8.2f}  MAPE={mape:>6.1f}%")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def build_future_regressors(model_df, ts_full, periods, freq="MS"):
    """Génère le dataframe future pour Prophet avec régresseurs."""
    last_date  = model_df["ds"].max()
    future_ds  = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                periods=periods, freq=freq)
    # Extrapoler les régresseurs : tendance linéaire sur les 12 derniers mois
    reg_vals = ts_full.set_index("ds")["nb_polices_actives"].ffill()
    trend    = np.polyfit(range(12), reg_vals.iloc[-12:].values, 1)
    future_reg = trend[0] * np.arange(1, periods + 1) + reg_vals.iloc[-1]

    future = pd.DataFrame({
        "ds": future_ds,
        "nb_polices_actives": future_reg.astype(int),
    })
    return pd.concat([model_df[["ds", "nb_polices_actives"]], future], ignore_index=True)


# ═══════════════════════════════════════════════════════════
# MODÈLE 1 — VOLUME DE SINISTRES
# ═══════════════════════════════════════════════════════════

print("=" * 55)
print("  MODÈLE 1 — Volume mensuel de sinistres")
print("=" * 55)

# ── Baseline : moyenne glissante 3M ─────────────────────────
baseline_pred_vol = ts_train["nb_sinistres"].rolling(3).mean().iloc[-HORIZON_VALID:].values
# Padding si nécessaire
if len(baseline_pred_vol) < HORIZON_VALID:
    baseline_pred_vol = np.full(HORIZON_VALID, ts_train["nb_sinistres"].mean())
baseline_metrics_vol = evaluate(ts_valid["nb_sinistres"].values,
                                 np.full(HORIZON_VALID, ts_train["nb_sinistres"].tail(3).mean()),
                                 "Baseline (moy. gliss. 3M)")

# ── Prophet univarié ─────────────────────────────────────────
df_vol_train = ts_train[["ds", "nb_sinistres"]].rename(columns={"nb_sinistres": "y"})
df_vol_valid = ts_valid[["ds", "nb_sinistres"]].rename(columns={"nb_sinistres": "y"})

m_vol = Prophet(
    yearly_seasonality   = True,
    weekly_seasonality   = False,
    daily_seasonality    = False,
    seasonality_mode     = "multiplicative",
    changepoint_prior_scale = 0.05,
    seasonality_prior_scale = 10,
    interval_width       = 0.90,
)
m_vol.fit(df_vol_train)

# Validation
future_val = m_vol.make_future_dataframe(periods=HORIZON_VALID, freq="MS", include_history=False)
forecast_val_vol = m_vol.predict(future_val)
prophet_metrics_vol = evaluate(df_vol_valid["y"].values,
                                forecast_val_vol["yhat"].values,
                                "Prophet univarié")

# Prévision future
future_full = m_vol.make_future_dataframe(periods=len(ts_valid) + HORIZON_PRED, freq="MS")
forecast_vol = m_vol.predict(future_full)

# Sauvegarde
with open("models/prophet_volume_sinistres.pkl", "wb") as f:
    pickle.dump(m_vol, f)

winner = "Prophet" if prophet_metrics_vol['MAE'] < baseline_metrics_vol['MAE'] else "Baseline"
print(f"   → Meilleur modèle : {winner}  "
      f"(MAE Prophet={prophet_metrics_vol['MAE']:.1f} vs Baseline={baseline_metrics_vol['MAE']:.1f})\n")


# ═══════════════════════════════════════════════════════════
# MODÈLE 2 — MONTANT MENSUEL DES INDEMNISATIONS
# ═══════════════════════════════════════════════════════════

print("=" * 55)
print("  MODÈLE 2 — Montant mensuel des indemnisations")
print("=" * 55)

baseline_metrics_mnt = evaluate(ts_valid["montant_indem"].values,
                                 np.full(HORIZON_VALID, ts_train["montant_indem"].tail(3).mean()),
                                 "Baseline (moy. gliss. 3M)")

df_mnt_train = ts_train[["ds", "montant_indem", "nb_polices_actives"]].rename(
    columns={"montant_indem": "y"}).fillna({"nb_polices_actives": ts_train["nb_polices_actives"].median()})
df_mnt_valid = ts_valid[["ds", "montant_indem", "nb_polices_actives"]].rename(
    columns={"montant_indem": "y"}).fillna({"nb_polices_actives": ts_valid["nb_polices_actives"].median()})

# Prophet avec régresseur exogène (nb polices actives)
m_mnt = Prophet(
    yearly_seasonality      = True,
    weekly_seasonality      = False,
    daily_seasonality       = False,
    seasonality_mode        = "multiplicative",
    changepoint_prior_scale = 0.1,
    interval_width          = 0.90,
)
m_mnt.add_regressor("nb_polices_actives", standardize=True)
m_mnt.fit(df_mnt_train)

# Validation
future_val_mnt = df_mnt_valid[["ds", "nb_polices_actives"]].copy()
forecast_val_mnt = m_mnt.predict(future_val_mnt)
prophet_metrics_mnt = evaluate(df_mnt_valid["y"].values,
                                forecast_val_mnt["yhat"].values,
                                "Prophet + régresseur")

# Prévision future
future_mnt_full = build_future_regressors(df_mnt_train, ts, HORIZON_PRED)
forecast_mnt = m_mnt.predict(future_mnt_full)

with open("models/prophet_montant_indem.pkl", "wb") as f:
    pickle.dump(m_mnt, f)

winner_mnt = "Prophet" if prophet_metrics_mnt['MAE'] < baseline_metrics_mnt['MAE'] else "Baseline"
print(f"   → Meilleur modèle : {winner_mnt}  "
      f"(MAE Prophet={prophet_metrics_mnt['MAE']:,.0f} vs Baseline={baseline_metrics_mnt['MAE']:,.0f} €)\n")


# ═══════════════════════════════════════════════════════════
# MODÈLE 3 — LOSS RATIO MENSUEL
# ═══════════════════════════════════════════════════════════

print("=" * 55)
print("  MODÈLE 3 — Loss Ratio mensuel")
print("=" * 55)

# Le Loss Ratio brut mensuel est très volatile (données synthétiques sur petit volume).
# On le prédit en lissant sur une fenêtre glissante 3M (S/P = moy. mob. sinistres / primes).
ts["lr_smooth"] = (
    ts["montant_indem"].rolling(3, center=True, min_periods=1).mean() /
    ts["primes"].rolling(3, center=True, min_periods=1).mean()
)

ts_train_lr = ts[ts["ds"] <= cutoff_date].copy()
ts_valid_lr = ts[ts["ds"] >  cutoff_date].copy()

baseline_pred_lr = np.full(HORIZON_VALID, ts_train_lr["lr_smooth"].tail(3).mean())
baseline_metrics_lr = evaluate(ts_valid_lr["lr_smooth"].values, baseline_pred_lr,
                                "Baseline (moy. gliss. 3M)")

df_lr_train = ts_train_lr[["ds", "lr_smooth"]].rename(columns={"lr_smooth": "y"})
df_lr_valid = ts_valid_lr[["ds", "lr_smooth"]].rename(columns={"lr_smooth": "y"})

m_lr = Prophet(
    yearly_seasonality      = True,
    weekly_seasonality      = False,
    daily_seasonality       = False,
    seasonality_mode        = "additive",
    changepoint_prior_scale = 0.05,
    interval_width          = 0.90,
)
m_lr.fit(df_lr_train)

future_val_lr   = m_lr.make_future_dataframe(periods=HORIZON_VALID, freq="MS", include_history=False)
forecast_val_lr = m_lr.predict(future_val_lr)
prophet_metrics_lr = evaluate(df_lr_valid["y"].values,
                               forecast_val_lr["yhat"].values,
                               "Prophet univarié (S/P lissé)")

future_lr_full = m_lr.make_future_dataframe(periods=len(ts_valid) + HORIZON_PRED, freq="MS")
forecast_lr    = m_lr.predict(future_lr_full)

# Clamp : S/P ne peut pas être négatif
forecast_lr["yhat"]       = forecast_lr["yhat"].clip(lower=0)
forecast_lr["yhat_lower"] = forecast_lr["yhat_lower"].clip(lower=0)
forecast_lr["yhat_upper"] = forecast_lr["yhat_upper"].clip(lower=0)

with open("models/prophet_loss_ratio.pkl", "wb") as f:
    pickle.dump(m_lr, f)

winner_lr = "Prophet" if prophet_metrics_lr['MAE'] < baseline_metrics_lr['MAE'] else "Baseline"
print(f"   → Meilleur modèle : {winner_lr}  "
      f"(MAE Prophet={prophet_metrics_lr['MAE']:.4f} vs Baseline={baseline_metrics_lr['MAE']:.4f})\n")


# ─────────────────────────────────────────
# SAUVEGARDE DES PRÉVISIONS
# ─────────────────────────────────────────

# Construire une table de prévision propre
last_obs = ts["ds"].max()
forecast_dates = pd.date_range(
    start=last_obs + pd.DateOffset(months=1),
    periods=HORIZON_PRED, freq="MS"
)

# Indexer les forecasts sur les dates futures
fc_vol_future = forecast_vol[forecast_vol["ds"].isin(forecast_dates)][["ds", "yhat", "yhat_lower", "yhat_upper"]]
fc_mnt_future = forecast_mnt[forecast_mnt["ds"].isin(forecast_dates)][["ds", "yhat", "yhat_lower", "yhat_upper"]]
fc_lr_future  = forecast_lr[forecast_lr["ds"].isin(forecast_dates)][["ds", "yhat", "yhat_lower", "yhat_upper"]]

df_previsions = pd.DataFrame({
    "mois"                      : forecast_dates,
    "nb_sinistres_pred"         : fc_vol_future["yhat"].values if len(fc_vol_future) == HORIZON_PRED else np.nan,
    "nb_sinistres_lower"        : fc_vol_future["yhat_lower"].values if len(fc_vol_future) == HORIZON_PRED else np.nan,
    "nb_sinistres_upper"        : fc_vol_future["yhat_upper"].values if len(fc_vol_future) == HORIZON_PRED else np.nan,
    "montant_indem_pred"        : fc_mnt_future["yhat"].values if len(fc_mnt_future) == HORIZON_PRED else np.nan,
    "montant_indem_lower"       : fc_mnt_future["yhat_lower"].values if len(fc_mnt_future) == HORIZON_PRED else np.nan,
    "montant_indem_upper"       : fc_mnt_future["yhat_upper"].values if len(fc_mnt_future) == HORIZON_PRED else np.nan,
    "loss_ratio_pred"           : np.clip(fc_lr_future["yhat"].values, 0, None) if len(fc_lr_future) == HORIZON_PRED else np.nan,
    "loss_ratio_lower"          : np.clip(fc_lr_future["yhat_lower"].values, 0, None) if len(fc_lr_future) == HORIZON_PRED else np.nan,
    "loss_ratio_upper"          : np.clip(fc_lr_future["yhat_upper"].values, 0, None) if len(fc_lr_future) == HORIZON_PRED else np.nan,
})
df_previsions.to_csv("data/processed/previsions_6mois.csv", index=False)
print(f"✅ Prévisions sauvegardées : data/processed/previsions_6mois.csv\n")


# ─────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────

print("📊 Génération des figures...")

fig, axes = plt.subplots(3, 2, figsize=(16, 15))
fig.suptitle("InsuraSight — Prévisions temporelles (Prophet)", fontsize=15, fontweight="bold")

def plot_forecast(ax, ts_hist, ts_val, forecast_df, target_col,
                  ylabel, title, color, fmt_y=None, seuil=None, seuil_label=None):
    """Helper : trace historique + validation + forecast + IC."""
    # Historique (train)
    ax.plot(ts_hist["ds"], ts_hist[target_col],
            color=COLORS["neutral"], linewidth=1.5, alpha=0.7, label="Historique")
    # Validation (observé vs prédit)
    ax.plot(ts_val["ds"], ts_val[target_col],
            color=COLORS["primary"], linewidth=2, linestyle="-", label="Observé (valid.)")

    # Forecast complet (passé + futur)
    hist_mask    = forecast_df["ds"] <= ts_hist["ds"].max()
    valid_mask   = (forecast_df["ds"] > ts_hist["ds"].max()) & (forecast_df["ds"] <= ts_val["ds"].max())
    future_mask  = forecast_df["ds"] > ts_val["ds"].max()

    ax.plot(forecast_df.loc[valid_mask, "ds"], forecast_df.loc[valid_mask, "yhat"],
            color=color, linewidth=2, linestyle="--", label="Prédit (valid.)")
    ax.fill_between(forecast_df.loc[valid_mask, "ds"],
                    forecast_df.loc[valid_mask, "yhat_lower"],
                    forecast_df.loc[valid_mask, "yhat_upper"],
                    alpha=0.15, color=color)

    ax.plot(forecast_df.loc[future_mask, "ds"], forecast_df.loc[future_mask, "yhat"],
            color=COLORS["forecast"], linewidth=2.5, linestyle="-", label="Prévision")
    ax.fill_between(forecast_df.loc[future_mask, "ds"],
                    forecast_df.loc[future_mask, "yhat_lower"],
                    forecast_df.loc[future_mask, "yhat_upper"],
                    alpha=0.20, color=COLORS["forecast"])

    # Ligne de coupure train/valid
    ax.axvline(ts_hist["ds"].max(), color=COLORS["warn"], linestyle=":", linewidth=1.5,
               label="Coupure train/valid")
    # Seuil métier
    if seuil is not None:
        ax.axhline(seuil, color=COLORS["accent"], linestyle="--", linewidth=1.2,
                   label=seuil_label or f"Seuil {seuil}")

    if fmt_y:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_y))

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, loc="upper left")

# ── Vol sinistres ─────────────────────────────────────────────

forecast_vol_merged = forecast_vol.merge(
    ts[["ds", "nb_sinistres"]], on="ds", how="left")

plot_forecast(
    ax=axes[0, 0],
    ts_hist=ts_train, ts_val=ts_valid,
    forecast_df=forecast_vol,
    target_col="nb_sinistres",
    ylabel="Nb sinistres", title="Volume mensuel de sinistres",
    color=COLORS["secondary"],
)

# ── Composantes Prophet : tendance + saisonnalité ─────────────

ax = axes[0, 1]
trend_data = forecast_vol[["ds", "trend"]].copy()
ax2 = ax.twinx()
ax.plot(trend_data["ds"], trend_data["trend"],
        color=COLORS["primary"], linewidth=2, label="Tendance")
yearly = forecast_vol[["ds", "yearly"]].copy() if "yearly" in forecast_vol.columns else None
if yearly is not None:
    ax2.bar(yearly["ds"], yearly["yearly"], width=25,
            color=COLORS["warn"], alpha=0.5, label="Saisonnalité annuelle")
ax.set_title("Composantes Prophet — Volume sinistres")
ax.set_ylabel("Tendance", color=COLORS["primary"])
ax2.set_ylabel("Effet saisonnier", color=COLORS["warn"])
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# ── Montant indemnisations ────────────────────────────────────

plot_forecast(
    ax=axes[1, 0],
    ts_hist=ts_train, ts_val=ts_valid,
    forecast_df=forecast_mnt,
    target_col="montant_indem",
    ylabel="Montant (€)", title="Montant mensuel des indemnisations",
    color=COLORS["secondary"],
    fmt_y=lambda x, _: f"{x/1000:.0f}k",
)

# ── Scatter : prédit vs observé (validation) ─────────────────

ax = axes[1, 1]
obs_mnt  = ts_valid["montant_indem"].values
pred_mnt = forecast_mnt[forecast_mnt["ds"].isin(ts_valid["ds"].values)]["yhat"].values
if len(pred_mnt) == len(obs_mnt):
    ax.scatter(obs_mnt, pred_mnt, color=COLORS["secondary"], s=60,
               edgecolors=COLORS["primary"], linewidths=0.8, zorder=5)
    lim = max(obs_mnt.max(), pred_mnt.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color=COLORS["accent"], linewidth=1.5, label="Parfait")
    for i, (o, p) in enumerate(zip(obs_mnt, pred_mnt)):
        ax.annotate(ts_valid["ds"].iloc[i].strftime("%m/%y"),
                    (o, p), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.legend(fontsize=8)
ax.set_title("Prédit vs Observé — Montant (validation)")
ax.set_xlabel("Observé (k€)")
ax.set_ylabel("Prédit (k€)")

# ── Loss Ratio ───────────────────────────────────────────────

plot_forecast(
    ax=axes[2, 0],
    ts_hist=ts_train_lr, ts_val=ts_valid_lr,
    forecast_df=forecast_lr,
    target_col="lr_smooth",
    ylabel="Loss Ratio (S/P)", title="Loss Ratio lissé 3M — Prévision 6 mois",
    color=COLORS["secondary"],
    fmt_y=lambda x, _: f"{x:.0%}",
    seuil=0.70, seuil_label="Seuil alerte 70%",
)

# ── Tableau récapitulatif des prévisions ─────────────────────

ax = axes[2, 1]
ax.axis("off")
ax.set_title("Prévisions Jan–Juin 2025", fontsize=11, fontweight="bold", pad=10)

table_rows = [["Mois", "Sinistres", "Indemnisations", "S/P"]]
for _, row in df_previsions.iterrows():
    nb_sin = row["nb_sinistres_pred"]
    mnt    = row["montant_indem_pred"]
    lr     = row["loss_ratio_pred"]
    table_rows.append([
        row["mois"].strftime("%b %Y") if hasattr(row["mois"], "strftime") else str(row["mois"])[:7],
        f"{nb_sin:.0f}" if pd.notna(nb_sin) else "—",
        f"{mnt/1000:.0f}k €" if pd.notna(mnt) else "—",
        f"{lr:.1%}" if pd.notna(lr) else "—",
    ])

tbl = ax.table(cellText=table_rows[1:], colLabels=table_rows[0],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.8)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor(COLORS["primary"])
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#eaf2fb")
    # Alerte si S/P > 70% prévu
    if col == 3 and row > 0:
        try:
            val = float(cell.get_text().get_text().strip("%")) / 100
            if val > 0.70:
                cell.set_facecolor("#fadbd8")
                cell.set_text_props(color=COLORS["accent"], fontweight="bold")
        except:
            pass

plt.tight_layout()
plt.savefig("reports/figures/10_previsions_temporelles.png", bbox_inches="tight")
plt.close()
print("   ✅ Sauvegardé : 10_previsions_temporelles.png")


# ─────────────────────────────────────────
# RÉSUMÉ FINAL
# ─────────────────────────────────────────

print("\n" + "=" * 55)
print("  RÉCAPITULATIF PRÉVISIONS")
print("=" * 55)
print(f"\n  Métriques de validation (6 derniers mois observés) :\n")
print(f"  {'Modèle':<35} {'MAE':>10} {'MAPE':>8}")
print(f"  {'-'*55}")
print(f"  {'Vol. sinistres — Baseline':<35} {baseline_metrics_vol['MAE']:>10.1f} "
      f"{baseline_metrics_vol['MAPE']:>7.1f}%")
print(f"  {'Vol. sinistres — Prophet':<35} {prophet_metrics_vol['MAE']:>10.1f} "
      f"{prophet_metrics_vol['MAPE']:>7.1f}%")
print(f"  {'Montant indem. — Baseline':<35} {baseline_metrics_mnt['MAE']:>10,.0f} "
      f"{baseline_metrics_mnt['MAPE']:>7.1f}%")
print(f"  {'Montant indem. — Prophet + rég.':<35} {prophet_metrics_mnt['MAE']:>10,.0f} "
      f"{prophet_metrics_mnt['MAPE']:>7.1f}%")
print(f"  {'Loss Ratio — Baseline':<35} {baseline_metrics_lr['MAE']:>10.4f} "
      f"{baseline_metrics_lr['MAPE']:>7.1f}%")
print(f"  {'Loss Ratio — Prophet':<35} {prophet_metrics_lr['MAE']:>10.4f} "
      f"{prophet_metrics_lr['MAPE']:>7.1f}%")

print(f"\n  Prévisions Jan–Juin 2025 :")
for _, row in df_previsions.iterrows():
    mois  = str(row["mois"])[:7]
    nb    = row["nb_sinistres_pred"]
    mnt   = row["montant_indem_pred"]
    lr    = row["loss_ratio_pred"]
    alerte = " ⚠️" if pd.notna(lr) and lr > 0.70 else ""
    print(f"  {mois}  | Sinistres : {nb:>5.0f}  | "
          f"Indem. : {mnt/1000:>6.0f}k €  | S/P : {lr:.1%}{alerte}")

print(f"\n✅ Partie 5 terminée | Prévisions dans data/processed/previsions_6mois.csv")
