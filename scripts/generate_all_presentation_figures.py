import os
import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("reports/figures", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.size"] = 10

df = pd.read_parquet("data/raw/localidades_eda.parquet")
from src.feature_engineering import filtrar_consistencia_localidades, calcular_metricas_relativas
from src.nlp_utils import pipeline_procesamiento_nlp

df_clean = filtrar_consistencia_localidades(df)
df_rel = calcular_metricas_relativas(df_clean)
df_enriquecido = pipeline_procesamiento_nlp(df_rel, "logical_seat_category")

# 1. FIG 1: Tags Frecuencia
tag_cols = [c for c in df_enriquecido.columns if c.startswith("tag_")]
tag_freq = df_enriquecido[tag_cols].sum().sort_values(ascending=False).reset_index()
tag_freq.columns = ["Tag", "Frecuencia"]
tag_freq["Tag"] = tag_freq["Tag"].str.replace("tag_", "").str.upper()

plt.figure(figsize=(9, 6.5))
sns.barplot(data=tag_freq, x="Frecuencia", y="Tag", hue="Tag", palette="mako", legend=False)
plt.title("Frecuencia de Atributos Estructurales y Espaciales (NLP)", fontsize=12, fontweight="bold")
plt.xlabel("Cantidad de Localidades")
for i, v in enumerate(tag_freq["Frecuencia"]):
    plt.text(v + 150, i, f"{v:,}", va="center", fontsize=8.5)
plt.xlim(0, max(tag_freq["Frecuencia"]) * 1.15)
plt.tight_layout()
plt.savefig("reports/figures/fig1_tags_frecuencia.png", dpi=200)
plt.close()

# 2. FIG 2: Distribuciones Catálogo Total
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
sns.histplot(df_enriquecido["ratio_precio_max"], bins=30, kde=True, ax=axes[0, 0], color="royalblue")
axes[0, 0].set_title("Ratio de Precio Relativo", fontsize=10, fontweight="bold")
sns.histplot(df_enriquecido["peso_aforo"], bins=30, kde=True, ax=axes[0, 1], color="crimson")
axes[0, 1].set_title("Peso de Aforo (% Venue)", fontsize=10, fontweight="bold")
sns.histplot(df_enriquecido["tasa_ocupacion"], bins=30, kde=True, ax=axes[1, 0], color="seagreen")
axes[1, 0].set_title("Tasa de Ocupación", fontsize=10, fontweight="bold")
sns.histplot(df_enriquecido["percentil_precio_evento"], bins=30, kde=True, ax=axes[1, 1], color="darkorange")
axes[1, 1].set_title("Percentil de Precio por Evento", fontsize=10, fontweight="bold")
plt.suptitle("Distribuciones de Variables Relativas (Catálogo Total)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/figures/fig2_distribuciones.png", dpi=200)
plt.close()

# 3. FIG 2B: Top Venues Admisión Única
df_unica = df_enriquecido[df_enriquecido["peso_aforo"] >= 0.99]
top_venues_unica = df_unica["site"].value_counts().head(10).reset_index()
top_venues_unica.columns = ["Venue", "Total"]
top_venues_unica["Pct"] = (top_venues_unica["Total"] / len(df_unica) * 100).round(1)

plt.figure(figsize=(9, 6.5))
sns.barplot(data=top_venues_unica, y="Venue", x="Total", hue="Venue", palette="Blues_r", legend=False)
plt.title("Top 10 Venues con Mayor Cantidad de Admisión Única (Tarifa Plana)", fontsize=11, fontweight="bold")
plt.xlabel("Cantidad de Funciones")
for i, r in top_venues_unica.iterrows():
    total_val = r["Total"]
    pct_val = r["Pct"]
    plt.text(total_val + 30, i, f"{total_val:,} ({pct_val}%)", va="center", fontsize=8.5)
plt.xlim(0, max(top_venues_unica["Total"]) * 1.18)
plt.tight_layout()
plt.savefig("reports/figures/fig2b_top_venues_unica.png", dpi=200)
plt.close()

# 4. FIG 2C: Distribuciones Multi-Zona (Gráfico 2B del notebook)
df_multi = df_enriquecido[df_enriquecido["peso_aforo"] < 0.99]
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
sns.histplot(df_multi["ratio_precio_max"], bins=30, kde=True, ax=axes[0, 0], color="royalblue")
axes[0, 0].set_title("Ratio de Precio (Solo Multi-Zona)", fontsize=10, fontweight="bold")
sns.histplot(df_multi["peso_aforo"], bins=30, kde=True, ax=axes[0, 1], color="crimson")
axes[0, 1].set_title("Peso de Aforo (Solo Multi-Zona)", fontsize=10, fontweight="bold")
sns.histplot(df_multi["tasa_ocupacion"], bins=30, kde=True, ax=axes[1, 0], color="seagreen")
axes[1, 0].set_title("Tasa de Ocupación (Solo Multi-Zona)", fontsize=10, fontweight="bold")
sns.histplot(df_multi["percentil_precio_evento"], bins=30, kde=True, ax=axes[1, 1], color="darkorange")
axes[1, 1].set_title("Percentil de Precio (Solo Multi-Zona)", fontsize=10, fontweight="bold")
plt.suptitle("Distribuciones en Eventos Multi-Zona (N=18,400)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/figures/fig2d_distribuciones_multizona.png", dpi=200)
plt.close()

# 5. FIG 3A: Demostración 17 Tags (Volumen y Discriminación)
dimensiones_nlp = {
    "Jerarquía Comercial": ["tag_palco", "tag_vip", "tag_platea", "tag_preferencial", "tag_general"],
    "Nivel Vertical": ["tag_balcon", "tag_piso_alto", "tag_piso_bajo"],
    "Orientación Espacial": ["tag_occidental", "tag_oriental", "tag_norte", "tag_sur", "tag_lateral", "tag_vista_parcial"],
    "Restricción de Acceso": ["tag_familiar", "tag_menores", "tag_movilidad_reducida"]
}
stats_17 = []
for dim, tags in dimensiones_nlp.items():
    for t in tags:
        sub = df_enriquecido[df_enriquecido[t] == 1]
        n_reg = len(sub)
        stats_17.append({
            "Dimensión": dim,
            "Tag": t.replace("tag_", "").upper(),
            "Total": n_reg,
            "Aforo": sub["peso_aforo"].mean() * 100 if n_reg > 0 else 0,
            "Precio": sub["med_unit_amt_itx"].median() if n_reg > 0 else 0
        })
df_17 = pd.DataFrame(stats_17)
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
palette = {"Jerarquía Comercial": "#2b5c8f", "Nivel Vertical": "#388e3c", "Orientación Espacial": "#f57c00", "Restricción de Acceso": "#d32f2f"}
sns.barplot(data=df_17, y="Tag", x="Total", hue="Dimensión", palette=palette, ax=axes[0], dodge=False)
axes[0].set_title("Volumen por Tag NLP", fontsize=10, fontweight="bold")
axes[0].set_xlabel("Registros")
for i, r in df_17.iterrows():
    tot = r["Total"]
    axes[0].text(tot + 150, i, f"{tot:,}", va="center", fontsize=7.5)
axes[0].set_xlim(0, max(df_17["Total"]) * 1.18)

sns.scatterplot(data=df_17, x="Aforo", y="Precio", hue="Dimensión", palette=palette, s=180, ax=axes[1])
axes[1].set_title("Precio Mediano vs. Aforo Medio (%)", fontsize=10, fontweight="bold")
axes[1].set_xlabel("Peso Aforo Medio (%)")
axes[1].set_ylabel("Mediana Precio COP")
for i, r in df_17.iterrows():
    axes[1].annotate(r["Tag"], (r["Aforo"], r["Precio"]), xytext=(5, 3), textcoords="offset points", fontsize=7.5,
                     fontweight="bold" if r["Tag"] in ["PALCO", "VIP", "PLATEA", "PREFERENCIAL", "GENERAL", "BALCON", "PISO_ALTO"] else "normal")
plt.suptitle("Evaluación de las 17 Variables NLP en 4 Dimensiones", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/figures/fig3a_demostracion_17tags.png", dpi=200)
plt.close()

# 6. FIG 3: Boxplots Bivariados 7 Tags
fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
tag_data = []
for tag in ["tag_palco", "tag_vip", "tag_preferencial", "tag_platea", "tag_general", "tag_balcon", "tag_piso_alto"]:
    temp = df_enriquecido[df_enriquecido[tag] == 1].copy()
    temp["tag_name"] = tag.replace("tag_", "").upper()
    tag_data.append(temp)
df_tags_long = pd.concat(tag_data, ignore_index=True)
sns.boxplot(data=df_tags_long, x="tag_name", y="ratio_precio_max", hue="tag_name", palette="Set2", legend=False, ax=axes[0])
axes[0].set_title("Ratio de Precio por Tag NLP", fontsize=10, fontweight="bold")
axes[0].tick_params(axis="x", rotation=30)
sns.boxplot(data=df_tags_long, x="tag_name", y="peso_aforo", hue="tag_name", palette="Set2", legend=False, ax=axes[1])
axes[1].set_title("Peso de Aforo (%) por Tag NLP", fontsize=10, fontweight="bold")
axes[1].tick_params(axis="x", rotation=30)
plt.suptitle("Comparación Bivariada de Precio y Aforo por Atributo Estructural", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/figures/fig3b_boxplots_bivariados.png", dpi=200)
plt.close()

# 7. FIG 4: Correlaciones
corr_cols = ["med_unit_amt_itx", "ratio_precio_max", "ratio_precio_mean", "percentil_precio_evento", "dn_quota", "peso_aforo", "tasa_ocupacion", "tasa_venta_paga", "ratio_cortesias"]
plt.figure(figsize=(9, 7))
sns.heatmap(df_enriquecido[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", cbar=True, vmin=-1, vmax=1, annot_kws={"size": 8.5})
plt.title("Matriz de Correlación de Variables Estructurales y Rendimiento", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/figures/fig4_correlaciones.png", dpi=200)
plt.close()

# 8. FIG 5: Cuadrantes de Demanda
plt.figure(figsize=(9, 6.5))
sample_df = df_enriquecido.sample(n=min(5000, len(df_enriquecido)), random_state=42)
sc = plt.scatter(sample_df["peso_aforo"], sample_df["ratio_precio_max"], c=sample_df["tasa_ocupacion"], cmap="viridis", alpha=0.6, s=25)
plt.colorbar(sc, label="Tasa de Ocupación")
plt.axvline(x=0.25, color="red", linestyle="--", alpha=0.7)
plt.axhline(y=0.5, color="red", linestyle="--", alpha=0.7)
plt.title("Mapa de Separabilidad Espacial (Cuadrantes de Demanda)", fontsize=11, fontweight="bold")
plt.xlabel("Peso de Aforo (dn_quota / performance_quota)")
plt.ylabel("Ratio de Precio (vs Máximo del Evento)")
plt.tight_layout()
plt.savefig("reports/figures/fig5_cuadrantes_demanda.png", dpi=200)
plt.close()

print("All 8 figures generated and saved in reports/figures/ successfully!")
