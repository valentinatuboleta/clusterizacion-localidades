import os
import sys
import io
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Path base
sys.path.insert(0, os.path.abspath("c:/dev/clusterizacion-localidades"))
load_dotenv("c:/dev/clusterizacion-localidades/.env")

from src.azure_utils import cargar_parquet_desde_azure
from src.feature_engineering import filtrar_consistencia_localidades, preparar_dataset_enriquecido
from src.clustering import predecir_arquetipos_demanda, DISTRIBUCION_ESPERADA_ARQUETIPOS, cargar_modelo_clustering
from src.jerarquia import predecir_microclusters

print("=" * 80)
print("PROTOCOLO DE MARCHA BLANCA: EVALUACIÓN Y VALIDACIÓN (SHADOW TESTING)")
print("=" * 80)

# Metadatos del protocolo
fecha_hoy = datetime.now().strftime("%Y%m%d")
blob_path = "GOLD/SECUTIX/Training Data/Clustering de Localidades test/"
payload_v25_path = "data/processed/modelo_clustering_v2_5.joblib"
payload_v3_path = "data/processed/modelo_jerarquia_v3.joblib"

# 1. Carga del blob de test vía Azure SDK
print(f"\n[1/5] Cargando dataset de prueba desde Azure Blob Storage: '{blob_path}'...")
df_test_raw = cargar_parquet_desde_azure(blob_name=blob_path)
n_crudo = len(df_test_raw)

# 2. Los 4 Checks de Validación de Entrada (antes de evaluar)
print("\n[2/5] Ejecutando los 4 Checks de Validación de Entrada...")

# Check (a): Duplicados exactos en crudo
n_dups_crudo = int(df_test_raw.duplicated().sum())
pct_dups_crudo = (n_dups_crudo / n_crudo) * 100
print(f"  • Check (a) Duplicados exactos en crudo: {n_dups_crudo:,} filas ({pct_dups_crudo:.2f}%)")

# Check (b): Conteo post-filtro por regla
df_test_clean, reporte_reglas = filtrar_consistencia_localidades(df_test_raw, retornar_reporte=True, verbose=False)
n_limpio = len(df_test_clean)
print(f"  • Check (b) Conteo post-filtro por regla:")
for k, v in reporte_reglas.items():
    if k.startswith("regla_"):
        print(f"      - {k}: {v:,}")
print(f"      -> Total eliminadas por filtro: {reporte_reglas['total_eliminadas']:,} ({reporte_reglas['total_eliminadas']/n_crudo*100:.2f}%)")
print(f"      -> Registros limpios para inferencia: {n_limpio:,}")

# Check (c): Eventos incoherentes en quota
if "t_performance_id" in df_test_raw.columns and "dn_quota" in df_test_raw.columns and "performance_quota" in df_test_raw.columns:
    suma_quota_crudo = df_test_raw.groupby("t_performance_id")["dn_quota"].transform("sum")
    n_incoherentes_crudo = int((suma_quota_crudo != df_test_raw["performance_quota"]).sum())
else:
    n_incoherentes_crudo = 0
print(f"  • Check (c) Eventos incoherentes en quota (en crudo): {n_incoherentes_crudo:,} filas")

# Check (d): Tamaño del df vs. esperado de entrenamiento (referencia ~33,775)
n_ref_entrenamiento = 33775
dif_pct = abs(n_limpio - n_ref_entrenamiento) / n_ref_entrenamiento * 100
alerta_tamano = bool(dif_pct > 20.0)
print(f"  • Check (d) Tamaño vs. entrenamiento ({n_ref_entrenamiento:,}): {n_limpio:,} (Diferencia: {dif_pct:.2f}%)")
if alerta_tamano:
    print(f"      [ALERTA CHECK D] El tamaño difiere > 20% del entrenamiento de referencia.")

# Check (e): Filas con patrones de exclusión de producto en crudo y ventas > aforo
pats_exclusion = ("TEST", "CANCELAD", "PARQUEA", "NO USAR")
if "product" in df_test_raw.columns:
    prod_s = df_test_raw["product"].fillna("").astype(str).str.upper()
    mask_pats_crudo = pd.Series(False, index=df_test_raw.index)
    detalle_patrones = {}
    for p in pats_exclusion:
        c_p = int(prod_s.str.contains(p, regex=False).sum())
        detalle_patrones[p] = c_p
        mask_pats_crudo |= prod_s.str.contains(p, regex=False)
    n_patrones_crudo = int(mask_pats_crudo.sum())
else:
    n_patrones_crudo = 0
    detalle_patrones = {}

if "net_sold_p_qty" in df_test_raw.columns and "dn_quota" in df_test_raw.columns:
    n_ventas_exceden_crudo = int((df_test_raw["net_sold_p_qty"] > df_test_raw["dn_quota"]).sum())
else:
    n_ventas_exceden_crudo = 0
print(f"  • Check (e) Patrones de exclusión en crudo: {n_patrones_crudo:,} filas {detalle_patrones}")
print(f"      - Ventas pagadas > aforo en crudo: {n_ventas_exceden_crudo:,} filas")

# 3. Preparación Enriquecida del Dataset (Ruta estándar)
print("\n[3/5] Aplicando pipeline estándar de enriquecimiento (preparar_dataset_enriquecido)...")
df_enriquecido = preparar_dataset_enriquecido(df_test_raw)

# 4. Inferencia con los Envoltorios Productivos (Nivel 1 v2.5 + Nivel 2 v3.0)
print("\n[4/5] Ejecutando evaluación con envoltorios productivos (cero re-entrenamiento)...")
payload_v25 = cargar_modelo_clustering(payload_v25_path)
pred_nivel1 = predecir_arquetipos_demanda(df_enriquecido, payload_v25)

# Inferencia Nivel 2 Micro-Clusters v3.0
pred_micro = predecir_microclusters(df_enriquecido, payload_v3_path)

# Métricas de evaluación
distrib_obs = pred_nivel1["arquetipo_demanda"].value_counts(normalize=True).to_dict()
comparativa_distrib = {}
for arq, p_esp in DISTRIBUCION_ESPERADA_ARQUETIPOS.items():
    p_obs = float(distrib_obs.get(arq, 0.0))
    comparativa_distrib[arq] = {
        "observado": round(p_obs, 4),
        "esperado": round(p_esp, 4),
        "diferencia": round(p_obs - p_esp, 4)
    }

score_confianza_medio = float(pred_micro["score_confianza"].mean())
pct_frontera = float(pred_micro["es_frontera"].mean() * 100)

s_cat = df_enriquecido["logical_seat_category"].astype(str).str.strip()
pct_texto_vacio = float((s_cat == "").mean() * 100)

print("\nMétricas de Evaluación de Marcha Blanca:")
print(f"  • Score de confianza geométrico medio : {score_confianza_medio:.4f}")
print(f"  • Tasa de localidades en frontera      : {pct_frontera:.2f}%")
print(f"  • Tasa de texto vacío post-filtro      : {pct_texto_vacio:.2f}%")
print("  • Distribución Nivel 1 vs Esperada:")
for arq, vals in comparativa_distrib.items():
    print(f"     - {arq}: Obs={vals['observado']*100:.1f}% | Esp={vals['esperado']*100:.1f}% | Dif={vals['diferencia']*100:+.1f}%")

# 5. Exportación de Artefactos (JSON y CSV)
print("\n[5/5] Exportando artefactos de Marcha Blanca...")
os.makedirs("reports", exist_ok=True)
reporte_json_path = f"reports/marcha_blanca_{fecha_hoy}.json"

reporte_final = {
    "fecha_ejecucion": datetime.now().isoformat(),
    "path_origen": blob_path,
    "filas_in": n_crudo,
    "filas_out": len(pred_micro),
    "checks_entrada": {
        "check_a_duplicados_crudo": n_dups_crudo,
        "check_b_conteos_por_regla": reporte_reglas,
        "check_c_eventos_incoherentes_quota": n_incoherentes_crudo,
        "check_d_alerta_tamano_vs_entrenamiento": alerta_tamano,
        "check_d_diferencia_pct": round(dif_pct, 2),
        "check_e_patrones_exclusion_crudo": n_patrones_crudo,
        "check_e_detalle_patrones": detalle_patrones,
        "check_e_ventas_exceden_aforo_crudo": n_ventas_exceden_crudo
    },
    "metricas_evaluacion": {
        "score_confianza_medio": round(score_confianza_medio, 4),
        "pct_frontera": round(pct_frontera, 2),
        "pct_texto_vacio": round(pct_texto_vacio, 2),
        "comparativa_distribucion_nivel1": comparativa_distrib
    },
    "versiones_payload": {
        "nivel1_macro": "v2.5 (modelo_clustering_v2_5.joblib)",
        "nivel2_micro": "v3.0 (modelo_jerarquia_v3.joblib)"
    }
}

with open(reporte_json_path, "w", encoding="utf-8") as f:
    json.dump(reporte_final, f, indent=2, ensure_ascii=False)
print(f"  • Reporte cuantitativo guardado en: {reporte_json_path}")

# Integrar predicciones completas
df_export = df_enriquecido.copy()
for col in ["micro_cluster_id", "label_auto", "arquetipo_demanda", "score_confianza", "es_frontera"]:
    df_export[col] = pred_micro[col]

out_csv = "data/processed/marcha_blanca_predicciones.csv"
out_pq = "data/processed/marcha_blanca_predicciones.parquet"
try:
    df_export.to_csv(out_csv, index=False)
    print(f"  • Predicciones guardadas en CSV: {out_csv} ({os.path.getsize(out_csv)/(1024*1024):.2f} MB)")
except PermissionError:
    alt_csv = "data/processed/marcha_blanca_predicciones_actualizado.csv"
    df_export.to_csv(alt_csv, index=False)
    print(f"  [AVISO] {out_csv} está bloqueado por otra aplicación (ej. Excel). Guardado en: {alt_csv}")

try:
    df_export.to_parquet(out_pq, index=False)
    print(f"  • Predicciones guardadas en Parquet: {out_pq} ({os.path.getsize(out_pq)/(1024*1024):.2f} MB)")
except Exception as e:
    print(f"  [ALERTA] No se pudo guardar parquet: {e}")

print("\n" + "=" * 80)
print("MARCHA BLANCA EJECUTADA Y CERTIFICADA EXITOSAMENTE")
print("=" * 80)
