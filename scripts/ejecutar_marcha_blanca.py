#!/usr/bin/env python3
"""
Script de Marcha Blanca (Shadow Testing) para la Clusterización Jerárquica v3.0.

Este script ejecuta la inferencia de 2 niveles (Nivel 1 Macro + Nivel 2 Micro)
sobre un conjunto de prueba de nuevos eventos curados por el pipeline de raw data.

Evalúa:
1. Asignación a los 19 micro-clusters canónicos y 6 macro-arquetipos.
2. Deriva estadística poblacional (Population Stability Index - PSI) frente a la línea base histórica.
3. Proporción de localidades en zona de frontera (incertidumbre de decisión).
4. Exportación de predicciones listas para consumo por modelos downstream (Pricing, Forecasting).

Uso CLI:
    python scripts/ejecutar_marcha_blanca.py --input data/raw/localidades_test.parquet
    python scripts/ejecutar_marcha_blanca.py --input data/raw/localidades_test.parquet --modelo data/processed/modelo_jerarquia_v3.joblib --output data/processed/marcha_blanca_predicciones.parquet
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib

# Asegurar path raíz del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

from src.jerarquia import (
    predecir_microclusters,
    evaluar_drift_microclusters,
    MICRO_CLUSTERS_CANONICAL,
    ARQ_PREFIX
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de Marcha Blanca para el Modelo de Micro-Clusters (v3.0) TuBoleta."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Ruta local o WASBS/DBFS al archivo parquet/csv curado de localidades de prueba."
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default="data/processed/modelo_jerarquia_v3.joblib",
        help="Ruta al artefacto del modelo jerárquico serializado (.joblib)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/marcha_blanca_predicciones.parquet",
        help="Ruta de destino para persistir las predicciones enriquecidas."
    )
    parser.add_argument(
        "--reporte_json",
        type=str,
        default="reports/marcha_blanca_reporte.json",
        help="Ruta de exportación del resumen cuantitativo de marcha blanca (JSON)."
    )
    parser.add_argument(
        "--umbral_psi",
        type=float,
        default=0.10,
        help="Umbral de alerta de PSI para detectar drift poblacional (por defecto 0.10)."
    )
    return parser.parse_args()


def cargar_datos_prueba(input_path: str) -> pd.DataFrame:
    """Carga el dataset curado de localidades soportando parquet o csv."""
    print(f"\n[1/5] Cargando dataset de prueba desde: {input_path}")
    if input_path.endswith(".parquet") or input_path.endswith(".pq"):
        df = pd.read_parquet(input_path)
    elif input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        # Intento por defecto como parquet
        try:
            df = pd.read_parquet(input_path)
        except Exception:
            df = pd.read_csv(input_path)
    print(f" -> Registros cargados: {len(df):,} | Columnas: {len(df.columns)}")
    return df


def ejecutar_marcha_blanca(
    input_path: str,
    modelo_path: str = "data/processed/modelo_jerarquia_v3.joblib",
    output_path: Optional[str] = "data/processed/marcha_blanca_predicciones.parquet",
    reporte_json_path: Optional[str] = "reports/marcha_blanca_reporte.json",
    umbral_psi: float = 0.10
) -> Dict[str, Any]:
    """
    Ejecuta el flujo completo de evaluación de marcha blanca.
    """
    t0 = time.time()
    print("=" * 70)
    print("PIPELINE DE EVALUACIÓN DE MARCHA BLANCA - TUBOLETA ML (v3.0)")
    print("=" * 70)

    # 1. Cargar datos
    df_test = cargar_datos_prueba(input_path)

    # 2. Cargar modelo jerárquico
    print(f"\n[2/5] Cargando artefacto jerárquico v3.0 desde: {modelo_path}")
    if not os.path.exists(modelo_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {modelo_path}")
    payload_hier = joblib.load(modelo_path)
    version = payload_hier.get("version", "desconocida")
    print(f" -> Modelo jerárquico cargado exitosamente (Versión: {version})")

    # 3. Inferencia jerárquica
    print("\n[3/5] Ejecutando inferencia jerárquica de 2 niveles (Nivel 1 Macro + Nivel 2 Micro)...")
    df_pred = predecir_microclusters(df_test, payload_hier)

    # Consolidar columnas entregables
    df_resultado = df_test.copy()
    for col in ["micro_cluster_id", "label_auto", "arquetipo_demanda", "score_confianza", "es_frontera"]:
        df_resultado[col] = df_pred[col]

    # 4. Diagnóstico y métricas de control
    print("\n[4/5] Evaluando métricas de control de marcha blanca...")
    n_total = len(df_resultado)
    
    # A. Distribución por macro-arquetipo
    dist_macro = (
        df_resultado["arquetipo_demanda"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_dict()
    )

    # B. Distribución por micro-cluster
    conteo_micro = df_resultado["micro_cluster_id"].value_counts().to_dict()
    dist_micro = {
        k: {
            "n": int(v),
            "pct": round(float(v / n_total * 100), 2)
        }
        for k, v in sorted(conteo_micro.items())
    }

    # C. Evaluación de Drift (PSI)
    res_drift = evaluar_drift_microclusters(df_resultado, payload_hier, umbral_psi=umbral_psi)
    psi_global = res_drift.get("psi_global", 0.0)
    estado_drift = res_drift.get("estado", "DESCONOCIDO")

    # D. Tasa de incertidumbre (fronteras)
    tasa_frontera = float(df_resultado["es_frontera"].mean() * 100)
    confianza_media = float(df_resultado["score_confianza"].mean())

    # E. Resumen visual en consola
    print("\n" + "-" * 50)
    print("RESUMEN DE ASIGNACIÓN MACRO (NIVEL 1):")
    print("-" * 50)
    for arq, pct in dist_macro.items():
        print(f"  • {arq:<40}: {pct:>6.2f}%")

    print("\n" + "-" * 50)
    print(f"EVALUACIÓN DE DRIFT POBLACIONAL (PSI):")
    print("-" * 50)
    print(f"  • PSI Global                  : {psi_global:.4f}")
    print(f"  • Estado de Estabilidad       : {estado_drift}")
    print(f"  • Tasa de Frontera (Incertidumbre): {tasa_frontera:.2f}%")
    print(f"  • Score de Confianza Promedio : {confianza_media:.4f}")

    if res_drift.get("alertas"):
        print("\n  [ALERTAS DETECTADAS]:")
        for alerta in res_drift["alertas"]:
            print(f"    - {alerta}")
    else:
        print("\n  [OK] Sin alertas críticas de deriva estadística.")

    # 5. Persistencia
    print(f"\n[5/5] Persistiendo resultados de marcha blanca...")
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if output_path.endswith(".parquet") or output_path.endswith(".pq"):
            df_resultado.to_parquet(output_path, index=False)
        else:
            df_resultado.to_csv(output_path, index=False)
        print(f" -> Predicciones guardadas en: {output_path}")

    # Guardar reporte JSON
    reporte_final = {
        "fecha_ejecucion": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tiempo_segundos": round(time.time() - t0, 2),
        "total_registros": n_total,
        "version_modelo": version,
        "psi_global": psi_global,
        "estado_drift": estado_drift,
        "umbral_psi": umbral_psi,
        "tasa_frontera_pct": round(tasa_frontera, 2),
        "confianza_media": round(confianza_media, 4),
        "distribucion_macro_pct": dist_macro,
        "distribucion_micro": dist_micro,
        "alertas": res_drift.get("alertas", [])
    }

    if reporte_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(reporte_json_path)), exist_ok=True)
        with open(reporte_json_path, "w", encoding="utf-8") as f:
            json.dump(reporte_final, f, indent=2, ensure_ascii=False)
        print(f" -> Reporte analítico guardado en: {reporte_json_path}")

    print("\n" + "=" * 70)
    print(f"MARCHA BLANCA FINALIZADA CON ÉXITO en {time.time() - t0:.2f}s")
    print("=" * 70)

    return {
        "df_resultado": df_resultado,
        "reporte": reporte_final
    }


if __name__ == "__main__":
    args = parse_arguments()
    ejecutar_marcha_blanca(
        input_path=args.input,
        modelo_path=args.modelo,
        output_path=args.output,
        reporte_json_path=args.reporte_json,
        umbral_psi=args.umbral_psi
    )
