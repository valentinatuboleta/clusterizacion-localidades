#!/usr/bin/env python3
"""
Script de Monitoreo Periodico de Drift (PSI y Distribucion de Arquetipos).

Este script evalua la estabilidad estadistica de nuevos lotes de localidades
frente a la referencia historica del modelo de clusterizacion v2.2.

Metricas calculadas:
1. Population Stability Index (PSI) por variable estructural.
2. Comparativa de distribucion de arquetipos observada vs esperada.
3. Proporcion de localidades en frontera de decision (score_confianza < 0.15).
4. Proporcion de localidades con texto fuera de vocabulario (cobertura < 0.20).

Uso:
    python scripts/monitorear_drift.py --datos data/raw/lote_nuevo.parquet
    python scripts/monitorear_drift.py --datos data/raw/lote_nuevo.parquet --output reports/drift_report.json
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

# Asegurar path raiz del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clustering import cargar_modelo_clustering, evaluar_drift_lote
from src.feature_engineering import filtrar_consistencia_localidades, calcular_metricas_relativas
from src.nlp_utils import pipeline_procesamiento_nlp


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auditoria de drift estadistico (PSI) para el modelo de clusterizacion de TuBoleta."
    )
    parser.add_argument(
        "--datos",
        type=str,
        required=True,
        help="Ruta al archivo parquet o csv con el lote de localidades a evaluar."
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default="data/processed/modelo_clustering_v2_2.joblib",
        help="Ruta al artefacto del modelo serializado (.joblib)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta opcional para exportar el reporte en formato JSON."
    )
    parser.add_argument(
        "--umbral_psi",
        type=float,
        default=0.25,
        help="Umbral de PSI para activar alerta de drift critico (por defecto: 0.25)."
    )
    return parser.parse_args()


def imprimir_reporte_consola(reporte: dict, umbral_psi: float):
    sep = "=" * 80
    subsep = "-" * 80
    
    print("\n" + sep)
    print(" REPORTE DE AUDITORIA DE DRIFT ESTADISTICO (POPULATION STABILITY INDEX - PSI)")
    print(sep)
    print(f" Total de localidades analizadas: {reporte['total_registros']:,}")
    print(f" Estado General del Lote:         [{reporte['estado_general']}]")
    print(f" PSI Maximo Observado:            {reporte['psi_maximo']:.4f} (Umbral Alerta: {umbral_psi:.2f})")
    print(f" Score de Confianza Promedio:     {reporte['score_confianza_promedio']:.4f}")
    print(f" Localidades en Frontera (<0.15): {reporte['pct_frontera']:.2%}")
    print(f" Localidades Texto Casi Vacio:    {reporte['pct_texto_casi_vacio']:.2%}")
    print(subsep)

    print("\n1. ESTABILIDAD DE VARIABLES DE ENTRADA (PSI):")
    print(f" {'Variable':<28} | {'PSI':<8} | {'Estado':<16} | Interpretacion")
    print(f" {'-'*28} | {'-'*8} | {'-'*16} | {'-'*20}")
    for var, data in reporte["psi_por_variable"].items():
        psi = data["psi"]
        estado = data["estado"]
        if estado == "ESTABLE":
            interp = "Sin variacion significativa"
        elif estado == "REVISAR":
            interp = "Variacion moderada"
        else:
            interp = "DRIFT CRITICO - RE-ENTRENAR"
        print(f" {var:<28} | {psi:<8.4f} | [{estado:<14}] | {interp}")

    print("\n2. DISTRIBUCION DE ARQUETIPOS (OBSERVADA VS ESPERADA):")
    print(f" {'Arquetipo':<38} | {'Esperado':<9} | {'Observado':<9} | {'Diferencia':<10}")
    print(f" {'-'*38} | {'-'*9} | {'-'*9} | {'-'*10}")
    for arq, data in reporte["distribucion_arquetipos"].items():
        esp = data["esperado"]
        obs = data["observado"]
        diff = data["diferencia"]
        print(f" {arq:<38} | {esp:<9.2%} | {obs:<9.2%} | {diff:+9.2%}")

    if reporte["alertas"]:
        print("\n3. ALERTAS OPERATIVAS DETECTADAS:")
        for al in reporte["alertas"]:
            print(f"   * [ALERTA] {al}")
    else:
        print("\n3. ALERTAS OPERATIVAS DETECTADAS: Ninguna. Lote en parametros esperados.")

    print("\n" + sep + "\n")


def main():
    args = parse_arguments()

    if not os.path.exists(args.datos):
        print(f"Error: No se encontro el archivo de datos en: {args.datos}", file=sys.stderr)
        sys.exit(2)

    if not os.path.exists(args.modelo):
        print(f"Error: No se encontro el artefacto del modelo en: {args.modelo}", file=sys.stderr)
        sys.exit(2)

    # Cargar datos
    if args.datos.endswith(".parquet"):
        df = pd.read_parquet(args.datos)
    else:
        df = pd.read_csv(args.datos)

    # Preparar features si vienen en crudo
    if "peso_aforo" not in df.columns or "ratio_precio_max" not in df.columns:
        df = filtrar_consistencia_localidades(df)
        df = calcular_metricas_relativas(df)

    if "texto_limpio" not in df.columns or "tag_palco" not in df.columns:
        col_nlp = next(
            (c for c in ["logical_seat_category", "product", "translation_name", "cd_name", "nombre_localidad"] if c in df.columns),
            None
        )
        if col_nlp:
            df = pipeline_procesamiento_nlp(df, col_nombre=col_nlp)

    # Evaluar drift
    reporte = evaluar_drift_lote(df, args.modelo, umbral_psi_alerta=args.umbral_psi)

    # Imprimir consola
    imprimir_reporte_consola(reporte, args.umbral_psi)

    # Exportar si se solicito
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(reporte, fp, indent=2, ensure_ascii=False)
        print(f"Reporte exportado exitosamente a: {args.output}")

    # Retorno de codigo de salida
    if reporte["estado_general"] == "DRIFT_CRITICO":
        print("[CRITICO] Se detecto drift critico en el lote. Se recomienda revision y re-entrenamiento.")
        sys.exit(1)
    else:
        print("[OK] Auditoria finalizada. Lote dentro de parametros operativos.")
        sys.exit(0)


if __name__ == "__main__":
    main()
