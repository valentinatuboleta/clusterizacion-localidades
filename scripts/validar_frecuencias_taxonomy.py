"""
Script Wrapper CLI de Validación de Compuerta de Frecuencias de Taxonomía de Venues v2 (Bloque 1.5).

Fuente de verdad analítica: `src.validar_frecuencias` (módulo reutilizable con pruebas herméticas).
Este script actúa exclusivamente como wrapper de ejecución CLI para generación y reporte en consola/disco.

Genera:
1. reports/frecuencias_type_site_v2.csv
2. Tabla impresa con conteos y porcentajes a nivel localidad y a nivel venue único.
3. Comparativa de migración v1 -> v2 (origen vs destino).
4. Verificación de flags automáticos (categoría vacía, cold start < 50, dominancia > 60%).
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from src.validar_frecuencias import calcular_reporte_frecuencias
from src.nlp_utils import normalizar_venue
from src.clustering import MAPEO_TAXONOMIA_V1_A_V2, CANONICAL_TYPE_SITE_CATEGORIES


def main():
    ruta_raw = "data/raw/localidades_eda.parquet"
    ruta_lookup = "data/lookup/site_type_lookup.csv"
    ruta_reporte = "reports/frecuencias_type_site_v2.csv"

    if not os.path.exists(ruta_raw):
        raise FileNotFoundError(f"No existe {ruta_raw}")
    if not os.path.exists(ruta_lookup):
        raise FileNotFoundError(f"No existe {ruta_lookup}")

    print("Cargando dataset crudo y lookup migrado v2...")
    df_raw = pd.read_parquet(ruta_raw)
    df_lookup = pd.read_csv(ruta_lookup)

    df_frec, diag = calcular_reporte_frecuencias(df_raw, df_lookup, site_column="site")

    os.makedirs(os.path.dirname(ruta_reporte), exist_ok=True)
    df_frec.to_csv(ruta_reporte, index=False)
    print(f"\n[OK] Reporte guardado en {ruta_reporte}")

    print("\n" + "=" * 90)
    print("           COMPUERTA DE VALIDACION DE FRECUENCIAS TAXONOMIA VENUE v2 (BLOQUE 1.5)")
    print("=" * 90)
    print(f"{'Categoria':<16} | {'Locs':<8} | {'% Locs':<8} | {'Sites':<6} | {'Vacia':<6} | {'ColdStart':<9} | {'Dominancia':<10}")
    print("-" * 90)
    for _, row in df_frec.iterrows():
        vacia_str = "SI" if row["flag_vacia"] else "NO"
        cs_str = "SI (<50)" if row["flag_cold_start"] else "NO"
        dom_str = "SI (>60%)" if row["flag_dominancia"] else "NO"
        print(f"{row['categoria']:<16} | {row['conteo_localidades']:<8d} | {row['porcentaje_localidades']:<7.2f}% | {row['conteo_sites_unicos']:<6d} | {vacia_str:<6} | {cs_str:<9} | {dom_str:<10}")
    print("=" * 90)

    # Comparativa v1 -> v2 (origen taxonomico de venues)
    print("\n=== RESUMEN DE MIGRACION v1 -> v2 (DISTRIBUCION DE VENUES EN LOOKUP) ===")
    print(f"{'Categoria v2':<16} | {'Total Venues':<12}")
    print("-" * 35)
    for cat, cnt in df_lookup["type_site"].value_counts().items():
        print(f"{cat:<16} | {cnt:<12d}")
    print("-" * 35)

    print("\n=== DIAGNOSTICO DE FLAGS AUTOMATICOS ===")
    print(f"Total Localidades evaluadas: {diag['total_localidades']:,}")
    print(f"Total Venues en Lookup:      {diag['total_sites_lookup']:,}")
    print(f"Flags Rojos (Parada):        {diag['flags_rojos']}")
    print(f"Flags Amarillos (Informativo): {diag['flags_amarillos']}")
    print(f"Pasa compuerta limpia:       {'SI' if diag['pasa_compuerta'] else 'NO (Requiere evaluacion)'}")

    return df_frec, diag


if __name__ == "__main__":
    main()
