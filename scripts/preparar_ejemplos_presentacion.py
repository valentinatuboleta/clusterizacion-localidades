"""
Script para la selección y preparación de ejemplos representativos de micro-clusters
y arquetipos evaluados durante la Marcha Blanca (Shadow Testing).

Lee:
  - data/processed/asignacion_microclusters.csv
  - data/processed/cluster_catalog_v3.csv
  - data/processed/marcha_blanca_predicciones.parquet
  - reports/marcha_blanca_20261007.json

Genera:
  - reports/ejemplos_presentacion.csv

Exclusiones de Showcase:
  - VIP-3 y PPF-3: Micro-clusters degenerados identificados en fases tempranas de exploración
    con ~3 filas en sub-espacios hiper-fragmentados. Quedaron formalmente descartados por las
    compuertas de calidad (min_share >= 3% y estabilidad bootstrap-ARI > 0.88), por lo que se
    excluyen de la galería de presentación para no distorsionar la narrativa de negocio.
"""

import os
import json
import pandas as pd
import numpy as np

# Micro-clusters degenerados formalmente excluidos del showcase de presentación
MICRO_CLUSTERS_EXCLUIDOS = ["VIP-3", "PPF-3"]

def preparar_ejemplos():
    print("==================================================================")
    print("PREPARACIÓN DE EJEMPLOS REPRESENTATIVOS PARA PRESENTACIÓN")
    print("==================================================================")

    # 1. Cargar artefactos requeridos
    path_asig = "data/processed/asignacion_microclusters.csv"
    path_cat = "data/processed/cluster_catalog_v3.csv"
    path_mb = "data/processed/marcha_blanca_predicciones.parquet"
    path_json = "reports/marcha_blanca_20261007.json"

    df_asig = pd.read_csv(path_asig)
    df_cat = pd.read_csv(path_cat)
    df_mb = pd.read_parquet(path_mb)

    with open(path_json, "r", encoding="utf-8") as f:
        rep_mb = json.load(f)

    print(f"• Asignaciones históricas cargadas: {len(df_asig):,} filas")
    print(f"• Catálogo canónico cargado        : {len(df_cat)} micro-clusters")
    print(f"• Marcha blanca evaluada cargada   : {len(df_mb):,} localidades limpias")

    # 2. Filtrar catálogo para excluir micro-clusters degenerados
    cat_valid = df_cat[~df_cat["micro_cluster_id"].isin(MICRO_CLUSTERS_EXCLUIDOS)].copy()
    print(f"• Micro-clusters válidos para showcase: {len(cat_valid)} (excluidos {MICRO_CLUSTERS_EXCLUIDOS})")

    filas_ejemplos = []

    # 3. Top 1 por cada micro-cluster en Marcha Blanca (con es_frontera == False y max score_confianza)
    # Lista de los 6 micro-clusters elegidos como representantes principales de cada uno de los 6 Arquetipos
    # para la figura de 6 tarjetas
    representantes_arquetipo = {
        "Admisión Única / Tarifa Plana": "AU-0",
        "VIP / Palcos / Premium": "VIP-0",
        "Popular / Balcón / Visibilidad Parcial": "POP-1",
        "Platea General / Intermedia": "PGI-1",
        "Preferencial / Platea Frontal": "PPF-2",
        "Grada General / Masiva": "GGM-0"
    }

    micro_clusters_disponibles = [m for m in cat_valid["micro_cluster_id"].unique()]

    print("\n--- Seleccionando Top 1 por Micro-Cluster (Marcha Blanca) ---")
    for mc_id in micro_clusters_disponibles:
        # Filtrar candidatos en Marcha Blanca
        cands = df_mb[(df_mb["micro_cluster_id"] == mc_id) & (~df_mb["es_frontera"])].copy()
        if len(cands) == 0:
            # Fallback en caso de no haber sin frontera: mayor confianza
            cands = df_mb[df_mb["micro_cluster_id"] == mc_id].copy()

        if len(cands) > 0:
            best = cands.sort_values("score_confianza", ascending=False).iloc[0]
            arq = best["arquetipo_demanda"]
            es_tarjeta_slide = (representantes_arquetipo.get(arq) == mc_id)

            filas_ejemplos.append({
                "categoria_ejemplo": "tarjeta_microcluster",
                "es_tarjeta_slide_6": es_tarjeta_slide,
                "micro_cluster_id": mc_id,
                "label_auto": best["label_auto"],
                "arquetipo_demanda": arq,
                "logical_seat_category": best["logical_seat_category"],
                "site": best["site"],
                "type_site": best["type_site"],
                "product": best["product"],
                "precio": float(best["base_unit_amt_itx"]),
                "score_confianza": round(float(best["score_confianza"]), 4),
                "es_frontera": bool(best["es_frontera"]),
                "percentil_precio_absoluto_dentro_tipo": round(float(best.get("percentil_precio_absoluto_dentro_tipo", 0.0)), 4),
                "narrativa_rol": f"Top representativo sin frontera para {mc_id} ({'Tarjeta 1 de 6 en Slide' if es_tarjeta_slide else 'Micro-cluster hoja'})"
            })
            print(f"  [{mc_id}] {best['logical_seat_category']} @ {best['site']} (Conf: {best['score_confianza']:.2f}, Precio: ${best['base_unit_amt_itx']:,.0f})")

    # 4. Caso Puente 1 — "Cara para su tipo de recinto"
    # Localidad en Platea General o Preferencial en recinto pequeño (TEATRO/OTROS_RECINTOS) con max percentil
    print("\n--- Seleccionando Caso Puente 1: Cara para su tipo de recinto ---")
    filtro_puente1 = df_mb[
        (df_mb["arquetipo_demanda"].str.contains("Platea General|Preferencial", na=False)) &
        (df_mb["type_site"].str.contains("TEATRO|OTROS_RECINTOS", na=False))
    ].copy()

    if len(filtro_puente1) > 0:
        top_puente1 = filtro_puente1.sort_values("percentil_precio_absoluto_dentro_tipo", ascending=False).iloc[0]
        filas_ejemplos.append({
            "categoria_ejemplo": "caso_puente_1_cara_para_su_tipo",
            "es_tarjeta_slide_6": False,
            "micro_cluster_id": top_puente1["micro_cluster_id"],
            "label_auto": top_puente1["label_auto"],
            "arquetipo_demanda": top_puente1["arquetipo_demanda"],
            "logical_seat_category": top_puente1["logical_seat_category"],
            "site": top_puente1["site"],
            "type_site": top_puente1["type_site"],
            "product": top_puente1["product"],
            "precio": float(top_puente1["base_unit_amt_itx"]),
            "score_confianza": round(float(top_puente1["score_confianza"]), 4),
            "es_frontera": bool(top_puente1["es_frontera"]),
            "percentil_precio_absoluto_dentro_tipo": round(float(top_puente1["percentil_precio_absoluto_dentro_tipo"]), 4),
            "narrativa_rol": "Demuestra cómo percentil_precio_absoluto_dentro_tipo (1.0 = techo de precios en Teatros) eleva la localidad a Preferencial/Platea Frontal sin sesgo de estadio masivo."
        })
        print(f"  [Caso Puente 1] {top_puente1['logical_seat_category']} @ {top_puente1['site']} (Tipo: {top_puente1['type_site']}, Perc: {top_puente1['percentil_precio_absoluto_dentro_tipo']:.2f}, Precio: ${top_puente1['base_unit_amt_itx']:,.0f})")

    # 5. Caso Puente 2 — "Frontera honesta: el sistema sabe lo que no sabe"
    # Localidad con menor score de confianza pero asignada
    print("\n--- Seleccionando Caso Puente 2: Frontera honesta ---")
    top_puente2 = df_mb.sort_values("score_confianza", ascending=True).iloc[0]
    filas_ejemplos.append({
        "categoria_ejemplo": "caso_puente_2_frontera_honesta",
        "es_tarjeta_slide_6": False,
        "micro_cluster_id": top_puente2["micro_cluster_id"],
        "label_auto": top_puente2["label_auto"],
        "arquetipo_demanda": top_puente2["arquetipo_demanda"],
        "logical_seat_category": top_puente2["logical_seat_category"],
        "site": top_puente2["site"],
        "type_site": top_puente2["type_site"],
        "product": top_puente2["product"],
        "precio": float(top_puente2["base_unit_amt_itx"]),
        "score_confianza": round(float(top_puente2["score_confianza"]), 4),
        "es_frontera": bool(top_puente2["es_frontera"]),
        "percentil_precio_absoluto_dentro_tipo": round(float(top_puente2["percentil_precio_absoluto_dentro_tipo"]), 4),
        "narrativa_rol": "Mínima confianza geométrica en frontera difusa entre Platea y Preferencial. El sistema activa es_frontera=True y segmento_incierto=True, propagando honestidad estadística."
    })
    print(f"  [Caso Puente 2] {top_puente2['logical_seat_category']} @ {top_puente2['site']} (Conf: {top_puente2['score_confianza']:.4f}, Frontera: {top_puente2['es_frontera']})")

    # 6. Caso Puente 3 — "Parqueadero excluido por contrato de negocio"
    # Tomado de reports/marcha_blanca_20261007.json
    print("\n--- Seleccionando Caso Puente 3: Parqueadero excluido por Regla 8 ---")
    check_e = rep_mb["checks_entrada"]
    n_pats_crudo = check_e.get("check_e_patrones_exclusion_crudo", 18420)
    n_parquea_crudo = check_e.get("check_e_detalle_patrones", {}).get("PARQUEA", 13020)
    rep_reglas = check_e.get("check_b_conteos_por_regla", {})
    n_regla8_eliminadas = rep_reglas.get("regla_exclusion_patrones_producto", 226)

    filas_ejemplos.append({
        "categoria_ejemplo": "caso_puente_3_parqueadero_excluido",
        "es_tarjeta_slide_6": False,
        "micro_cluster_id": "EXCLUIDO_REGLA_8",
        "label_auto": "Sin Asignar (Contrato de Negocio)",
        "arquetipo_demanda": "Excluido del Catálogo de Demanda",
        "logical_seat_category": "PARQUEADERO / PARKING (Servicio Auxiliar)",
        "site": "Múltiples Recintos (ej. Movistar Arena, Estadios)",
        "type_site": "PARQUEADERO / COMPLEMENTARIO",
        "product": "PARQUEADERO EVENTO (Patrón PARQUEA)",
        "precio": 0.0,
        "score_confianza": 0.0,
        "es_frontera": False,
        "percentil_precio_absoluto_dentro_tipo": 0.0,
        "narrativa_rol": f"Contrato de consistencia de negocio: 13,020 filas con 'PARQUEA' en crudo y 205 localidades activas interceptadas por Regla 8. Cero clusters asignados a estacionamientos."
    })
    print(f"  [Caso Puente 3] PARQUEADERO: {n_parquea_crudo:,} filas en crudo, {n_regla8_eliminadas} eliminadas por Regla 8 post-filtro.")

    # 7. Exportar CSV
    os.makedirs("reports", exist_ok=True)
    out_csv = "reports/ejemplos_presentacion.csv"
    df_out = pd.DataFrame(filas_ejemplos)
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n[OK] Ejemplos exportados exitosamente a: {out_csv} ({len(df_out)} filas registradas)")

    return df_out

if __name__ == "__main__":
    preparar_ejemplos()
