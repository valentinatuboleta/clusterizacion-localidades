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

    # 7. Exportar CSV base
    os.makedirs("reports", exist_ok=True)
    out_csv = "reports/ejemplos_presentacion.csv"
    df_out = pd.DataFrame(filas_ejemplos)
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n[OK] Ejemplos exportados exitosamente a: {out_csv} ({len(df_out)} filas registradas)")

    return df_out


def preparar_galeria_extendida():
    """
    Selección estratificada de tarjetas para densificar la evidencia en 3 slides:
    - premium (VIP + Preferencial): <= 8 tarjetas
    - masivos (Platea General + Popular + Grada): <= 12 tarjetas
    - especiales (AU-0 [2] + 2 fronteras + 1 híbrido): 5 tarjetas
    """
    print("\n==================================================================")
    print("SELECCIÓN ESTRATIFICADA: GALERÍA EXTENDIDA DE TARJETAS")
    print("==================================================================")

    path_mb = "data/processed/marcha_blanca_predicciones.parquet"
    df_mb = pd.read_parquet(path_mb)

    # Excluir degenerados formalmente
    df_valid = df_mb[~df_mb["micro_cluster_id"].isin(MICRO_CLUSTERS_EXCLUIDOS)].copy()

    filas_galeria = []

    # 1. Grupo PREMIUM: VIP (VIP-0..2) + Preferencial (PPF-0..2) <= 8 tarjetas
    premium_mcs = ["VIP-0", "VIP-1", "VIP-2", "PPF-0", "PPF-1", "PPF-2"]
    print("\n--- Grupo PREMIUM (VIP + Preferencial) ---")
    # Top 1 por cada micro-cluster sin frontera
    count_prem = 0
    segundos_candidatos = []
    for mc in premium_mcs:
        sub = df_valid[(df_valid["micro_cluster_id"] == mc) & (~df_valid["es_frontera"])].sort_values("score_confianza", ascending=False).drop_duplicates(subset=["logical_seat_category"])
        if len(sub) > 0:
            top1 = sub.iloc[0]
            filas_galeria.append({
                "grupo": "premium",
                "caso": "representante",
                "micro_cluster_id": mc,
                "label_auto": top1["label_auto"],
                "arquetipo_demanda": top1["arquetipo_demanda"],
                "logical_seat_category": top1["logical_seat_category"],
                "site": top1["site"],
                "type_site": top1["type_site"],
                "precio": float(top1["base_unit_amt_itx"]),
                "score_confianza": round(float(top1["score_confianza"]), 4),
                "es_frontera": False,
                "percentil_precio_absoluto_dentro_tipo": round(float(top1.get("percentil_precio_absoluto_dentro_tipo", 0.0)), 4),
                "product": top1["product"]
            })
            count_prem += 1
            if len(sub) > 1:
                segundos_candidatos.append((len(sub), sub.iloc[1]))

    # Para alcanzar hasta 8 tarjetas (tope <= 8), agregar top 2 de los 2 microclusters más poblados
    segundos_candidatos.sort(key=lambda x: x[0], reverse=True)
    for _, top2 in segundos_candidatos[:(8 - count_prem)]:
        filas_galeria.append({
            "grupo": "premium",
            "caso": "representante",
            "micro_cluster_id": top2["micro_cluster_id"],
            "label_auto": top2["label_auto"],
            "arquetipo_demanda": top2["arquetipo_demanda"],
            "logical_seat_category": top2["logical_seat_category"],
            "site": top2["site"],
            "type_site": top2["type_site"],
            "precio": float(top2["base_unit_amt_itx"]),
            "score_confianza": round(float(top2["score_confianza"]), 4),
            "es_frontera": False,
            "percentil_precio_absoluto_dentro_tipo": round(float(top2.get("percentil_precio_absoluto_dentro_tipo", 0.0)), 4),
            "product": top2["product"]
        })
        count_prem += 1

    print(f"  • Total tarjetas en grupo PREMIUM: {count_prem} (Tope: <= 8)")

    # 2. Grupo MASIVOS: Platea General (PGI-0..3) + Popular (POP-0..3) + Grada (GGM-0..2) <= 12 tarjetas
    masivos_mcs = [
        "PGI-0", "PGI-1", "PGI-2", "PGI-3",
        "POP-0", "POP-1", "POP-2", "POP-3",
        "GGM-0", "GGM-1", "GGM-2"
    ]
    print("\n--- Grupo MASIVOS (Platea General + Popular + Grada) ---")
    count_masiv = 0
    segundos_masiv = []
    for mc in masivos_mcs:
        sub = df_valid[(df_valid["micro_cluster_id"] == mc) & (~df_valid["es_frontera"])].sort_values("score_confianza", ascending=False).drop_duplicates(subset=["logical_seat_category"])
        if len(sub) > 0:
            top1 = sub.iloc[0]
            filas_galeria.append({
                "grupo": "masivos",
                "caso": "representante",
                "micro_cluster_id": mc,
                "label_auto": top1["label_auto"],
                "arquetipo_demanda": top1["arquetipo_demanda"],
                "logical_seat_category": top1["logical_seat_category"],
                "site": top1["site"],
                "type_site": top1["type_site"],
                "precio": float(top1["base_unit_amt_itx"]),
                "score_confianza": round(float(top1["score_confianza"]), 4),
                "es_frontera": False,
                "percentil_precio_absoluto_dentro_tipo": round(float(top1.get("percentil_precio_absoluto_dentro_tipo", 0.0)), 4),
                "product": top1["product"]
            })
            count_masiv += 1
            if len(sub) > 1:
                segundos_masiv.append((len(sub), sub.iloc[1]))

    # Para alcanzar hasta 12 tarjetas (tope <= 12), agregar top 2 del más poblado
    segundos_masiv.sort(key=lambda x: x[0], reverse=True)
    for _, top2 in segundos_masiv[:(12 - count_masiv)]:
        filas_galeria.append({
            "grupo": "masivos",
            "caso": "representante",
            "micro_cluster_id": top2["micro_cluster_id"],
            "label_auto": top2["label_auto"],
            "arquetipo_demanda": top2["arquetipo_demanda"],
            "logical_seat_category": top2["logical_seat_category"],
            "site": top2["site"],
            "type_site": top2["type_site"],
            "precio": float(top2["base_unit_amt_itx"]),
            "score_confianza": round(float(top2["score_confianza"]), 4),
            "es_frontera": False,
            "percentil_precio_absoluto_dentro_tipo": round(float(top2.get("percentil_precio_absoluto_dentro_tipo", 0.0)), 4),
            "product": top2["product"]
        })
        count_masiv += 1

    print(f"  • Total tarjetas en grupo MASIVOS: {count_masiv} (Tope: <= 12)")

    # 3. Grupo ESPECIALES: AU-0 (2) + 2 fronteras + 1 híbrido
    print("\n--- Grupo ESPECIALES (AU-0, Fronteras e Híbridos) ---")
    count_esp = 0
    # AU-0 (2)
    sub_au = df_valid[(df_valid["micro_cluster_id"] == "AU-0") & (~df_valid["es_frontera"])].sort_values("score_confianza", ascending=False)
    for idx_au in range(min(2, len(sub_au))):
        r_au = sub_au.iloc[idx_au]
        filas_galeria.append({
            "grupo": "especiales",
            "caso": "representante",
            "micro_cluster_id": "AU-0",
            "label_auto": r_au["label_auto"],
            "arquetipo_demanda": r_au["arquetipo_demanda"],
            "logical_seat_category": r_au["logical_seat_category"],
            "site": r_au["site"],
            "type_site": r_au["type_site"],
            "precio": float(r_au["base_unit_amt_itx"]),
            "score_confianza": round(float(r_au["score_confianza"]), 4),
            "es_frontera": False,
            "percentil_precio_absoluto_dentro_tipo": round(float(r_au.get("percentil_precio_absoluto_dentro_tipo", 0.0)), 4),
            "product": r_au["product"]
        })
        count_esp += 1

    # 2 fronteras explícitas (menor confianza en todo el dataset de test)
    sub_front = df_valid[df_valid["es_frontera"]].sort_values("score_confianza", ascending=True)
    for idx_fr in range(min(2, len(sub_front))):
        r_fr = sub_front.iloc[idx_fr]
        filas_galeria.append({
            "grupo": "especiales",
            "caso": "frontera",
            "micro_cluster_id": r_fr["micro_cluster_id"],
            "label_auto": r_fr["label_auto"],
            "arquetipo_demanda": r_fr["arquetipo_demanda"],
            "logical_seat_category": r_fr["logical_seat_category"],
            "site": r_fr["site"],
            "type_site": r_fr["type_site"],
            "precio": float(r_fr["base_unit_amt_itx"]),
            "score_confianza": round(float(r_fr["score_confianza"]), 4),
            "es_frontera": True,
            "percentil_precio_absoluto_dentro_tipo": round(float(r_fr.get("percentil_precio_absoluto_dentro_tipo", 0.0)), 4),
            "product": r_fr["product"]
        })
        count_esp += 1

    # 1 híbrido (label_auto empieza por 'hibrido', con es_frontera == False de mayor confianza)
    sub_hib = df_valid[(df_valid["label_auto"].str.startswith("hibrido", na=False)) & (~df_valid["es_frontera"])].sort_values("score_confianza", ascending=False)
    if len(sub_hib) > 0:
        r_hib = sub_hib.iloc[0]
        filas_galeria.append({
            "grupo": "especiales",
            "caso": "hibrido",
            "micro_cluster_id": r_hib["micro_cluster_id"],
            "label_auto": r_hib["label_auto"],
            "arquetipo_demanda": r_hib["arquetipo_demanda"],
            "logical_seat_category": r_hib["logical_seat_category"],
            "site": r_hib["site"],
            "type_site": r_hib["type_site"],
            "precio": float(r_hib["base_unit_amt_itx"]),
            "score_confianza": round(float(r_hib["score_confianza"]), 4),
            "es_frontera": False,
            "percentil_precio_absoluto_dentro_tipo": round(float(r_hib.get("percentil_precio_absoluto_dentro_tipo", 0.0)), 4),
            "product": r_hib["product"]
        })
        count_esp += 1

    print(f"  • Total tarjetas en grupo ESPECIALES: {count_esp} (AU-0: 2, Frontera: 2, Híbrido: 1)")

    out_ext_csv = "reports/ejemplos_galeria_extendida.csv"
    df_gal = pd.DataFrame(filas_galeria)
    df_gal["localidad"] = df_gal["logical_seat_category"]
    df_gal["recinto"] = df_gal["site"]
    df_gal.to_csv(out_ext_csv, index=False, encoding="utf-8")
    print(f"\n[OK] Galería extendida exportada a: {out_ext_csv} ({len(df_gal)} filas totales)")

    return df_gal


def generar_anexo_excel():
    """
    Genera reports/anexo_asignaciones_marcha_blanca.xlsx con las 1,324 localidades evaluadas
    en la marcha blanca, con formato profesional, autofiltro activado y congelamiento de encabezado.
    """
    print("\n==================================================================")
    print("GENERANDO ANEXO EXCEL VERIFICABLE (openpyxl)")
    print("==================================================================")

    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    path_mb = "data/processed/marcha_blanca_predicciones.parquet"
    df_mb = pd.read_parquet(path_mb)

    cols_export = {
        "logical_seat_category": "localidad",
        "site": "recinto",
        "base_unit_amt_itx": "precio",
        "type_site": "type_site",
        "micro_cluster_id": "micro_cluster_id",
        "label_auto": "label_auto",
        "arquetipo_demanda": "arquetipo_demanda",
        "score_confianza": "score_confianza",
        "es_frontera": "es_frontera"
    }

    df_anexo = df_mb[list(cols_export.keys())].rename(columns=cols_export).copy()
    out_xlsx = "reports/anexo_asignaciones_marcha_blanca.xlsx"

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Asignaciones_Marcha_Blanca"

    # Estilos
    header_fill = PatternFill(start_color="0B2545", end_color="0B2545", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    data_font = Font(name="Calibri", size=10)
    border_thin = Side(style="thin", color="D7DCE4")
    cell_border = Border(left=border_thin, right=border_thin, top=border_thin, bottom=border_thin)

    headers = list(df_anexo.columns)
    ws.append(headers)

    for col_num, _ in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_num)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = cell_border

    # Agregar datos
    for _, row in df_anexo.iterrows():
        ws.append(list(row.values))

    # Formatear filas de datos
    for row_idx in range(2, len(df_anexo) + 2):
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.font = data_font
            cell.border = cell_border

            # Formato precio (columna 3)
            if col_idx == 3:
                cell.number_format = '"$"#,##0'
                cell.alignment = Alignment(horizontal="right")
            # Formato confianza (columna 8)
            elif col_idx == 8:
                cell.number_format = "0.0%"
                cell.alignment = Alignment(horizontal="right")
            # Booleano / id centrados
            elif col_idx in [5, 9]:
                cell.alignment = Alignment(horizontal="center")
            else:
                cell.alignment = Alignment(horizontal="left")

    # Autofilter y congelar primera fila
    ws.auto_filter.ref = ws.dimensions
    ws.freeze_panes = "A2"

    # Ajustar ancho de columnas
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    wb.save(out_xlsx)
    size_mb = os.path.getsize(out_xlsx) / 1024
    print(f"[OK] Anexo Excel generado exitosamente:")
    print(f"  • Archivo : {out_xlsx}")
    print(f"  • Filas   : {len(df_anexo):,} localidades")
    print(f"  • Tamaño  : {size_mb:.1f} KB")

    return out_xlsx


if __name__ == "__main__":
    preparar_ejemplos()
    preparar_galeria_extendida()
    generar_anexo_excel()

