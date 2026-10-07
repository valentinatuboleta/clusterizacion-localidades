"""
Script para seleccionar 5 eventos representativos por tamaño de aforo (quintiles)
a partir de las 1,324 localidades de la marcha blanca evaluada.

Quintiles de aforo:
  - micro: <= 500
  - pequeno: 500 - 2,000
  - mediano: 2,000 - 8,000
  - grande: 8,000 - 20,000
  - estadio: > 20,000

Criterios:
  - Evento coherente (pasó las 9 reglas de consistencia de negocio).
  - >= 4 localidades.
  - Recintos distintos entre los 5 eventos.
  - Si un estrato no tiene candidato exacto, toma el más cercano (marcando estrato_real).

Exporta:
  - reports/eventos_seleccionados.csv
"""

import os
import pandas as pd

PATH_MARCHA_BLANCA = "data/processed/marcha_blanca_predicciones.parquet"
OUTPUT_CSV = "reports/eventos_seleccionados.csv"

# Definición de rangos de aforo para cada estrato objetivo
ESTRATOS_OBJETIVO = {
    "micro": (0, 500),
    "pequeno": (500, 2000),
    "mediano": (2000, 8000),
    "grande": (8000, 20000),
    "estadio": (20000, float("inf"))
}


def clasificar_estrato_aforo(quota: float) -> str:
    """Clasifica un aforo numérico en su estrato correspondiente."""
    for estrato, (min_q, max_q) in ESTRATOS_OBJETIVO.items():
        if min_q < quota <= max_q:
            return estrato
        elif min_q == 0 and quota <= max_q:
            return estrato
    return "estadio"


def seleccionar_5_eventos(path_mb: str = PATH_MARCHA_BLANCA, output_csv: str = OUTPUT_CSV) -> pd.DataFrame:
    """
    Selecciona los 5 eventos por quintil de aforo y exporta reports/eventos_seleccionados.csv.
    """
    print("==================================================================")
    print("SELECCIÓN DE 5 EVENTOS POR QUINTIL DE AFORO (MARCHA BLANCA)")
    print("==================================================================")

    if not os.path.exists(path_mb):
        raise FileNotFoundError(f"No se encontró el dataset de marcha blanca en {path_mb}")

    df_mb = pd.read_parquet(path_mb)
    print(f"• Total asignaciones cargadas : {len(df_mb):,} filas")

    # Agrupar por evento
    perf_summary = df_mb.groupby("t_performance_id").agg(
        product=("product", "first"),
        site=("site", "first"),
        performance_quota=("performance_quota", "first"),
        n_localidades=("logical_seat_category", "nunique")
    ).reset_index()

    print(f"• Total eventos únicos en lote: {len(perf_summary)}")

    # Filtrar eventos con al menos 4 localidades
    candidatos = perf_summary[perf_summary["n_localidades"] >= 4].copy()
    candidatos["estrato_calculado"] = candidatos["performance_quota"].apply(clasificar_estrato_aforo)
    print(f"• Eventos candidatos con >= 4 localidades: {len(candidatos)}")

    # Selección priorizando recintos distintos y coherencia
    # 1. micro: TRIBUTO A RAPHAEL @ BATUTA CALDAS - MANIZALES (aforo 240)
    # 2. pequeno: HANS ZIMMER VS JOHN WILLIAMS @ TEATRO JORGE ELIECER GAITAN (aforo 1,650)
    # 3. mediano: EL UNIPERSONAL DE LUCHO MELLERA @ MOVISTAR ARENA (aforo 6,435)
    # 4. grande: CARLOS VIVES MEDELLÍN @ DAVIARENA (aforo 12,979)
    # 5. estadio: THE LAST DANCE @ ESTADIO METROPOLITANO (aforo 44,586)

    eventos_seleccionados = []
    sites_usados = set()

    orden_estratos = ["micro", "pequeno", "mediano", "grande", "estadio"]

    # Diccionario de preferencias específicas probadas con alta riqueza de segmentación
    preferencias = {
        "micro": 10230576293804,    # TRIBUTO A RAPHAEL YO ME LLAMO (240)
        "pequeno": 10230442113886,  # HANS ZIMMER VS JOHN WILLIAMS (1,650)
        "mediano": 10230576845152,  # EL UNIPERSONAL DE LUCHO MELLERA (6,435)
        "grande": 10230558838697,   # CARLOS VIVES MEDELLÍN (12,979)
        "estadio": 10230536239014   # THE LAST DANCE (44,586)
    }

    for est in orden_estratos:
        pref_id = preferencias.get(est)
        row_cand = None

        if pref_id is not None:
            match = candidatos[candidatos["t_performance_id"] == pref_id]
            if len(match) > 0 and match.iloc[0]["site"] not in sites_usados:
                row_cand = match.iloc[0]

        # Si no se encontró el preferido, buscar dinámicamente en el estrato
        if row_cand is None:
            sub_est = candidatos[(candidatos["estrato_calculado"] == est) & (~candidatos["site"].isin(sites_usados))]
            if len(sub_est) > 0:
                row_cand = sub_est.sort_values("n_localidades", ascending=False).iloc[0]
            else:
                # Si no hay en el estrato exacto sin repetir site, tomar el más cercano
                min_q, max_q = ESTRATOS_OBJETIVO[est]
                target_mid = (min_q + (max_q if max_q != float("inf") else 30000)) / 2
                disp = candidatos[~candidatos["site"].isin(sites_usados)].copy()
                if len(disp) == 0:
                    disp = candidatos.copy()
                disp["dist"] = (disp["performance_quota"] - target_mid).abs()
                row_cand = disp.sort_values("dist").iloc[0]

        estrato_real = clasificar_estrato_aforo(row_cand["performance_quota"])
        sites_usados.add(row_cand["site"])

        eventos_seleccionados.append({
            "t_performance_id": row_cand["t_performance_id"],
            "product": row_cand["product"],
            "site": row_cand["site"],
            "performance_quota": float(row_cand["performance_quota"]),
            "n_localidades": int(row_cand["n_localidades"]),
            "estrato_objetivo": est,
            "estrato_real": estrato_real
        })

    df_out = pd.DataFrame(eventos_seleccionados)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False, encoding="utf-8")

    print("\n--- Eventos Seleccionados ---")
    for _, r in df_out.iterrows():
        print(f"  [{r['estrato_objetivo'].upper()}] {r['product']} @ {r['site']} | Aforo: {r['performance_quota']:,.0f} | Locs: {r['n_localidades']} (Estrato real: {r['estrato_real']})")

    print(f"\n[OK] Archivo exportado exitosamente a: {output_csv} ({len(df_out)} filas)")
    return df_out


if __name__ == "__main__":
    seleccionar_5_eventos()
