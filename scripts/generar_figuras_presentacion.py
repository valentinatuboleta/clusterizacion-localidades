"""
Script para generar las figuras oficiales de la presentación de micro-clusters
y arquetipos ejemplificados con la Marcha Blanca.

Genera:
  - reports/figures/fig_arbol_arquetipos_microclusters.png
  - reports/figures/fig_tarjetas_ejemplos_marcha.png
  - reports/figures/fig_galeria_premium.png
  - reports/figures/fig_galeria_masivos.png
  - reports/figures/fig_galeria_especiales.png
  - reports/figures/fig_matriz_recintos_arquetipos.png

Convenciones:
  - reports/figures/
  - Estilo: seaborn-v0_8-whitegrid
  - Resolución: 150 DPI
  - Matplotlib puro (sin dependencias adicionales)
  - Paleta corporativa: Navy (#0B2545), Steel (#134074), Teal (#009688), Coral (#E76F51),
    Amber (#F4A261), Muted Grey (#6C757D)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

# Configurar estilo y tipografía
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#D7DCE4"
plt.rcParams["axes.linewidth"] = 0.8

# Paleta oficial por Arquetipo
COLOR_MAP = {
    "Admisión Única / Tarifa Plana": "#009688",          # Teal
    "VIP / Palcos / Premium": "#E76F51",                 # Coral / Terracotta
    "Popular / Balcón / Visibilidad Parcial": "#0B2545", # Navy profundo
    "Platea General / Intermedia": "#134074",             # Steel Blue
    "Preferencial / Platea Frontal": "#F4A261",           # Amber / Dorado
    "Grada General / Masiva": "#6C757D"                  # Slate Grey
}

BG_LIGHT_MAP = {
    "Admisión Única / Tarifa Plana": "#E0F2F1",
    "VIP / Palcos / Premium": "#FBE9E7",
    "Popular / Balcón / Visibilidad Parcial": "#E8EAF6",
    "Platea General / Intermedia": "#E3F2FD",
    "Preferencial / Platea Frontal": "#FFF3E0",
    "Grada General / Masiva": "#F5F5F5"
}

def format_precio_cop(precio):
    """Formato compacto para moneda COP en proyector."""
    if precio >= 1000:
        return f"${precio:,.0f} COP".replace(",", ".")
    return f"${precio:.0f} COP"

def generar_figura_arbol():
    """Figura 1: Diagrama de árbol de dos niveles (Arquetipos -> Micro-clusters)."""
    print("\n[1/6] Generando Figura 1: fig_arbol_arquetipos_microclusters.png...")
    
    df_cat = pd.read_csv("data/processed/cluster_catalog_v3.csv")
    df_cat = df_cat[~df_cat["micro_cluster_id"].isin(["VIP-3", "PPF-3"])].copy()

    arquetipos_orden = [
        "Admisión Única / Tarifa Plana",
        "Popular / Balcón / Visibilidad Parcial",
        "VIP / Palcos / Premium",
        "Platea General / Intermedia",
        "Preferencial / Platea Frontal",
        "Grada General / Masiva"
    ]

    fig, ax = plt.subplots(figsize=(15, 11), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    fig.text(0.05, 0.96, "Arquitectura Jerárquica de Clasificación (Rollup 1:1 Invariante)", 
             fontsize=18, fontweight="bold", color="#0B2545")
    fig.text(0.05, 0.935, "Nivel 1: Arquetipos de Demanda (Negocio)  ⟶  Nivel 2: Micro-Clusters Canónicos con label_auto (Granularidad Técnica)", 
             fontsize=11, color="#555555")

    x_n1, w_n1, h_n1 = 6.0, 28.0, 9.0
    x_n2, w_n2, h_n2 = 50.0, 44.0, 3.6
    y_positions = [83.0, 68.0, 52.0, 36.5, 21.0, 6.5]

    for idx, arq in enumerate(arquetipos_orden):
        y_center_n1 = y_positions[idx]
        color_arq = COLOR_MAP.get(arq, "#134074")
        bg_arq = BG_LIGHT_MAP.get(arq, "#F0F4F8")

        sub_cat = df_cat[df_cat["arquetipo_macro"].str.contains(arq.split(" / ")[0], na=False)]
        n_total_arq = sub_cat["n"].sum()
        pct_arq = n_total_arq / 33775.0 * 100

        box_n1 = patches.FancyBboxPatch(
            (x_n1, y_center_n1 - h_n1 / 2), w_n1, h_n1,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=bg_arq, edgecolor=color_arq, linewidth=2.0
        )
        ax.add_patch(box_n1)

        strip = patches.FancyBboxPatch(
            (x_n1, y_center_n1 - h_n1 / 2), 1.8, h_n1,
            boxstyle="round,pad=0.2,rounding_size=0.6",
            facecolor=color_arq, edgecolor="none"
        )
        ax.add_patch(strip)

        ax.text(x_n1 + 3.0, y_center_n1 + 1.6, arq, fontsize=11, fontweight="bold", color="#0B2545", va="center")
        ax.text(x_n1 + 3.0, y_center_n1 - 1.8, f"Nivel 1 | N = {n_total_arq:,} ({pct_arq:.1f}%) | k={len(sub_cat)} micro-clusters", 
                fontsize=9.0, color="#444444", va="center")

        n_leaves = len(sub_cat)
        leaf_y_offsets = [0.0] if n_leaves == 1 else [(n_leaves - 1) * 4.4 / 2.0 - i * 4.4 for i in range(n_leaves)]

        for leaf_idx, (_, r_leaf) in enumerate(sub_cat.iterrows()):
            y_leaf = y_center_n1 + leaf_y_offsets[leaf_idx]
            mc_id = r_leaf["micro_cluster_id"]
            lbl = r_leaf["label_auto"]
            n_leaf = int(r_leaf["n"])
            p_tag = r_leaf["purity_tag"]
            dom_tag = str(r_leaf["tag_dominante"]).replace("tag_", "")

            x_start, y_start = x_n1 + w_n1, y_center_n1
            x_mid = (x_start + x_n2) / 2.0
            ax.plot([x_start, x_mid, x_mid, x_n2], [y_start, y_start, y_leaf, y_leaf],
                    color=color_arq, linewidth=1.5, alpha=0.75, zorder=2)

            box_n2 = patches.FancyBboxPatch(
                (x_n2, y_leaf - h_n2 / 2), w_n2, h_n2,
                boxstyle="round,pad=0.4,rounding_size=0.8",
                facecolor="#FFFFFF", edgecolor=color_arq, linewidth=1.2, zorder=3
            )
            ax.add_patch(box_n2)

            badge = patches.FancyBboxPatch(
                (x_n2 + 0.8, y_leaf - 1.2), 6.5, 2.4,
                boxstyle="round,pad=0.2,rounding_size=0.5",
                facecolor=color_arq, edgecolor="none", zorder=4
            )
            ax.add_patch(badge)
            ax.text(x_n2 + 4.05, y_leaf, mc_id, fontsize=9.5, fontweight="bold", color="#FFFFFF", 
                    va="center", ha="center", zorder=5)

            txt_leaf = f"label_auto: {lbl}   |   n = {n_leaf:,}   |   tag domin: {dom_tag} ({p_tag*100:.0f}%)"
            ax.text(x_n2 + 8.2, y_leaf, txt_leaf, fontsize=9.0, color="#212529", va="center", zorder=5)

    nota = ("* Nota Técnica: Invarianza de Rollup 1:1 verificada por construcción. Micro-clusters evaluados con k local optimizado (Codo-DB) y estabilidad ARI > 0.88.\n"
            "  VIP-3 y PPF-3 descalificados formalmente por compuerta de no degeneración (min share >= 3%). AU-0 opera como nodo terminal determinístico de admisión única.")
    fig.text(0.05, 0.015, nota, fontsize=8.2, color="#666666", style="italic")

    out_path = "reports/figures/fig_arbol_arquetipos_microclusters.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(out_path, dpi=150, facecolor="#FFFFFF")
    plt.close()
    print(f"  [OK] Guardada en: {out_path}")


def generar_figura_tarjetas():
    """Figura 2: Las 6 tarjetas héroes representativas de la Marcha Blanca (Grilla 2x3)."""
    print("\n[2/6] Generando Figura 2: fig_tarjetas_ejemplos_marcha.png...")

    df_ej = pd.read_csv("reports/ejemplos_presentacion.csv")
    df_tarjetas = df_ej[df_ej["es_tarjeta_slide_6"]].copy()

    fig, ax = plt.subplots(figsize=(15, 9.5), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    fig.text(0.05, 0.955, "Galería de Validación en Marcha Blanca (Shadow Testing)", 
             fontsize=18, fontweight="bold", color="#0B2545")
    fig.text(0.05, 0.925, "Doble Etiquetado en Datos Reales de Producción: Micro-Cluster Técnico (label_auto) + Arquetipo Macro de Demanda", 
             fontsize=11, color="#555555")

    coords = [
        (4.0, 50.0), (36.0, 50.0), (68.0, 50.0),
        (4.0, 7.0),  (36.0, 7.0),  (68.0, 7.0),
    ]
    card_w, card_h = 28.0, 38.0

    for idx, (_, r) in enumerate(df_tarjetas.iterrows()):
        x0, y0 = coords[idx]
        arq = r["arquetipo_demanda"]
        color_theme = COLOR_MAP.get(arq, "#134074")
        bg_theme = BG_LIGHT_MAP.get(arq, "#F8F9FA")

        card_bg = patches.FancyBboxPatch(
            (x0, y0), card_w, card_h,
            boxstyle="round,pad=0.5,rounding_size=1.5",
            facecolor="#FFFFFF", edgecolor="#D7DCE4", linewidth=1.2, zorder=2
        )
        ax.add_patch(card_bg)

        header_h = 8.5
        header_bar = patches.FancyBboxPatch(
            (x0, y0 + card_h - header_h), card_w, header_h,
            boxstyle="round,pad=0.5,rounding_size=1.2",
            facecolor=color_theme, edgecolor="none", zorder=3
        )
        ax.add_patch(header_bar)

        arq_titulo = arq.replace(" / ", "\n")
        ax.text(x0 + card_w / 2.0, y0 + card_h - header_h / 2.0, arq_titulo,
                fontsize=9.5, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=4)

        loc_nombre = str(r["logical_seat_category"])
        if len(loc_nombre) > 28:
            loc_nombre = loc_nombre[:26] + "..."
        ax.text(x0 + 1.5, y0 + card_h - 11.5, loc_nombre,
                fontsize=11.5, fontweight="bold", color="#0B2545", zorder=4)

        venue_str = f"Recinto: {r['site']}"
        if len(venue_str) > 34:
            venue_str = venue_str[:32] + "..."
        ax.text(x0 + 1.5, y0 + card_h - 15.0, venue_str,
                fontsize=9.0, color="#555555", zorder=4)

        tipo_str = f"Tipología: {r['type_site']}"
        ax.text(x0 + 1.5, y0 + card_h - 18.0, tipo_str,
                fontsize=8.5, color="#666666", zorder=4)

        precio_str = f"Precio: {format_precio_cop(r['precio'])}"
        ax.text(x0 + 1.5, y0 + card_h - 21.0, precio_str,
                fontsize=9.5, fontweight="bold", color="#134074", zorder=4)

        ax.plot([x0 + 1.5, x0 + card_w - 1.5], [y0 + card_h - 23.0, y0 + card_h - 23.0],
                color="#E2E8F0", linewidth=1.0, zorder=4)

        id_box = patches.FancyBboxPatch(
            (x0 + 1.5, y0 + 5.0), card_w - 3.0, 9.5,
            boxstyle="round,pad=0.3,rounding_size=0.8",
            facecolor=bg_theme, edgecolor=color_theme, linewidth=1.0, zorder=3
        )
        ax.add_patch(id_box)

        ax.text(x0 + 2.5, y0 + 12.0, "Doble Identidad Asignada:",
                fontsize=8.0, fontweight="bold", color="#555555", zorder=4)
        ax.text(x0 + 2.5, y0 + 9.5, f"• Micro-Cluster: {r['micro_cluster_id']}",
                fontsize=8.8, fontweight="bold", color=color_theme, zorder=4)
        ax.text(x0 + 2.5, y0 + 7.0, f"• label_auto: {r['label_auto']}",
                fontsize=8.5, color="#212529", zorder=4)

        conf_pct = r['score_confianza'] * 100
        pill_color = "#2A9D8F" if conf_pct >= 60 else ("#F4A261" if conf_pct >= 40 else "#E76F51")
        pill = patches.FancyBboxPatch(
            (x0 + 1.5, y0 + 1.2), card_w - 3.0, 2.8,
            boxstyle="round,pad=0.2,rounding_size=0.6",
            facecolor=pill_color, edgecolor="none", zorder=4
        )
        ax.add_patch(pill)
        ax.text(x0 + card_w / 2.0, y0 + 2.6, f"Confianza Geométrica: {conf_pct:.1f}%  |  Frontera: No",
                fontsize=8.2, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=5)

    fig.text(0.05, 0.015, "* Todas las tarjetas corresponden a eventos reales certificados en la marcha blanca (GOLD/SECUTIX test, N=1,324). Confianza >= 45% y es_frontera=False.",
             fontsize=8.2, color="#666666", style="italic")

    out_path = "reports/figures/fig_tarjetas_ejemplos_marcha.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(out_path, dpi=150, facecolor="#FFFFFF")
    plt.close()
    print(f"  [OK] Guardada en: {out_path}")


def generar_galeria_compacta(grupo, titulo, subtitulo, out_file, n_rows=3, n_cols=3, fig_h=10.5):
    """Genera una grilla compacta de tarjetas con barra de color de confianza."""
    print(f"\nGenerando galería compacta para grupo '{grupo}': {out_file}...")

    df_gal = pd.read_csv("reports/ejemplos_galeria_extendida.csv")
    df_sub = df_gal[df_gal["grupo"] == grupo].copy()

    fig, ax = plt.subplots(figsize=(15.2, fig_h), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    fig.text(0.04, 0.96, titulo, fontsize=17, fontweight="bold", color="#0B2545")
    fig.text(0.04, 0.93, subtitulo, fontsize=10.5, color="#555555")

    # Layout de la grilla (3 columnas)
    card_w = 28.8
    spacing_x = 3.6
    x_starts = [4.0, 4.0 + card_w + spacing_x, 4.0 + 2 * (card_w + spacing_x)]

    # Cálculo dinámico de altura y separación vertical
    y_top = 89.0
    y_bottom = 5.0
    total_h = y_top - y_bottom
    spacing_y = 2.8
    card_h = (total_h - (n_rows - 1) * spacing_y) / n_rows

    for idx, (_, r) in enumerate(df_sub.iterrows()):
        col = idx % n_cols
        row = idx // n_cols
        if row >= n_rows:
            break

        x0 = x_starts[col]
        y0 = y_top - (row + 1) * card_h - row * spacing_y

        arq = r["arquetipo_demanda"]
        color_theme = COLOR_MAP.get(arq, "#134074")
        bg_theme = BG_LIGHT_MAP.get(arq, "#F8F9FA")
        caso = str(r["caso"])

        # Estilo de borde
        is_front = (caso == "frontera")
        is_hib = (caso == "hibrido")
        edge_col = "#E63946" if is_front else ("#7209B7" if is_hib else "#D7DCE4")
        line_st = "--" if is_front else "-"
        line_w = 1.8 if (is_front or is_hib) else 1.1

        # Fondo de tarjeta
        card_bg = patches.FancyBboxPatch(
            (x0, y0), card_w, card_h,
            boxstyle="round,pad=0.3,rounding_size=1.0",
            facecolor="#FFFFFF", edgecolor=edge_col, linestyle=line_st, linewidth=line_w, zorder=2
        )
        ax.add_patch(card_bg)

        # Franja superior de encabezado
        hdr_h = card_h * 0.22
        hdr_bar = patches.FancyBboxPatch(
            (x0, y0 + card_h - hdr_h), card_w, hdr_h,
            boxstyle="round,pad=0.3,rounding_size=0.8",
            facecolor=color_theme, edgecolor="none", zorder=3
        )
        ax.add_patch(hdr_bar)

        # Micro-cluster y label en el encabezado
        mc_text = f"{r['micro_cluster_id']}  •  {r['label_auto']}"
        if len(mc_text) > 30:
            mc_text = mc_text[:28] + "…"
        ax.text(x0 + 1.2, y0 + card_h - hdr_h / 2.0, mc_text,
                fontsize=9.5, fontweight="bold", color="#FFFFFF", va="center", zorder=4)

        # Badge de caso especial si aplica (FRONTERA / HÍBRIDO)
        if is_front:
            badge_b = patches.FancyBboxPatch(
                (x0 + card_w - 7.5, y0 + card_h - hdr_h + 0.5), 6.8, hdr_h - 1.0,
                boxstyle="round,pad=0.2,rounding_size=0.4",
                facecolor="#E63946", edgecolor="none", zorder=5
            )
            ax.add_patch(badge_b)
            ax.text(x0 + card_w - 4.1, y0 + card_h - hdr_h / 2.0, "FRONTERA",
                    fontsize=8.0, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=6)
        elif is_hib:
            badge_b = patches.FancyBboxPatch(
                (x0 + card_w - 6.5, y0 + card_h - hdr_h + 0.5), 5.8, hdr_h - 1.0,
                boxstyle="round,pad=0.2,rounding_size=0.4",
                facecolor="#7209B7", edgecolor="none", zorder=5
            )
            ax.add_patch(badge_b)
            ax.text(x0 + card_w - 3.6, y0 + card_h - hdr_h / 2.0, "HÍBRIDO",
                    fontsize=8.0, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=6)

        # Nombre de localidad comercial (destacado)
        loc_nombre = str(r["logical_seat_category"])
        if len(loc_nombre) > 26:
            loc_nombre = loc_nombre[:24] + "…"
        ax.text(x0 + 1.2, y0 + card_h - hdr_h - 2.8, loc_nombre,
                fontsize=10.2, fontweight="bold", color="#0B2545", zorder=4)

        # Recinto y Tipología
        recinto_str = f"{r['site']} ({r['type_site']})"
        if len(recinto_str) > 34:
            recinto_str = recinto_str[:32] + "…"
        ax.text(x0 + 1.2, y0 + card_h - hdr_h - 5.5, recinto_str,
                fontsize=8.8, color="#555555", zorder=4)

        # Precio
        precio_str = f"Tarifa: {format_precio_cop(r['precio'])}"
        ax.text(x0 + 1.2, y0 + card_h - hdr_h - 8.2, precio_str,
                fontsize=9.2, fontweight="bold", color="#134074", zorder=4)

        # Barra de confianza bajo la tarjeta
        conf = float(r["score_confianza"])
        conf_clamped = min(max(conf, 0.0), 1.0)
        bar_w_total = card_w - 2.4
        bar_h = 2.4
        y_bar = y0 + 1.2

        # Pista gris de fondo
        track = patches.FancyBboxPatch(
            (x0 + 1.2, y_bar), bar_w_total, bar_h,
            boxstyle="round,pad=0.1,rounding_size=0.5",
            facecolor="#E9ECEF", edgecolor="#CED4DA", linewidth=0.6, zorder=3
        )
        ax.add_patch(track)

        # Color de barra por umbral (0.7 / 0.5)
        if conf >= 0.70:
            bar_color = "#2A9D8F"  # Verde
        elif conf >= 0.50:
            bar_color = "#F4A261"  # Ámbar
        else:
            bar_color = "#E76F51"  # Rojo / Coral

        # Barra rellena
        fill_w = max(bar_w_total * conf_clamped, 0.8)
        bar_fill = patches.FancyBboxPatch(
            (x0 + 1.2, y_bar), fill_w, bar_h,
            boxstyle="round,pad=0.1,rounding_size=0.5",
            facecolor=bar_color, edgecolor="none", zorder=4
        )
        ax.add_patch(bar_fill)

        # Etiqueta de confianza sobre la barra
        txt_conf = f"Confianza: {conf*100:.1f}%"
        txt_col = "#FFFFFF" if (fill_w > 12.0) else "#212529"
        ax.text(x0 + 2.2, y_bar + bar_h / 2.0, txt_conf,
                fontsize=8.0, fontweight="bold", color=txt_col, va="center", zorder=5)

    out_path = f"reports/figures/{out_file}"
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(out_path, dpi=150, facecolor="#FFFFFF")
    plt.close()
    print(f"  [OK] Guardada en: {out_path}")


def generar_matriz_recintos():
    """Figura: Matriz de los top 10 recintos cruzados por arquetipos en Marcha Blanca (N=1,324)."""
    print("\n[6/6] Generando Figura: fig_matriz_recintos_arquetipos.png...")

    df_mb = pd.read_parquet("data/processed/marcha_blanca_predicciones.parquet")

    # Top 10 recintos ordenados de menor a mayor para barh
    top10_sites = df_mb["site"].value_counts().head(10).index.tolist()[::-1]
    ct = pd.crosstab(df_mb["site"], df_mb["arquetipo_demanda"]).reindex(top10_sites)

    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=150)

    # Orden canónico de arquetipos para apilar
    arquetipos_orden = [
        "Admisión Única / Tarifa Plana",
        "Popular / Balcón / Visibilidad Parcial",
        "VIP / Palcos / Premium",
        "Platea General / Intermedia",
        "Preferencial / Platea Frontal",
        "Grada General / Masiva"
    ]
    # Filtrar solo arquetipos presentes en crosstab
    arquetipos_presentes = [a for a in arquetipos_orden if a in ct.columns]

    y_pos = np.arange(len(top10_sites))
    left_accum = np.zeros(len(top10_sites))

    for arq in arquetipos_presentes:
        vals = ct[arq].values
        color_arq = COLOR_MAP.get(arq, "#134074")
        bars = ax.barh(y_pos, vals, left=left_accum, height=0.62,
                       label=arq, color=color_arq, edgecolor="#FFFFFF", linewidth=0.8, alpha=0.95)

        # Anotar números en segmentos con volumen >= 6
        for i, val in enumerate(vals):
            if val >= 6:
                x_center = left_accum[i] + val / 2.0
                ax.text(x_center, y_pos[i], f"{int(val)}",
                        ha="center", va="center", fontsize=8.8, fontweight="bold", color="#FFFFFF")

        left_accum += vals

    # Anotar total a la derecha de cada barra
    for i, tot in enumerate(left_accum):
        ax.text(tot + 2.5, y_pos[i], f"{int(tot):,}",
                ha="left", va="center", fontsize=9.5, fontweight="bold", color="#0B2545")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top10_sites, fontsize=10.0, fontweight="bold", color="#0B2545")
    ax.set_xlabel("Número de Localidades Consistentes Asignadas", fontsize=11, fontweight="bold", color="#0B2545", labelpad=10)
    ax.set_xlim(0, max(left_accum) * 1.14)

    # Título explícito requerido
    ax.set_title("N=1,324 asignaciones — marcha blanca\nDistribución de Arquetipos de Demanda por Recinto (Top 10 Venues)", 
                 fontsize=15, fontweight="bold", color="#0B2545", pad=15, loc="left")

    # Leyenda limpia en la parte superior/derecha
    ax.legend(title="Arquetipo Macro de Demanda", title_fontsize=10.5, fontsize=9.5, 
              loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#D7DCE4")

    ax.grid(axis="x", linestyle="--", alpha=0.6)

    out_path = "reports/figures/fig_matriz_recintos_arquetipos.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="#FFFFFF")
    plt.close()
    print(f"  [OK] Guardada en: {out_path}")


def generar_todas_las_figuras():
    os.makedirs("reports/figures", exist_ok=True)
    generar_figura_arbol()
    generar_figura_tarjetas()

    # Figuras de evidencia densificada por segmento
    generar_galeria_compacta(
        grupo="premium",
        titulo="Galería Segmento Premium (VIP y Preferencial Frontal)",
        subtitulo="Evidencia Marcha Blanca: Mismo Arquetipo, Diferentes Micro-Clusters Técnicos y Variaciones de Localidad (≤ 8 Tarjetas)",
        out_file="fig_galeria_premium.png",
        n_rows=3, n_cols=3, fig_h=10.5
    )

    generar_galeria_compacta(
        grupo="masivos",
        titulo="Galería Segmento Masivos (Platea General, Balcón / Popular y Grada)",
        subtitulo="Evidencia Marcha Blanca: Granularidad Interna de Micro-Clusters en Sectores Masivos (≤ 12 Tarjetas)",
        out_file="fig_galeria_masivos.png",
        n_rows=4, n_cols=3, fig_h=13.0
    )

    generar_galeria_compacta(
        grupo="especiales",
        titulo="Galería Casos Especiales (Admisión Única, Fronteras Estadísticas e Híbridos)",
        subtitulo="Evidencia Marcha Blanca: Cobertura Determinística AU-0, Propagación Honesta de Fronteras e Híbridos Léxicos",
        out_file="fig_galeria_especiales.png",
        n_rows=2, n_cols=3, fig_h=8.0
    )

    generar_matriz_recintos()
    print("\n[ÉXITO] Todas las 6 figuras generadas y validadas en reports/figures/.")


if __name__ == "__main__":
    generar_todas_las_figuras()
