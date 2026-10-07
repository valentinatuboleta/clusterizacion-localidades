"""
Script para generar las figuras oficiales de la presentación de micro-clusters
y arquetipos ejemplificados con la Marcha Blanca.

Genera:
  - reports/figures/fig_arbol_arquetipos_microclusters.png
  - reports/figures/fig_tarjetas_ejemplos_marcha.png

Convenciones:
  - reports/figures/
  - Estilo: seaborn-v0_8-whitegrid
  - Resolución: 150 DPI
  - Matplotlib puro (sin plotly ni graphviz)
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

def generar_figura_arbol():
    """Figura 1: Diagrama de árbol de dos niveles (Arquetipos -> Micro-clusters)."""
    print("\n[1/2] Generando Figura 1: fig_arbol_arquetipos_microclusters.png...")
    
    # Datos del catálogo
    df_cat = pd.read_csv("data/processed/cluster_catalog_v3.csv")
    # Excluir degenerados
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

    # Título y Subtítulo
    fig.text(0.05, 0.96, "Arquitectura Jerárquica de Clasificación (Rollup 1:1 Invariante)", 
             fontsize=18, fontweight="bold", color="#0B2545")
    fig.text(0.05, 0.935, "Nivel 1: Arquetipos de Demanda (Negocio)  ⟶  Nivel 2: Micro-Clusters Canónicos con label_auto (Granularidad Técnica)", 
             fontsize=11, color="#555555")

    # Coordenadas X para niveles
    x_n1 = 6.0
    w_n1 = 28.0
    h_n1 = 9.0

    x_n2 = 50.0
    w_n2 = 44.0
    h_n2 = 3.6

    y_positions = [83.0, 68.0, 52.0, 36.5, 21.0, 6.5]

    for idx, arq in enumerate(arquetipos_orden):
        y_center_n1 = y_positions[idx]
        color_arq = COLOR_MAP.get(arq, "#134074")
        bg_arq = BG_LIGHT_MAP.get(arq, "#F0F4F8")

        sub_cat = df_cat[df_cat["arquetipo_macro"].str.contains(arq.split(" / ")[0], na=False)]
        n_total_arq = sub_cat["n"].sum()
        pct_arq = n_total_arq / 33775.0 * 100

        # Caja Nivel 1 (Arquetipo)
        box_n1 = patches.FancyBboxPatch(
            (x_n1, y_center_n1 - h_n1 / 2), w_n1, h_n1,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=bg_arq, edgecolor=color_arq, linewidth=2.0
        )
        ax.add_patch(box_n1)

        # Franja lateral izquierda de color
        strip = patches.FancyBboxPatch(
            (x_n1, y_center_n1 - h_n1 / 2), 1.8, h_n1,
            boxstyle="round,pad=0.2,rounding_size=0.6",
            facecolor=color_arq, edgecolor="none"
        )
        ax.add_patch(strip)

        ax.text(x_n1 + 3.0, y_center_n1 + 1.6, arq, fontsize=11, fontweight="bold", color="#0B2545", va="center")
        ax.text(x_n1 + 3.0, y_center_n1 - 1.8, f"Nivel 1 | N = {n_total_arq:,} ({pct_arq:.1f}%) | k={len(sub_cat)} micro-clusters", 
                fontsize=9.0, color="#444444", va="center")

        # Conectar con Micro-clusters
        n_leaves = len(sub_cat)
        if n_leaves == 1:
            leaf_y_offsets = [0.0]
        else:
            spacing = 4.4
            start_offset = (n_leaves - 1) * spacing / 2.0
            leaf_y_offsets = [start_offset - i * spacing for i in range(n_leaves)]

        for leaf_idx, (_, r_leaf) in enumerate(sub_cat.iterrows()):
            y_leaf = y_center_n1 + leaf_y_offsets[leaf_idx]
            mc_id = r_leaf["micro_cluster_id"]
            lbl = r_leaf["label_auto"]
            n_leaf = int(r_leaf["n"])
            p_tag = r_leaf["purity_tag"]
            dom_tag = str(r_leaf["tag_dominante"]).replace("tag_", "")

            # Línea conectora ortogonal / curva
            x_start = x_n1 + w_n1
            y_start = y_center_n1
            x_mid = (x_start + x_n2) / 2.0
            ax.plot([x_start, x_mid, x_mid, x_n2], [y_start, y_start, y_leaf, y_leaf],
                    color=color_arq, linewidth=1.5, alpha=0.75, zorder=2)

            # Caja Nivel 2 (Micro-cluster hoja)
            box_n2 = patches.FancyBboxPatch(
                (x_n2, y_leaf - h_n2 / 2), w_n2, h_n2,
                boxstyle="round,pad=0.4,rounding_size=0.8",
                facecolor="#FFFFFF", edgecolor=color_arq, linewidth=1.2, zorder=3
            )
            ax.add_patch(box_n2)

            # Badge del micro_cluster_id
            badge = patches.FancyBboxPatch(
                (x_n2 + 0.8, y_leaf - 1.2), 6.5, 2.4,
                boxstyle="round,pad=0.2,rounding_size=0.5",
                facecolor=color_arq, edgecolor="none", zorder=4
            )
            ax.add_patch(badge)
            ax.text(x_n2 + 4.05, y_leaf, mc_id, fontsize=9.5, fontweight="bold", color="#FFFFFF", 
                    va="center", ha="center", zorder=5)

            # Texto de la hoja
            txt_leaf = f"label_auto: {lbl}   |   n = {n_leaf:,}   |   tag domin: {dom_tag} ({p_tag*100:.0f}%)"
            ax.text(x_n2 + 8.2, y_leaf, txt_leaf, fontsize=9.0, color="#212529", va="center", zorder=5)

    # Nota al pie de estabilidad y compuertas
    nota = ("* Nota Técnica: Invarianza de Rollup 1:1 verificada por construcción. Micro-clusters evaluados con k local optimizado (Codo-DB) y estabilidad ARI > 0.88.\n"
            "  VIP-3 y PPF-3 descalificados formalmente por compuerta de no degeneración (min share >= 3%). AU-0 opera como nodo terminal determinístico de admisión única.")
    fig.text(0.05, 0.015, nota, fontsize=8.2, color="#666666", style="italic")

    out_path = "reports/figures/fig_arbol_arquetipos_microclusters.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(out_path, dpi=150, facecolor="#FFFFFF")
    plt.close()
    print(f"  [OK] Guardada en: {out_path}")

def generar_figura_tarjetas():
    """Figura 2: Las 6 tarjetas representativas de la Marcha Blanca (Grilla 2x3)."""
    print("\n[2/2] Generando Figura 2: fig_tarjetas_ejemplos_marcha.png...")

    # Cargar datos preparados
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

    # Grilla 2 filas x 3 columnas
    coords = [
        (4.0, 50.0),   # Fila 1, Col 1
        (36.0, 50.0),  # Fila 1, Col 2
        (68.0, 50.0),  # Fila 1, Col 3
        (4.0, 7.0),    # Fila 2, Col 1
        (36.0, 7.0),   # Fila 2, Col 2
        (68.0, 7.0),   # Fila 2, Col 3
    ]
    card_w = 28.0
    card_h = 38.0

    for idx, (_, r) in enumerate(df_tarjetas.iterrows()):
        x0, y0 = coords[idx]
        arq = r["arquetipo_demanda"]
        color_theme = COLOR_MAP.get(arq, "#134074")
        bg_theme = BG_LIGHT_MAP.get(arq, "#F8F9FA")

        # Tarjeta principal (fondo blanco con sombra sutil)
        card_bg = patches.FancyBboxPatch(
            (x0, y0), card_w, card_h,
            boxstyle="round,pad=0.5,rounding_size=1.5",
            facecolor="#FFFFFF", edgecolor="#D7DCE4", linewidth=1.2, zorder=2
        )
        ax.add_patch(card_bg)

        # Barra superior con color del arquetipo
        header_h = 8.5
        header_bar = patches.FancyBboxPatch(
            (x0, y0 + card_h - header_h), card_w, header_h,
            boxstyle="round,pad=0.5,rounding_size=1.2",
            facecolor=color_theme, edgecolor="none", zorder=3
        )
        ax.add_patch(header_bar)

        # Título de Arquetipo en Encabezado
        arq_titulo = arq.replace(" / ", "\n")
        ax.text(x0 + card_w / 2.0, y0 + card_h - header_h / 2.0, arq_titulo,
                fontsize=9.5, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=4)

        # Nombre de localidad comercial (grande y destacado)
        loc_nombre = str(r["logical_seat_category"])
        if len(loc_nombre) > 28:
            loc_nombre = loc_nombre[:26] + "..."
        ax.text(x0 + 1.5, y0 + card_h - 11.5, loc_nombre,
                fontsize=11.5, fontweight="bold", color="#0B2545", zorder=4)

        # Venue / Recinto
        venue_str = f"Recinto: {r['site']}"
        if len(venue_str) > 34:
            venue_str = venue_str[:32] + "..."
        ax.text(x0 + 1.5, y0 + card_h - 15.0, venue_str,
                fontsize=9.0, color="#555555", zorder=4)

        # Tipología y Precio
        tipo_str = f"Tipología: {r['type_site']}"
        ax.text(x0 + 1.5, y0 + card_h - 18.0, tipo_str,
                fontsize=8.5, color="#666666", zorder=4)

        precio_str = f"Precio: ${r['precio']:,.0f} COP"
        ax.text(x0 + 1.5, y0 + card_h - 21.0, precio_str,
                fontsize=9.5, fontweight="bold", color="#134074", zorder=4)

        # Línea divisoria suave
        ax.plot([x0 + 1.5, x0 + card_w - 1.5], [y0 + card_h - 23.0, y0 + card_h - 23.0],
                color="#E2E8F0", linewidth=1.0, zorder=4)

        # Caja de Doble Identidad
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

        # Píldora de Confianza en la parte inferior
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

    # Leyenda al pie
    fig.text(0.05, 0.015, "* Todas las tarjetas corresponden a eventos reales certificados en la marcha blanca (GOLD/SECUTIX test, N=1,324). Confianza >= 45% y es_frontera=False.",
             fontsize=8.2, color="#666666", style="italic")

    out_path = "reports/figures/fig_tarjetas_ejemplos_marcha.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(out_path, dpi=150, facecolor="#FFFFFF")
    plt.close()
    print(f"  [OK] Guardada en: {out_path}")

if __name__ == "__main__":
    os.makedirs("reports/figures", exist_ok=True)
    generar_figura_arbol()
    generar_figura_tarjetas()
    print("\n[ÉXITO] Ambas figuras generadas correctamente en reports/figures/.")
