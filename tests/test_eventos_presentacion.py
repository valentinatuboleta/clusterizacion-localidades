"""
Tests herméticos para la selección de 5 eventos por quintil de aforo
y generación de figuras de eventos y frecuencias (Bloques 1 y 2).
"""

import os
import pytest
import pandas as pd

from scripts.seleccionar_eventos_presentacion import (
    seleccionar_5_eventos,
    clasificar_estrato_aforo,
    ESTRATOS_OBJETIVO
)

CSV_EVENTOS_PATH = "reports/eventos_seleccionados.csv"
FIG_EVENTOS = [
    "reports/figures/fig_evento_micro.png",
    "reports/figures/fig_evento_pequeno.png",
    "reports/figures/fig_evento_mediano.png",
    "reports/figures/fig_evento_grande.png",
    "reports/figures/fig_evento_estadio.png",
    "reports/figures/fig_frecuencia_microclusters.png"
]


def test_existencia_csv_eventos():
    """Valida que el archivo reports/eventos_seleccionados.csv exista."""
    assert os.path.exists(CSV_EVENTOS_PATH), f"No existe {CSV_EVENTOS_PATH}"


def test_esquema_columnas_eventos():
    """Valida que contenga exactamente las columnas requeridas por el contrato."""
    df = pd.read_csv(CSV_EVENTOS_PATH)
    columnas_esperadas = {
        "t_performance_id", "product", "site",
        "performance_quota", "n_localidades",
        "estrato_objetivo", "estrato_real"
    }
    assert columnas_esperadas.issubset(set(df.columns)), (
        f"Columnas faltantes en {CSV_EVENTOS_PATH}. Esperadas: {columnas_esperadas}"
    )


def test_conteo_exacto_5_eventos():
    """Valida que se hayan seleccionado exactamente 5 eventos (uno por estrato)."""
    df = pd.read_csv(CSV_EVENTOS_PATH)
    assert len(df) == 5, f"Se esperaban 5 eventos, pero hay {len(df)}"


def test_ausencia_duplicados_eventos_y_recintos():
    """Valida que no haya eventos repetidos ni recintos duplicados entre los 5 seleccionados."""
    df = pd.read_csv(CSV_EVENTOS_PATH)
    assert df["t_performance_id"].nunique() == 5, "Existen t_performance_id duplicados"
    assert df["site"].nunique() == 5, "Existen recintos (site) duplicados entre los 5 eventos"


def test_criterio_minimo_localidades():
    """Valida que cada uno de los 5 eventos tenga al menos 4 localidades evaluadas."""
    df = pd.read_csv(CSV_EVENTOS_PATH)
    assert (df["n_localidades"] >= 4).all(), (
        f"Uno o más eventos tienen menos de 4 localidades: {df[['product', 'n_localidades']].to_dict('records')}"
    )


def test_estratos_correctos_y_concordancia_aforo():
    """
    Valida que los 5 estratos objetivo estén cubiertos:
      - micro: <= 500
      - pequeno: 500 - 2,000
      - mediano: 2,000 - 8,000
      - grande: 8,000 - 20,000
      - estadio: > 20,000
    """
    df = pd.read_csv(CSV_EVENTOS_PATH)
    estratos_esperados = {"micro", "pequeno", "mediano", "grande", "estadio"}
    assert set(df["estrato_objetivo"]) == estratos_esperados, (
        f"Estratos no coinciden: {set(df['estrato_objetivo'])} vs {estratos_esperados}"
    )

    for _, r in df.iterrows():
        est_obj = r["estrato_objetivo"]
        quota = float(r["performance_quota"])
        est_calc = clasificar_estrato_aforo(quota)
        assert r["estrato_real"] == est_calc, (
            f"estrato_real '{r['estrato_real']}' no coincide con aforo {quota} ({est_calc})"
        )


def test_existencia_figuras_generadas():
    """Valida que las figuras de escaleras por evento y frecuencia de microclusters existan."""
    for fig_path in FIG_EVENTOS:
        assert os.path.exists(fig_path), f"No existe la figura {fig_path}"
        assert os.path.getsize(fig_path) > 10000, f"Figura {fig_path} sospechosamente pequeña (<10KB)"
