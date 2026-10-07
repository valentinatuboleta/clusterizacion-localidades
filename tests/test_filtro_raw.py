import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import filtrar_consistencia_localidades


class TestFiltradoConsistenciaLocalidades:
    """
    Tests herméticos con datos sintéticos para validar las reglas de consistencia de filtrar_consistencia_localidades:
    1. Deduplicación exacta por clave de negocio vs preservación entre eventos distintos (Hallazgo 3).
    2. Eliminación de categorías lógicas vacías, nulas o con whitespace.
    3. Eliminación de precios <= 0, nulos o negativos (consistente con > 0).
    4. Eliminación de aforos no positivos o cantidades negativas.
    5. Consistencia de aforo total de evento (suma dn_quota == performance_quota).
    6. Verificación exacta de los conteos reportados en el diccionario de métricas.
    """

    @pytest.fixture
    def df_base_valido(self):
        """Genera un DataFrame sintético mínimo consistente."""
        return pd.DataFrame({
            "t_performance_id": [101, 101],
            "site": ["MOVISTAR ARENA", "MOVISTAR ARENA"],
            "logical_seat_category": ["PLATEA 101", "SEGUNDO PISO 201"],
            "dn_quota": [500, 500],
            "performance_quota": [1000, 1000],
            "med_base_unit_amt_itx": [150000.0, 80000.0],
            "net_sold_p_qty": [400.0, 350.0],
            "net_sold_c_qty": [20.0, 10.0]
        })

    def test_duplicados_exactos_eliminados_y_eventos_distintos_preservados(self, df_base_valido):
        """
        Valida que duplicados exactos en el mismo evento se eliminen,
        pero la misma localidad en dos eventos distintos (Hallazgo 3) se preserve intacta.
        """
        # Fila duplicada exacta para el evento 101
        fila_clon = df_base_valido.iloc[[0]].copy()

        # Fila legítima con la MISMA localidad pero en un EVENTO DISTINTO (102)
        evento_102 = pd.DataFrame({
            "t_performance_id": [102],
            "site": ["MOVISTAR ARENA"],
            "logical_seat_category": ["PLATEA 101"],  # Misma localidad, evento distinto
            "dn_quota": [500],
            "performance_quota": [500],
            "med_base_unit_amt_itx": [150000.0],
            "net_sold_p_qty": [450.0],
            "net_sold_c_qty": [10.0]
        })

        df_test = pd.concat([df_base_valido, fila_clon, evento_102], ignore_index=True)
        # Total inicial: 2 base + 1 clon + 1 evento 102 = 4 filas

        df_clean, rep = filtrar_consistencia_localidades(df_test, retornar_reporte=True, verbose=False)

        # Debe eliminar el clon del evento 101
        assert rep["regla_deduplicacion_clave_negocio"] == 1
        # La localidad en el evento 102 debe preservarse
        assert len(df_clean[df_clean["t_performance_id"] == 102]) == 1
        assert df_clean[df_clean["t_performance_id"] == 102]["logical_seat_category"].iloc[0] == "PLATEA 101"
        # Total final debe ser 3
        assert len(df_clean) == 3

    def test_categoria_vacia_o_nula_eliminada(self, df_base_valido):
        """Valida que filas con categoría nula, vacía, espacios o strings espurios sean descartadas."""
        filas_invalidas = pd.DataFrame({
            "t_performance_id": [103, 103, 103, 103],
            "site": ["TEATRO MAYOR"] * 4,
            "logical_seat_category": [None, "", "   ", "nan"],
            "dn_quota": [200, 200, 200, 200],
            "performance_quota": [800, 800, 800, 800],
            "med_base_unit_amt_itx": [100000.0] * 4,
            "net_sold_p_qty": [100.0] * 4,
            "net_sold_c_qty": [0.0] * 4
        })

        df_test = pd.concat([df_base_valido, filas_invalidas], ignore_index=True)
        df_clean, rep = filtrar_consistencia_localidades(df_test, retornar_reporte=True, verbose=False)

        assert rep["regla_categoria_no_vacia"] == 4
        assert not df_clean["logical_seat_category"].isna().any()
        assert not (df_clean["logical_seat_category"].str.strip() == "").any()
        assert not df_clean["logical_seat_category"].str.lower().isin(["nan", "none", "null"]).any()

    def test_precio_cero_nan_y_negativo_eliminados(self, df_base_valido):
        """
        Valida que precios 0, NaN o negativos sean eliminados de forma consistente con > 0.
        Decisión documentada: precio negativo se elimina como inválido.
        """
        filas_precio_malo = pd.DataFrame({
            "t_performance_id": [104, 105, 106],
            "site": ["ESTADIO"] * 3,
            "logical_seat_category": ["OCCIDENTAL ALTA", "ORIENTAL BAJA", "NORTE"],
            "dn_quota": [1000, 1000, 1000],
            "performance_quota": [1000, 1000, 1000],
            "med_base_unit_amt_itx": [0.0, np.nan, -50000.0],  # 0, NaN, Negativo
            "net_sold_p_qty": [10.0] * 3,
            "net_sold_c_qty": [0.0] * 3
        })

        df_test = pd.concat([df_base_valido, filas_precio_malo], ignore_index=True)
        df_clean, rep = filtrar_consistencia_localidades(df_test, retornar_reporte=True, verbose=False)

        assert rep["regla_precio_positivo"] == 3
        assert (df_clean["med_base_unit_amt_itx"] > 0).all()
        assert df_clean["med_base_unit_amt_itx"].notna().all()

    def test_columna_precio_inexistente_genera_warning_y_continua(self, df_base_valido, caplog):
        """Valida que si no existe columna de precio, se registre un warning y continúe sin fallar."""
        df_sin_precio = df_base_valido.drop(columns=["med_base_unit_amt_itx"])
        
        with caplog.at_level("WARNING"):
            df_clean, rep = filtrar_consistencia_localidades(df_sin_precio, retornar_reporte=True, verbose=False)

        assert rep["regla_precio_positivo"] == 0
        assert "Columna de precio no encontrada" in caplog.text
        assert len(df_clean) == len(df_sin_precio)

    def test_aforos_y_cantidades_no_negativas(self, df_base_valido):
        """Valida dn_quota <= 0, performance_quota <= 0 y cantidades net_sold < 0."""
        filas_invalidas = pd.DataFrame({
            "t_performance_id": [107, 108, 109],
            "site": ["SALA"] * 3,
            "logical_seat_category": ["PALCO 1", "PALCO 2", "PALCO 3"],
            "dn_quota": [0, 500, 500],  # dn_quota = 0
            "performance_quota": [1000, 0, 500],  # perf_quota = 0
            "med_base_unit_amt_itx": [100000.0] * 3,
            "net_sold_p_qty": [50.0, 50.0, -5.0],  # net_sold_p_qty negativo
            "net_sold_c_qty": [0.0, 0.0, 0.0]
        })

        df_test = pd.concat([df_base_valido, filas_invalidas], ignore_index=True)
        df_clean, rep = filtrar_consistencia_localidades(df_test, retornar_reporte=True, verbose=False)

        assert rep["regla_dn_quota_positiva"] == 1
        assert rep["regla_performance_quota_positiva"] == 1
        assert rep["regla_cantidades_no_negativas"] == 1

    def test_consistencia_aforo_evento(self, df_base_valido):
        """Valida que si la suma de dn_quota no coincide con performance_quota, el evento sea descartado."""
        evento_incoherente = pd.DataFrame({
            "t_performance_id": [999, 999],
            "site": ["COLISEO", "COLISEO"],
            "logical_seat_category": ["ZONA A", "ZONA B"],
            "dn_quota": [300, 300],  # suma = 600
            "performance_quota": [1000, 1000],  # aforo total esperado = 1000 != 600
            "med_base_unit_amt_itx": [50000.0, 50000.0],
            "net_sold_p_qty": [200.0, 200.0],
            "net_sold_c_qty": [0.0, 0.0]
        })

        df_test = pd.concat([df_base_valido, evento_incoherente], ignore_index=True)
        df_clean, rep = filtrar_consistencia_localidades(df_test, retornar_reporte=True, verbose=False)

        assert rep["regla_consistencia_aforo_evento"] == 2
        assert 999 not in df_clean["t_performance_id"].values

    def test_reporte_conteos_exactos_e_integridad(self, df_base_valido):
        """Valida que la suma de eliminadas y totales cuadre exactamente con len inicial y final."""
        # Generar un dataset mixto con fallas conocidas
        f_cat = df_base_valido.iloc[[0]].copy()
        f_cat["logical_seat_category"] = ""

        f_precio = df_base_valido.iloc[[0]].copy()
        f_precio["med_base_unit_amt_itx"] = -10.0

        f_clon = df_base_valido.iloc[[1]].copy()

        df_test = pd.concat([df_base_valido, f_cat, f_precio, f_clon], ignore_index=True)
        n_ini = len(df_test)

        df_clean, rep = filtrar_consistencia_localidades(df_test, retornar_reporte=True, verbose=False)

        assert rep["total_inicial"] == n_ini
        assert rep["total_final"] == len(df_clean)
        assert rep["total_eliminadas"] == n_ini - len(df_clean)
        assert "regla_exclusion_patrones_producto" in rep
        assert "regla_ventas_no_exceden_aforo" in rep
        assert df_clean.attrs["reporte_filtro"] == rep

    def test_regla_8_exclusion_patrones_producto(self):
        """
        Valida la Regla 8: Exclusión por patrones en el nombre del producto (Spark mirror).
        - 'BOLETO TEST EVENTO': eliminado por contener 'TEST'.
        - 'PARQUEADERO CONCIERTO': eliminado por contener 'PARQUEA' (causa raíz del bug en marcha blanca).
        - 'EVENTO CANCELADO FINAL': eliminado por contener 'CANCELAD'.
        - 'LOCALIDAD NO USAR': eliminado por contener 'NO USAR'.
        - 'OBRA TEATRAL CANONICA': preservado (producto legítimo sin patrones).
        
        Nota metodológica:
        El comportamiento nativo heredado de Spark (~F.upper(col('product')).contains('...')) opera
        mediante coincidencia literal de subcadena (regex=False). Por tanto, palabra completa NO es el criterio:
        cualquier subcadena que coincida con los patrones descartará la fila (e.g. 'PARQUEA' descarta 'PARQUEADERO'
        y 'TEST' descarta subcadenas como 'TESTAMENTO').
        """
        df_prod = pd.DataFrame({
            "t_performance_id": [201, 202, 203, 204, 205, 206],
            "site": ["TEATRO"] * 6,
            "product": [
                "BOLETO TEST EVENTO",
                "PARQUEADERO CONCIERTO",
                "EVENTO CANCELADO FINAL",
                "LOCALIDAD NO USAR",
                "OBRA TEATRAL CANONICA",
                "TESTAMENTO TEATRAL"
            ],
            "logical_seat_category": ["PLATEA"] * 6,
            "dn_quota": [100] * 6,
            "performance_quota": [100] * 6,
            "med_base_unit_amt_itx": [50000.0] * 6,
            "net_sold_p_qty": [50.0] * 6,
            "net_sold_c_qty": [0.0] * 6
        })

        df_clean, rep = filtrar_consistencia_localidades(
            df_prod,
            col_producto="product",
            patrones_exclusion=("TEST", "CANCELAD", "PARQUEA", "NO USAR"),
            retornar_reporte=True,
            verbose=False
        )

        # 5 eliminadas (TEST x2, PARQUEA, CANCELAD, NO USAR) y 1 preservada (OBRA TEATRAL CANONICA)
        assert rep["regla_exclusion_patrones_producto"] == 5
        assert len(df_clean) == 1
        assert df_clean["product"].iloc[0] == "OBRA TEATRAL CANONICA"
        assert "BOLETO TEST EVENTO" not in df_clean["product"].values
        assert "PARQUEADERO CONCIERTO" not in df_clean["product"].values

    def test_regla_9_ventas_no_exceden_aforo(self):
        """
        Valida la Regla 9: Ventas pagadas no exceden aforo físico (net_sold_p_qty <= dn_quota).
        - net_sold_p_qty = dn_quota + 1 -> eliminado.
        - net_sold_p_qty = dn_quota -> preservado.
        - net_sold_p_qty < dn_quota -> preservado.
        """
        df_ventas = pd.DataFrame({
            "t_performance_id": [301, 302, 303],
            "site": ["ARENA"] * 3,
            "product": ["CONCIERTO A", "CONCIERTO B", "CONCIERTO C"],
            "logical_seat_category": ["PREFERENCIAL"] * 3,
            "dn_quota": [100, 100, 100],
            "performance_quota": [100, 100, 100],
            "med_base_unit_amt_itx": [80000.0] * 3,
            "net_sold_p_qty": [101.0, 100.0, 80.0],  # 101 > 100 (invalido), 100 == 100 (valido), 80 < 100 (valido)
            "net_sold_c_qty": [0.0, 0.0, 0.0]
        })

        df_clean, rep = filtrar_consistencia_localidades(df_ventas, retornar_reporte=True, verbose=False)

        assert rep["regla_ventas_no_exceden_aforo"] == 1
        assert len(df_clean) == 2
        assert 301 not in df_clean["t_performance_id"].values
        assert 302 in df_clean["t_performance_id"].values
        assert 303 in df_clean["t_performance_id"].values

    def test_modo_legacy_inmune_a_reglas_8_y_9(self):
        """
        Valida que el modo legacy (modo_legacy_v3=True) sea estrictamente inmune a las reglas 8 y 9,
        reproduciendo la definición histórica de v2.5 (33,775 filas en dataset crudo).
        """
        # Caso 1: Sintético con patrones de producto que en modo estricto se descartarían
        df_sintetico = pd.DataFrame({
            "t_performance_id": [401],
            "site": ["TEATRO"],
            "product": ["EVENTO TEST"],
            "logical_seat_category": ["PLATEA 1"],
            "dn_quota": [50],
            "performance_quota": [50],
            "med_base_unit_amt_itx": [40000.0],
            "net_sold_p_qty": [10.0],
            "net_sold_c_qty": [0.0]
        })

        df_leg, rep_leg = filtrar_consistencia_localidades(
            df_sintetico,
            modo_legacy_v3=True,
            retornar_reporte=True,
            verbose=False
        )
        assert len(df_leg) == 1
        assert rep_leg["modo"] == "legacy_v3"
        assert rep_leg["total_eliminadas"] == 0

        # Caso 2: Dataset real si existe
        import os
        ruta_real = "data/raw/localidades_eda.parquet"
        if os.path.exists(ruta_real):
            df_real = pd.read_parquet(ruta_real)
            df_real_clean = filtrar_consistencia_localidades(df_real, modo_legacy_v3=True, verbose=False)
            assert len(df_real_clean) == 33775, f"Regresión legacy: esperado 33,775, obtenido {len(df_real_clean)}"

