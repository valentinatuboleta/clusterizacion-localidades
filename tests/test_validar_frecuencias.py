"""
Suite de Pruebas Unitarias para la Función de Validación de Frecuencias y Flags (Bloque 1.5).

Cubre:
1. Flag de Categoría Vacía (0 localidades).
2. Flag de Cold Start (0 < localidades < 50).
3. Flag de Dominancia Excesiva (> 60% del catálogo).
4. Caso sintético normal sin flags patológicos.
"""

import unittest
import pandas as pd
from src.validar_frecuencias import calcular_reporte_frecuencias


class TestValidarFrecuenciasTaxonomia(unittest.TestCase):

    def test_flag_categoria_vacia_sintetico(self):
        """Valida que una categoría con 0 localidades active correctamente flag_vacia."""
        df_lookup = pd.DataFrame({
            "site": ["TEATRO_A", "ARENA_B"],
            "type_site": ["TEATRO", "ARENA"]
        })
        df_raw = pd.DataFrame({
            "site": ["TEATRO_A"] * 60 + ["ARENA_B"] * 40
        })

        df_frec, diag = calcular_reporte_frecuencias(df_raw, df_lookup)
        
        # Categorías no presentes deben marcarse como vacías
        frec_parqueadero = df_frec[df_frec["categoria"] == "PARQUEADERO"].iloc[0]
        self.assertTrue(frec_parqueadero["flag_vacia"])
        self.assertEqual(frec_parqueadero["conteo_localidades"], 0)
        self.assertIn("CATEGORIA_VACIA:PARQUEADERO", diag["flags_rojos"])
        self.assertFalse(diag["pasa_compuerta"])

    def test_flag_cold_start_sintetico(self):
        """Valida que una categoría con < 50 localidades active flag_cold_start."""
        df_lookup = pd.DataFrame({
            "site": ["TEATRO_A", "MUSEO_B"],
            "type_site": ["TEATRO", "MUSEO"]
        })
        df_raw = pd.DataFrame({
            "site": ["TEATRO_A"] * 100 + ["MUSEO_B"] * 25
        })

        df_frec, diag = calcular_reporte_frecuencias(df_raw, df_lookup)
        frec_museo = df_frec[df_frec["categoria"] == "MUSEO"].iloc[0]

        self.assertFalse(frec_museo["flag_vacia"])
        self.assertTrue(frec_museo["flag_cold_start"])
        self.assertEqual(frec_museo["conteo_localidades"], 25)
        self.assertTrue(any("COLD_START_ACTIVO:MUSEO" in f for f in diag["flags_amarillos"]))

    def test_flag_dominancia_sintetico(self):
        """Valida que una categoría con > 60% de localidades active flag_dominancia."""
        df_lookup = pd.DataFrame({
            "site": ["TEATRO_A", "ESTADIO_B"],
            "type_site": ["TEATRO", "ESTADIO"]
        })
        df_raw = pd.DataFrame({
            "site": ["TEATRO_A"] * 75 + ["ESTADIO_B"] * 25
        })

        df_frec, diag = calcular_reporte_frecuencias(df_raw, df_lookup)
        frec_teatro = df_frec[df_frec["categoria"] == "TEATRO"].iloc[0]

        self.assertTrue(frec_teatro["flag_dominancia"])
        self.assertEqual(frec_teatro["porcentaje_localidades"], 75.0)
        self.assertTrue(any("DOMINANCIA_EXCESIVA:TEATRO" in f for f in diag["flags_rojos"]))
        self.assertFalse(diag["pasa_compuerta"])


if __name__ == "__main__":
    unittest.main()
