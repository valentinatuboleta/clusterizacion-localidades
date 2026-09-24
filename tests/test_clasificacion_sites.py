"""
Suite de Pruebas Automatizadas para Clasificación Léxica de Venues y Casos Borde.

Valida:
1. "SALA 2" genérico va a 'otro' / revisión (no a 'cine_sala_cultural').
2. "VIA 40 BARRANQUILLA" se clasifica como 'parque_aire_libre' (no como 'bar_club' por BARRANQUILLA).
3. "PARQUEADERO" se clasifica como 'otro' (no como 'parque_aire_libre').
4. "BIBLOS CAR WASH" se clasifica como 'otro' (no como 'parque_aire_libre').
5. Normalización estricta y coincidencia por tokens con límites de palabra.
6. Trazabilidad ética: ningún registro en site_type_lookup.csv tiene fuente 'ia_consenso'.
"""

import os
import unittest
import pandas as pd
from scripts.clasificar_sites import (
    normalizar_recinto,
    pasaje_a_clasificar,
    pasaje_b_clasificar
)


class TestClasificacionVenuesCasosBorde(unittest.TestCase):

    def test_caso_borde_sala_2_generico(self):
        """
        'SALA 2' es un nombre genérico: no debe ser categorizado como cine_sala_cultural
        a menos que indique explícitamente Cinemateca.
        """
        tipo_a, conf_a = pasaje_a_clasificar("SALA 2", aforo_max=100)
        tipo_b, conf_b = pasaje_b_clasificar("SALA 2", aforo_max=100)

        # No debe confundirse con 'SALA 2 CINEMATECA'
        self.assertNotEqual(tipo_a, "cine_sala_cultural", "SALA 2 genérico no debe ser cine_sala_cultural en Pasada A")
        self.assertNotEqual(tipo_b, "cine_sala_cultural", "SALA 2 genérico no debe ser cine_sala_cultural en Pasada B")

        # Debe ir a 'otro'
        self.assertEqual(tipo_a, "otro")
        self.assertEqual(tipo_b, "otro")

        # Sala 2 con Cinemateca explícita sí debe ser cine_sala_cultural
        tipo_cine, conf_cine = pasaje_a_clasificar("SALA 2 CINEMATECA", aforo_max=100)
        self.assertEqual(tipo_cine, "cine_sala_cultural")

    def test_caso_borde_via_40_barranquilla(self):
        """
        'VIA 40 BARRANQUILLA': la subcadena 'BAR' dentro de 'BARRANQUILLA' no debe
        activar erradamente la categoría 'bar_club'. Debe ser 'parque_aire_libre'.
        """
        tipo_a, conf_a = pasaje_a_clasificar("VIA 40 BARRANQUILLA", aforo_max=5000)
        tipo_b, conf_b = pasaje_b_clasificar("VIA 40 BARRANQUILLA", aforo_max=5000)

        self.assertNotEqual(tipo_a, "bar_club", "VIA 40 BARRANQUILLA no debe ser bar_club en Pasada A")
        self.assertNotEqual(tipo_b, "bar_club", "VIA 40 BARRANQUILLA no debe ser bar_club en Pasada B")

        self.assertEqual(tipo_a, "parque_aire_libre")
        self.assertEqual(tipo_b, "parque_aire_libre")

    def test_caso_borde_parqueadero(self):
        """
        'PARQUEADERO': no debe activar la categoría 'parque_aire_libre'.
        """
        for nombre in ["PARQUEADERO", "PARQUEADERO CENTRAL", "PARKING NORTE"]:
            tipo_a, _ = pasaje_a_clasificar(nombre, aforo_max=300)
            tipo_b, _ = pasaje_b_clasificar(nombre, aforo_max=300)

            self.assertNotEqual(tipo_a, "parque_aire_libre", f"{nombre} no debe ser parque_aire_libre en Pasada A")
            self.assertNotEqual(tipo_b, "parque_aire_libre", f"{nombre} no debe ser parque_aire_libre en Pasada B")
            self.assertEqual(tipo_a, "otro")
            self.assertEqual(tipo_b, "otro")

    def test_caso_borde_biblos_car_wash(self):
        """
        'BIBLOS CAR WASH': debe ir a 'otro' (o revisión), nunca a 'parque_aire_libre'.
        """
        tipo_a, _ = pasaje_a_clasificar("BIBLOS CAR WASH", aforo_max=6200)
        tipo_b, _ = pasaje_b_clasificar("BIBLOS CAR WASH", aforo_max=6200)

        self.assertNotEqual(tipo_a, "parque_aire_libre", "BIBLOS CAR WASH no debe ser parque_aire_libre")
        self.assertNotEqual(tipo_b, "parque_aire_libre", "BIBLOS CAR WASH no debe ser parque_aire_libre")
        self.assertEqual(tipo_a, "otro")
        self.assertEqual(tipo_b, "otro")

    def test_normalizacion_recinto(self):
        """
        Valida que la normalización colapse tildes, mayúsculas y espacios múltiples.
        """
        norm = normalizar_recinto("   Teatro  Colón   Bogotá  ")
        self.assertEqual(norm, "TEATRO COLON BOGOTA")

    def test_trazabilidad_fuente_lookup(self):
        """
        Audita que en site_type_lookup.csv ninguna fila tenga fuente 'ia_consenso',
        garantizando honestidad en la trazabilidad de datos.
        """
        lookup_path = "data/lookup/site_type_lookup.csv"
        if not os.path.exists(lookup_path):
            self.skipTest("No se encontró site_type_lookup.csv")

        df_lookup = pd.read_csv(lookup_path)
        fuentes = set(df_lookup["fuente"].dropna().unique())

        self.assertNotIn("ia_consenso", fuentes, "No debe existir la fuente ficticia 'ia_consenso'")
        self.assertTrue(fuentes.issubset({"reglas_heuristicas", "revision_humana"}))


if __name__ == "__main__":
    unittest.main()
