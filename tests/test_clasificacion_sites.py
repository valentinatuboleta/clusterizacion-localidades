"""
Suite de Pruebas Automatizadas para Clasificación Léxica de Venues y Casos Borde (Taxonomía v2).

Valida:
1. "SALA 2" genérico va a 'OTROS_RECINTOS' / revisión (no a 'CINEMATECA').
2. "VIA 40 BARRANQUILLA" se clasifica como 'PARQUE' (no como 'RESTAURANTE' por BARRANQUILLA).
3. "PARQUEADERO" se clasifica como 'PARQUEADERO' (no como 'PARQUE').
4. "BIBLOS CAR WASH" se clasifica como 'OTROS_RECINTOS' (no como 'PARQUE').
5. Normalización estricta y coincidencia por tokens con límites de palabra.
6. Trazabilidad ética: ningún registro en site_type_lookup.csv tiene fuente 'ia_consenso'.
7. Nuevos tests: completitud de mapeo, 10 categorías canónicas y trazabilidad v2.
"""

import os
import unittest
import pandas as pd
from scripts.clasificar_sites import (
    normalizar_recinto,
    pasaje_a_clasificar,
    pasaje_b_clasificar,
    CATEGORIAS_VALIDAS
)
from src.clustering import (
    CANONICAL_TYPE_SITE_CATEGORIES,
    MAPEO_TAXONOMIA_V1_A_V2
)


class TestClasificacionVenuesCasosBorde(unittest.TestCase):

    def test_caso_borde_sala_2_generico(self):
        """
        'SALA 2' es un nombre genérico: no debe ser categorizado como CINEMATECA
        a menos que indique explícitamente Cinemateca.
        """
        tipo_a, conf_a = pasaje_a_clasificar("SALA 2", aforo_max=100)
        tipo_b, conf_b = pasaje_b_clasificar("SALA 2", aforo_max=100)

        # No debe confundirse con 'SALA 2 CINEMATECA'
        self.assertNotEqual(tipo_a, "CINEMATECA", "SALA 2 genérico no debe ser CINEMATECA en Pasada A")
        self.assertNotEqual(tipo_b, "CINEMATECA", "SALA 2 genérico no debe ser CINEMATECA en Pasada B")

        # Debe ir a OTROS_RECINTOS
        self.assertEqual(tipo_a, "OTROS_RECINTOS")
        self.assertEqual(tipo_b, "OTROS_RECINTOS")

        # Sala 2 con Cinemateca explícita sí debe ser CINEMATECA
        tipo_cine, conf_cine = pasaje_a_clasificar("SALA 2 CINEMATECA", aforo_max=100)
        self.assertEqual(tipo_cine, "CINEMATECA")

    def test_caso_borde_via_40_barranquilla(self):
        """
        'VIA 40 BARRANQUILLA': la subcadena 'BAR' dentro de 'BARRANQUILLA' no debe
        activar erradamente la categoría 'RESTAURANTE'. Debe ser 'PARQUE'.
        """
        tipo_a, conf_a = pasaje_a_clasificar("VIA 40 BARRANQUILLA", aforo_max=5000)
        tipo_b, conf_b = pasaje_b_clasificar("VIA 40 BARRANQUILLA", aforo_max=5000)

        self.assertNotEqual(tipo_a, "RESTAURANTE", "VIA 40 BARRANQUILLA no debe ser RESTAURANTE en Pasada A")
        self.assertNotEqual(tipo_b, "RESTAURANTE", "VIA 40 BARRANQUILLA no debe ser RESTAURANTE en Pasada B")

        self.assertEqual(tipo_a, "PARQUE")
        self.assertEqual(tipo_b, "PARQUE")

    def test_caso_borde_parqueadero(self):
        """
        'PARQUEADERO': debe activar la categoría 'PARQUEADERO' (no 'PARQUE' ni 'otro').
        """
        for nombre in ["PARQUEADERO", "PARQUEADERO CENTRAL", "PARKING NORTE"]:
            tipo_a, _ = pasaje_a_clasificar(nombre, aforo_max=300)
            tipo_b, _ = pasaje_b_clasificar(nombre, aforo_max=300)

            self.assertNotEqual(tipo_a, "PARQUE", f"{nombre} no debe ser PARQUE en Pasada A")
            self.assertNotEqual(tipo_b, "PARQUE", f"{nombre} no debe ser PARQUE en Pasada B")
            self.assertEqual(tipo_a, "PARQUEADERO")
            self.assertEqual(tipo_b, "PARQUEADERO")

    def test_caso_borde_biblos_car_wash(self):
        """
        'BIBLOS CAR WASH': debe ir a 'OTROS_RECINTOS', nunca a 'PARQUE'.
        """
        tipo_a, _ = pasaje_a_clasificar("BIBLOS CAR WASH", aforo_max=6200)
        tipo_b, _ = pasaje_b_clasificar("BIBLOS CAR WASH", aforo_max=6200)

        self.assertNotEqual(tipo_a, "PARQUE", "BIBLOS CAR WASH no debe ser PARQUE")
        self.assertNotEqual(tipo_b, "PARQUE", "BIBLOS CAR WASH no debe ser PARQUE")
        self.assertEqual(tipo_a, "OTROS_RECINTOS")
        self.assertEqual(tipo_b, "OTROS_RECINTOS")

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

    def test_completitud_mapeo_taxonomia(self):
        """Valida que toda categoría vieja mapea a una categoría destino válida y sin elementos huérfanos."""
        categorias_esperadas_v1 = [
            "arena_cubierta", "cine_sala_cultural", "estadio_abierto", "coliseo",
            "teatro", "parque_aire_libre", "bar_club", "auditorio",
            "centro_eventos_carpa", "centro_convenciones", "sala_conciertos",
            "cabaret_comedia", "otro"
        ]
        for cat_v1 in categorias_esperadas_v1:
            self.assertIn(cat_v1, MAPEO_TAXONOMIA_V1_A_V2, f"Categoría {cat_v1} sin mapeo")
            destino = MAPEO_TAXONOMIA_V1_A_V2[cat_v1]
            if destino != "re-clasificar":
                self.assertIn(destino, CANONICAL_TYPE_SITE_CATEGORIES, f"Destino {destino} no es canónico")

    def test_parqueadero_a_parqueadero(self):
        """Valida explícitamente la regla de negocio: PARQUEADERO / PARKING mapea a PARQUEADERO."""
        for token in ["PARQUEADERO CORFERIAS", "PARKING AEROPUERTO", "ESTACIONAMIENTO NORTE"]:
            t_a, conf_a = pasaje_a_clasificar(token, 100)
            t_b, conf_b = pasaje_b_clasificar(token, 100)
            self.assertEqual(t_a, "PARQUEADERO")
            self.assertEqual(t_b, "PARQUEADERO")
            self.assertGreaterEqual(conf_a, 0.90)
            self.assertGreaterEqual(conf_b, 0.90)

    def test_presencia_10_categorias_canonicas(self):
        """Valida que CANONICAL_TYPE_SITE_CATEGORIES contiene exactamente las 10 categorías de negocio."""
        esperadas = {
            "ARENA", "CINEMATECA", "ESTADIO", "COLISEO", "MUSEO",
            "OTROS_RECINTOS", "PARQUE", "PARQUEADERO", "RESTAURANTE", "TEATRO"
        }
        self.assertEqual(set(CANONICAL_TYPE_SITE_CATEGORIES), esperadas)
        self.assertEqual(len(CANONICAL_TYPE_SITE_CATEGORIES), 10)
        self.assertNotIn("desconocido", CANONICAL_TYPE_SITE_CATEGORIES)



if __name__ == "__main__":
    unittest.main()
