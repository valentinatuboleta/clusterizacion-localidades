# DEPRECATED: La fuente de verdad analítica es src.validar_frecuencias
import warnings
from src.validar_frecuencias import calcular_reporte_frecuencias

warnings.warn("scripts/validar_frecuencias_taxonomy.py está deprecado; use src/validar_frecuencias.py", DeprecationWarning, stacklevel=2)
