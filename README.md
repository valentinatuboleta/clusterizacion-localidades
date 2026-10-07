# Clusterización de Localidades - TuBoleta

Proyecto integral de Data Science y Machine Learning para la segmentación y clasificación automatizada de localidades en espectáculos públicos a partir de datos transaccionales almacenados en formato `.parquet` en Azure Blob Storage.

El modelo implementa un **espacio vectorial mixto de 36 dimensiones** (Modelo v2.5: características numéricas relativas *ex-ante*, percentil dentro de tipología de venue, tags estructurales extraídos mediante NLP, tipología de venue one-hot ponderada con 10 categorías canónicas y representaciones TF-IDF) para agrupar el catálogo en **6 arquetipos estandarizados de demanda** a través de una arquitectura en dos etapas.

---

## Estructura del Repositorio

```text
clusterizacion-localidades/
├── .env.example                                # Plantilla segura de variables de entorno (Azure y Gemini)
├── .gitignore                                  # Exclusión de credenciales, datos y entornos
├── .pre-commit-config.yaml                     # Hook pre-commit con nbstripout
├── LICENSE                                     # Licencia MIT del proyecto
├── README.md                                   # Guía general de uso y arquitectura
├── requirements.txt                            # Dependencias locales (sin PySpark)
├── requirements-databricks.txt                 # Dependencias exclusivas para Databricks
├── DOCUMENTACION_MODELO_CLUSTERIZACION.md      # Especificación técnica y matemática exhaustiva (v2.5)
│
├── .github/                                    # Integración Continua (CI)
│   └── workflows/
│       └── ci.yml                              # Pipeline automatizado de GitHub Actions
│
├── data/                                       # Datos locales y tablas maestras
│   ├── lookup/                                 # Tablas maestras de venues y trazabilidad
│   │   ├── recintos_unicos.csv                 # 494 venues únicos extraídos del catálogo
│   │   ├── site_type_lookup.csv                # Tabla de verdad consolidada v2 (type_site)
│   │   └── site_type_revision_humana.csv       # Discrepancias enviadas a curaduría humana
│   ├── raw/                                    # Parquets descargados de Azure
│   └── processed/                              # Datasets con features y clusters asignados
│
├── notebooks/                                  # Flujo interactivo paso a paso
│   ├── 00_databricks_raw_data.ipynb            # Extracción y preparación inicial en Databricks
│   ├── 01_eda_clusterizacion.ipynb             # Análisis exploratorio, consistencia y 17 tags
│   ├── 02_clustering_espacio_mixto.ipynb       # Espacio mixto (36D), K-Means/GMM y arquetipos (v2.5)
│   └── 03_marcha_blanca_evaluacion.ipynb       # Protocolo y evaluación de Marcha Blanca (v3.0 / v2.5)
│
├── src/                                        # Módulos Python reutilizables de producción
│   ├── __init__.py
│   ├── azure_utils.py                          # Conexión y descarga segura desde Azure Blob Storage
│   ├── nlp_utils.py                            # Limpieza de marketing y extracción de 17 tags NLP
│   ├── feature_engineering.py                  # Normalización relativa por evento, consistencia y venue
│   ├── llm_classifier.py                       # Clasificador de venues con Gemini 3.8 Flash Medium y auditoría
│   ├── clustering.py                           # Espacio mixto 36D, clustering bietápico, persistencia e inferencia (v2.5)
│   ├── jerarquia.py                            # Arquitectura jerárquica, scoring y drift de micro-clusters (v3.0-hier.2)
│   └── validar_frecuencias.py                  # Compuerta de validación de frecuencias de taxonomía
│
├── tests/                                      # Suite de pruebas automatizadas y aseguramiento de calidad
│   ├── __init__.py
│   ├── test_clustering_golden_set.py           # Golden Set (20 casos), consistencia, persistencia y selector auto
│   ├── test_filtro_raw.py                      # Tests herméticos de reglas de filtrado y consistencia
│   ├── test_clasificacion_sites.py             # Casos borde toponímicos, límites de palabra y trazabilidad v2
│   ├── test_llm_classifier.py                  # Inferencia LLM hermética con mocks para CI
│   ├── test_feature_type_site.py               # Tests del feature type_site (10 categorías) y percentil
│   ├── test_validar_frecuencias.py             # Tests unitarios sintéticos de la compuerta de frecuencias
│   ├── test_microclusters.py                   # Tests de la exploración de micro-clusters
│   ├── test_jerarquia.py                       # Tests de arquitectura jerárquica v3.0 (rollup 1:1, Codo-DB)
│   └── test_diagnostico_subespacios.py         # Tests de diagnóstico de pureza, cobertura y oráculo
│
├── scripts/                                    # Automatización, diagnóstico y análisis
│   ├── clasificar_sites.py                     # Pipeline de clasificación de venues (Reglas + LLM + Humano)
│   ├── migrar_lookup_v2.py                     # Script de migración y trazabilidad de lookup v1 -> v2
│   ├── validar_frecuencias_taxonomy.py         # Validación formal de frecuencias y compuertas
│   ├── entrenar_v24.py                         # Re-entrenamiento, evaluación y persistencia modelo v2.4 / v2.5
│   ├── entrenar_jerarquia_microclusters.py     # Pipeline jerárquico v3.0-hier.2 (Nivel 1 v2.5 -> Nivel 2 micro-clusters)
│   ├── diagnosticar_subespacios.py             # Diagnóstico de separabilidad, oráculo y sweep condicional VIP
│   ├── comparar_v22_vs_v23.py                  # Comparación y ablación de versiones
│   ├── diagnostico_y_benchmark_avanzado.py     # Diagnóstico previo, sweep de pesos y benchmark de algoritmos
│   ├── optimizar_k_multizona.py                # Búsqueda formal de k óptimo (Codo Ortogonal + Davies-Bouldin)
│   ├── verificar_k5_perfiles.py                # Inspección de centroides y activación de tags
│   ├── monitorear_drift.py                     # Monitoreo PSI de drift
│   ├── explorar_microclusters.py               # Exploración de micro-clusters para normalización de nombres
│   ├── entrenar_jerarquia_microclusters.py     # Pipeline jerárquico v3.0 (Nivel 1 macro -> Nivel 2 micro-clusters)
│   ├── diagnosticar_subespacios.py             # Diagnóstico de separabilidad, oráculo y sweep condicional VIP
│   ├── build_presentation.py                   # Generación de presentación ejecutiva de EDA (10 diapositivas)
│   ├── build_presentation_from_template.py     # Inyección de insights en plantilla corporativa PPTX
│   ├── build_full_notebook_presentation.py     # Generación de presentación ejecutiva completa
│   ├── generate_all_presentation_figures.py    # Generación automatizada de figuras para reportes
│   └── generate_all_23_figures.py              # Suite exhaustiva de figuras analíticas (23 gráficos)
│
└── reports/                                    # Entregables ejecutivos
    └── figures/                                # Figuras generadas en alta resolución (PNG)
```

---

## Configuración Inicial

### 1. Clonar el repositorio y configurar entorno
```bash
git clone https://github.com/valentinatuboleta/clusterizacion-localidades.git
cd clusterizacion-localidades

python3 -m venv .venv
source .venv/bin/activate  # En macOS / Linux
# .venv\Scripts\activate   # En Windows
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar credenciales en `.env`
Copia la plantilla de ejemplo y completa con las credenciales de tu cuenta de Azure y Gemini:
```bash
cp .env.example .env
```

Edita `.env` con tus datos:
```env
AZURE_STORAGE_ACCOUNT_NAME="tu_cuenta_de_almacenamiento"
AZURE_STORAGE_ACCOUNT_KEY="tu_account_key"
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=tu_cuenta_de_almacenamiento;AccountKey=tu_account_key;EndpointSuffix=core.windows.net"
AZURE_CONTAINER_NAME="nombre_del_contenedor"
AZURE_BLOB_NAME="ruta/al/archivo_datos.parquet"

# Agente LLM Gemini (Opcional para inferencia con IA)
GEMINI_API_KEY="tu_gemini_api_key"
GEMINI_MODEL="gemini-3.8-flash-medium"
```

### 4. Clasificación y Estandarización de Venues (`type_site`)
El sistema cuenta con un esquema de precedencia de 3 niveles: `revision_humana > llm > reglas_heuristicas`.

* **Modo Léxico Determinístico (por defecto / sin consumo de API):**
  ```bash
  python scripts/clasificar_sites.py
  ```
  Ejecuta la doble pasada ortogonal con límites estrictos de palabra (`\b`).

* **Modo Agente LLM (Gemini 3.8 Flash Medium, temp=0, salida JSON):**
  ```bash
  python scripts/clasificar_sites.py --llm
  ```
  Si no se encuentra configurada `GEMINI_API_KEY`, el script alertará en consola y aplicará el fallback determinístico de forma segura.

* **Consolidar revisiones manuales hacia la tabla maestra:**
  ```bash
  python scripts/clasificar_sites.py --aplicar-revision
  ```

> [!NOTE]
> `src/azure_utils.py` cuenta con validaciones preventivas (*guards*) que alertarán inmediatamente si las variables conservan los nombres de placeholder sin configurar.

---

## Arquitectura de Datos y Modelado

El pipeline transforma $33,775$ registros certificados a través de 3 componentes:

1. **Ingeniería de Características Relativas (`src/feature_engineering.py`):**
   * Normalización intra-función (`ratio_precio_max`, `percentil_precio_evento`, `peso_aforo`).
   * Validación de integridad física estricta ($\sum dn\_quota = performance\_quota$).
   * *Exclusión deliberada de `tasa_ocupacion` en el modelado:* Al ser una variable de absorción comercial *ex-post*, su exclusión garantiza una segmentación *ex-ante* pura basada en jerarquía física y precio.

2. **Procesamiento de Lenguaje Natural (`src/nlp_utils.py`):**
   * Eliminación de ruido publicitario y nombres de gira (*"EXPERIENCIA"*, *"TOUR"*, *"ETAPA 1"*).
   * Extracción de **17 tags binarios** en 4 dimensiones ortogonales:
     * **Jerarquía Comercial:** `tag_palco`, `tag_vip`, `tag_platea`, `tag_preferencial`, `tag_general`.
     * **Nivel Vertical:** `tag_balcon`, `tag_piso_alto`, `tag_piso_bajo`.
     * **Orientación Espacial:** `tag_occidental`, `tag_oriental`, `tag_norte`, `tag_sur`, `tag_lateral`, `tag_vista_parcial`.
     * **Restricciones de Acceso:** `tag_familiar`, `tag_menores`, `tag_movilidad_reducida`.

3. **Arquitectura en Dos Etapas y Espacio Mixto 36D (`src/clustering.py` - Modelo v2.5 Optimizado):**
   * **Etapa 1 (Determinística):** Aislamiento de funciones de admisión única / tarifa plana a nivel evento ($15,375$ registros, $45.5\%$ del catálogo: Cinemateca, Maloka, museos). Asignación directa a *Admisión Única / Tarifa Plana*.
   * **Etapa 2 (Machine Learning Multi-Zona):** Modelado en espacio mixto de 36 dimensiones sobre el catálogo zonificado ($18,400$ registros, $54.5\%$):
     * $4$ métricas numéricas relativas *ex-ante* (`RobustScaler`, incluyendo percentil de precio dentro del tipo de venue).
     * $7$ tags estructurales densos (5 comerciales + 2 verticales) en escala $[0, 1]$.
     * $10$ categorías canónicas one-hot de tipología de venue (`type_site`) ponderadas en $\omega_{\text{venue}} = 0.5$.
     * $15$ características TF-IDF reentrenadas exclusivamente sobre multi-zona con ponderación calibrada $\omega_{\text{nlp}} = 0.2$.
   * **Algoritmo & Etiquetado:** K-Means ($k=5$, óptimo formal por codo ortogonal y mínimo Davies-Bouldin) con correspondencia biyectiva de centroides geométricos 1-a-1 mediante el Algoritmo Húngaro en 36D.

---

## Taxonomía de Venues (type_site) y Migración v2

El modelo v2.5 consolida una taxonomía formal de **10 categorías canónicas de negocio** (más el fallback interno `desconocido` para casos no comerciales fuera del vector one-hot):

1. **`ARENA`**: Escenarios multipropósito modernos cubiertos de gran escala para conciertos internacionales (ej. Movistar Arena).
2. **`CINEMATECA`**: Salas de cine arte y centros audiovisuales con butacas numeradas individuales.
3. **`ESTADIO`**: Escenarios deportivos y de conciertos masivos al aire libre con graderías y cancha.
4. **`COLISEO`**: Escenarios polideportivos municipales tradicionales cerrados con graderías de hormigón.
5. **`MUSEO`**: Galerías, centros de exposiciones y sedes patrimoniales con aforo controlado.
6. **`OTROS_RECINTOS`**: Consolidación estratégica de auditorios, centros de convenciones, salas de concierto intermedias y cabaret/comedia.
7. **`PARQUE`**: Áreas verdes y espacios abiertos para festivales masivos al aire libre sin silletería fija.
8. **`PARQUEADERO`**: Establecimientos físicos cuya infraestructura es exclusivamente estacionamiento vehicular.
9. **`RESTAURANTE`**: Bares, restaurantes, clubes nocturnos y gastrobares con consumo de alimentos y bebidas.
10. **`TEATRO`**: Teatros tradicionales con distribución clásica en platea, palcos y balcones.

> [!NOTE]
> **Semántica Operativa de PARQUEADERO y Frecuencia Histórica:**  
> En TuBoleta, las boletas de parqueadero se venden como productos complementarios (*add-on*) asociados a eventos en estadios o arenas, heredando el tipo del venue anfitrión (`ESTADIO` o `ARENA`). Por diseño de negocio, ningún venue histórico opera autónomamente como parqueadero de espectáculos artísticos ($0$ localidades en el histórico). La categoría se preserva canónicamente para futuros desarrollos sin recurrir a buckets genéricos.

### Nota de Migración v1 a v2 y Compuerta de Frecuencias
* **Mapeo Explícito:** `arena_cubierta` $\rightarrow$ `ARENA`, `cine_sala_cultural` $\rightarrow$ `CINEMATECA`, `estadio_abierto` $\rightarrow$ `ESTADIO`, `coliseo` $\rightarrow$ `COLISEO`, `teatro` $\rightarrow$ `TEATRO`, `parque_aire_libre` $\rightarrow$ `PARQUE`, `bar_club` $\rightarrow$ `RESTAURANTE`, `{auditorio, centro_convenciones, sala_conciertos, cabaret_comedia}` $\rightarrow$ `OTROS_RECINTOS`, y re-clasificación curada del residual `otro` hacia `MUSEO`, `RESTAURANTE`, `PARQUEADERO` y `ARENA`.
* **Trazabilidad Garantizada:** El archivo `data/lookup/site_type_lookup.csv` mantiene la fuente original de cada asignación e incorpora `taxonomia_version="v2"` y `fecha_reclasificacion="2026-09-28"`.
* **Compuerta de Frecuencias:** Validada mediante `scripts/validar_frecuencias_taxonomy.py` con reporte generado en `reports/frecuencias_type_site_v2.csv`.

---

## Los 6 Arquetipos de Demanda (Modelo v2.5 Optimizado)

| Arquetipo Estandarizado | Etapa | Registros | % Catálogo | Ratio Precio | Peso Aforo | Precio Mediano COP | Localidades Típicas Clasificadas |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Admisión Única / Tarifa Plana** | Etapa 1 | 15,375 | **45.52%** | 0.99 | 100.0% | **$13,572** | *Cinemateca Bogotá, Maloka, YAWA, funciones monozona* |
| **Popular / Balcón / Visibilidad Parcial** | Etapa 2 | 6,128 | **18.14%** | 0.35 | 9.5% | **$45,000** | *Balcón 2do/3er Piso, Grada Alta Posterior, Visibilidad Parcial* |
| **VIP / Palcos / Premium** | Etapa 2 | 4,885 | **14.46%** | 0.81 | 5.5% | **$142,000** | *Palcos Corporativos, Suites, Mesas VIP, Boxes de lujo* |
| **Platea General / Intermedia** | Etapa 2 | 3,463 | **10.25%** | 0.71 | 32.8% | **$49,000** | *Platea Media, Balcón Delantero, Localidades intermedias* |
| **Preferencial / Platea Frontal** | Etapa 2 | 2,934 | **8.69%** | 0.86 | 17.0% | **$122,000** | *Platea 1, Platea Delantera, Sillas Centrales, Preferencial* |
| **Grada General / Masiva** | Etapa 2 | 990 | **2.93%** | 0.81 | 78.1% | **$66,000** | *Graderías masivas de estadios, Gradas Norte/Sur completas* |
| **TOTAL CATÁLOGO** | **v2.5** | **33,775** | **100.0%** | — | — | — | *Calidad y consistencia física 100% certificada* |

---

## Flujo de Ejecución

### Opción A: Exploración Interactiva en Jupyter
```bash
jupyter notebook notebooks/01_eda_clusterizacion.ipynb
jupyter notebook notebooks/02_clustering_espacio_mixto.ipynb
```

### Opción B: Pipeline en Script / Generación de Entregables
1. **Generar figuras analíticas para presentaciones:**
   ```bash
   python scripts/generate_all_presentation_figures.py
   ```
2. **Construir la presentación ejecutiva en PowerPoint:**
   ```bash
   python scripts/build_presentation_from_template.py
   ```

### Opción C: Ejecución de la Suite de Pruebas Automatizadas
```bash
python -m unittest discover tests/ -v
# o bien, con pytest instalado:
pytest tests/ -v
```

### Opción D: Inferencia en Producción con Observabilidad de Confianza
```python
import pandas as pd
from src.clustering import cargar_modelo_clustering, predecir_arquetipos_demanda

# Cargar modelo serializado v2.5
modelo = cargar_modelo_clustering("data/processed/modelo_clustering_v2_5.joblib")

# Predecir arquetipos con observabilidad completa (score de confianza, frontera, OOV)
df_segmentado = predecir_arquetipos_demanda(df_nuevas_localidades, modelo)

# Columnas de observabilidad devueltas:
# - arquetipo_demanda: Etiqueta de negocio asignada
# - score_confianza: Margen geometrico relativo [0, 1] (1.0 = certeza maxima)
# - es_frontera: True si score_confianza < 0.15 (requiere revision en pricing)
# - segundo_arquetipo: Arquetipo competidor mas cercano en frontera
# - cobertura_texto: Ratio de tokens en vocabulario TF-IDF [0, 1]
# - texto_casi_vacio: True si cobertura_texto < 0.20 (sin senal lexica)
# - probabilidad_gmm: Certeza posterior por mezcla de gaussianas
```

### Opción E: Auditoría Periódica de Drift Estadístico (PSI)
```bash
# Monitoreo por lote o ejecucion mensual en pipelines de datos
python scripts/monitorear_drift.py --datos data/raw/localidades_eda.parquet --output reports/drift_report.json
```

---

## Micro-clusters (exploración)

Evaluación experimental de particiones finas ($k > 10$) sobre las $18,400$ localidades multi-zona en el espacio vectorial 35D, con el propósito de normalización técnica de nombres de localidades en capas de backend (manteniendo siempre inalterado el nombre comercial `logical_seat_category`).

* **Decisión de Negocio:** El `cluster_id` es el identificador técnico de backend. Las etiquetas legibles (`label_auto`) se generan de forma automática según la combinación de tags dominantes ($\ge 60\%$) y términos TF-IDF, sin requerir compuertas de aprobación humana.
* **Script de Ejecución:**
  ```bash
  python scripts/explorar_microclusters.py
  ```
* **Artefactos Generados:**
  * Tabla comparativa de métricas: [`reports/microclusters_resultados.csv`](reports/microclusters_resultados.csv)
  * Catálogo de referencia con rollup a v2.3: [`data/processed/cluster_catalog.csv`](data/processed/cluster_catalog.csv)
* **Conclusión Cuantitativa:** Ningún candidato (K-Means $k=8..16$, GMM $k=8..16$, HDBSCAN grid) cumplió simultáneamente los criterios de aceptación (pureza $\ge 0.85$, bootstrap-ARI $\ge 0.85$ y cero clusters degenerados $<2\%$). En consecuencia, no se altera el pipeline de producción ni se persiste ningún modelo nuevo.

---

## Jerarquía de micro-clusters (v3.0-hier.2)

Arquitectura jerárquica en dos niveles desarrollada como evolución a la limitación de la exploración plana ($k > 10$ en el espacio unificado, donde la heterogeneidad global colapsa la pureza y separabilidad léxico-estructural).

* **Aprobación de la Jerarquía como Feature Generator:**
  La jerarquía queda **aprobada como generador de features** bajo sus criterios propios de calidad: alta estabilidad bootstrap-ARI en sub-espacios clave (hasta $0.9740$ en Popular y $0.9362$ en Platea Intermedia), cobertura total ($100\%$ del catálogo sin descarte de datos), rollup 1:1 estricto y 0 clusters degenerados. La compuerta de pureza léxica queda archivada como criterio propio del caso de uso de normalización de nombres, no del de generación de features predictivas (ver diagnóstico en Módulo 3.8). El modelo v2.5 de producción actúa como baseline de Nivel 1.
* **Arquitectura de Dos Niveles:**
  1. **Nivel 1 (Producción v2.5):** Separa localidades de tarifa plana (`Admisión Única`, $15,375$ registros clasificados directamente como micro-cluster terminal `AU-0`) y clasifica las $18,400$ localidades multi-zona en los 5 arquetipos macro de demanda certificados (espacio mixto 36D).
  2. **Nivel 2 (Sub-clustering por Arquetipo Macro):** Para cada uno de los 5 arquetipos macro multi-zona, se entrena un sub-modelo K-Means en un sub-espacio propio de **32 dimensiones**:
     * 4 numéricas relativas estandarizadas con `RobustScaler` ajustado localmente.
     * 13 tags estructurales binarios expandidos (`tag_palco`, `tag_vip`, `tag_platea`, `tag_preferencial`, `tag_general`, `tag_balcon`, `tag_piso_alto`, `tag_lateral`, `tag_occidental`, `tag_oriental`, `tag_norte`, `tag_sur`, `tag_mesa`).
     * 15 componentes TF-IDF calibrados sobre el vocabulario léxico propio del arquetipo ($\omega_{\text{nlp}} = 0.2$).
     * Se prescinde de la codificación one-hot de `type_site` para evitar ruido y sobrefragmentación dentro de un mismo arquetipo de demanda.
* **Selección de k y Calidad Estadística:**
  * Búsqueda en $k \in \{2, 3, 4, 5\}$ mediante optimización Codo-DB local.
  * Filtro de no degeneración: Descalificación de cualquier solución con clusters $< 3\%$ del sub-espacio.
  * Estabilidad bootstrap-ARI (20 réplicas al 80%) con promedio superior al $90\%$ en multi-zona clave.
* **Métricas Obtenidas por Sub-espacio (3.0-hier.2):**
  * **VIP / Palcos / Premium:** $k=3$, $N=5,100$, bootstrap-ARI $= 0.9342$, min share $= 15.75\%$.
  * **Popular / Balcón / Visibilidad Parcial:** $k=4$, $N=5,775$, bootstrap-ARI $= 0.9846$, min share $= 21.21\%$.
  * **Platea General / Intermedia:** $k=4$, $N=3,299$, bootstrap-ARI $= 0.9793$, min share $= 22.73\%$.
  * **Preferencial / Platea Frontal:** $k=4$, $N=3,243$, bootstrap-ARI $= 0.8803$, min share $= 12.03\%$ (supera todas las compuertas).
  * **Grada General / Masiva:** $k=3$, $N=983$, bootstrap-ARI $= 0.9611$, min share $= 30.93\%$.
  * **Total Micro-Clusters Global:** 19 particiones (1 de Admisión Única + 18 multi-zona).
* **Lineamientos de Negocio y Trazabilidad:**
  * Preservación irrestricta de `logical_seat_category` comercial.
  * `micro_cluster_id` como clave técnica de backend.
  * Generación determinística de `label_auto` (Title Case con tags activos $\ge 60\%$, deduplicación insensible a acentos y fallback a `hibrido_k{n}`).
  * Propagación de incertidumbre: Si `es_frontera=True` en Nivel 1, se activa `segmento_incierto=True` y se audita mediante `es_frontera_pct` en el catálogo.
  * Invarianza de Rollup Jerárquico: Todo micro-cluster pertenece a un único arquetipo macro (asociación 1:1 estricta validada por construcción).
* **Script de Ejecución:**
  ```bash
  python scripts/entrenar_jerarquia_microclusters.py
  ```
* **Artefactos Persistidos (data/processed/):**
  * `data/processed/modelo_jerarquia_v3.joblib`: Modelo jerárquico serializado con versión `3.0-hier.2`, sub-modelos y distribución de referencia.
  * `data/processed/cluster_catalog_v3.csv`: Catálogo de los 19 micro-clusters con pureza, términos dominantes y etiquetas.
  * `data/processed/asignacion_microclusters.csv`: Asignación individual para las 33,775 localidades.
* **Diagnóstico de Pureza y Separabilidad Oracle:** Evaluación formal de la compuerta de pureza ($\ge 0.85$) y sweep condicional en [`scripts/diagnosticar_subespacios.py`](scripts/diagnosticar_subespacios.py) con reporte estructurado en [`reports/diagnostico_subespacios.csv`](reports/diagnostico_subespacios.csv).

---

## Contrato de Features (v3.0-hier.2)

Especificación técnica para el consumo operativo de micro-clusters y arquetipos como features en modelos de demanda, propensión y pricing:

* **Columnas Entregadas:**
  * `micro_cluster_id`: Categoría técnica de 19 niveles (`AU-0` terminal para monozona + 18 particiones en 5 arquetipos: `VIP-0..2`, `POP-0..3`, `PGI-0..3`, `PPF-0..3`, `GGM-0..2`).
  * `arquetipo_demanda`: Segmento macro de 6 niveles (`Admisión Única / Tarifa Plana`, `VIP / Palcos / Premium`, `Popular / Balcón / Visibilidad Parcial`, `Platea General / Intermedia`, `Preferencial / Platea Frontal`, `Grada General / Masiva`).
  * Ambas son features categóricas derivadas de nombre, precio relativo, aforo y venue *ex-ante* (completamente seguras para modelos de demanda, sin fuga de información transaccional).
* **Encoding Recomendado:**
  * **One-Hot Encoding** para ambas variables (19 y 6 niveles, trivial y altamente eficiente para modelos basados en árboles como LightGBM, XGBoost o CatBoost).
  * Si el consumidor prefiere *target encoding* o *mean encoding*, debe realizarse obligatoriamente mediante validación cruzada *out-of-fold* (K-Fold CV) para prevenir fuga de datos (*target leakage*).
* **Freshness:**
  * La feature se asigna dinámicamente vía `src.jerarquia.predecir_microclusters(df_lote, payload_jerarquia)` al momento de scoring.
  * El payload se congela por versión (`3.0-hier.2`). Ante nuevos venues o datos faltantes, el predictor asigna flags seguros (`segmento_incierto`, `tipo_desconocido`) sin arrojar excepciones.
* **Monitoreo Continuo:**
  * Drift estadístico evaluado por lote mediante Population Stability Index (PSI) sobre la distribución observada de los 19 micro-clusters frente a la distribución de referencia persistida.
  * Umbrales: $\text{PSI} < 0.10$ (Estable), $0.10 \le \text{PSI} \le 0.25$ (Revisar), $\text{PSI} > 0.25$ (Drift Crítico). Si hay alerta activa, revisar la composición del lote antes de re-scoring masivo.
* **Snippet de Consumo (5 líneas):**
  ```python
  import pandas as pd
  from src.jerarquia import predecir_microclusters

  df_lote = pd.read_parquet("data/raw/localidades_eda.parquet")
  df_features = predecir_microclusters(df_lote, "data/processed/modelo_jerarquia_v3.joblib")
  # Features listas: df_features[["micro_cluster_id", "arquetipo_demanda"]]
  ```

---

## Marcha Blanca (Shadow Testing)

El proceso de **Marcha Blanca** permite auditar y evaluar el comportamiento de los modelos de clusterización (Nivel 1 Macro-Arquetipos v2.5 + Nivel 2 Micro-Clusters Canónicos v3.0) de forma aislada sobre nuevos eventos curados de Secutix, sin impacto sobre los sistemas en producción.

### 1. Origen de Datos y Parámetros
* **Ruta oficial en Azure Blob Storage (Capa Gold):**
  `GOLD/SECUTIX/Training Data/Clustering de Localidades test/`
* **Variables de entorno requeridas ([`.env`](.env)):**
  * `AZURE_STORAGE_CONNECTION_STRING` o `AZURE_STORAGE_ACCOUNT_NAME` + `AZURE_STORAGE_ACCOUNT_KEY`
  * `AZURE_CONTAINER_NAME=tuboleta`
* **Carga programática vía SDK:**
  ```python
  from src.azure_utils import cargar_parquet_desde_azure
  df_test_raw = cargar_parquet_desde_azure(blob_name="GOLD/SECUTIX/Training Data/Clustering de Localidades test/")
  ```

### 2. Ejecución de la Evaluación
Se puede ejecutar interactivamente mediante el notebook oficial o vía CLI:

* **Opción Notebook:** Abrir y ejecutar [`notebooks/03_marcha_blanca_evaluacion.ipynb`](notebooks/03_marcha_blanca_evaluacion.ipynb).
* **Opción CLI:**
  ```bash
  python scripts/ejecutar_evaluacion_marcha_blanca.py
  ```

### 3. Artefactos Producidos
* **Reporte Cuantitativo JSON:** [`reports/marcha_blanca_YYYYMMDD.json`](reports/) con metadatos de ejecución, los 4 checks de entrada, conteos por regla del filtro corregido y métricas de calidad (score de confianza, tasa de frontera, distribución observada vs. esperada).
* **Predicciones Enriquecidas:** [`data/processed/marcha_blanca_predicciones.csv`](data/processed/) y formato complementario `.parquet` con las asignaciones de `arquetipo_demanda` (Macro) y `micro_cluster_id` (19 micro-clusters canónicos) junto al nombre del evento (`product`).
* **Deuda Técnica Conocida:** Ver Sección 10.3 en [DOCUMENTACION_MODELO_CLUSTERIZACION.md](DOCUMENTACION_MODELO_CLUSTERIZACION.md#103-deuda-conocida-alineación-filtro-entrenamiento) sobre la alineación filtro-entrenamiento y tratamiento de precios en $0.

---

## Convenciones de Contribución y Commits

Para preservar la trazabilidad, reproducibilidad e higiene del repositorio:
* **Mensajes de Commit:** Los mensajes de commit deben redactarse en formato convencional (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
* **Higiene de Commits:** Commits de prueba (e.g. `marcha_blanca_test`) deben squasharse antes del push a `main`.
* **Notebooks:** Se debe aplicar obligatoriamente `nbstripout` sobre cualquier notebook antes de commitear para no versionar salidas o binarios pesados.

---

## Documentación Técnica Detallada
Para consultar la justificación matemática, fórmulas de normalización, descomposiciones de varianza PCA y pseudocódigo, consulta:
 **[DOCUMENTACION_MODELO_CLUSTERIZACION.md](DOCUMENTACION_MODELO_CLUSTERIZACION.md)**



