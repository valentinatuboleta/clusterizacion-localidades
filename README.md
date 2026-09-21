# Clusterización de Localidades - TuBoleta

Proyecto integral de Data Science y Machine Learning para la segmentación y clasificación automatizada de localidades en espectáculos públicos a partir de datos transaccionales almacenados en formato `.parquet` en Azure Blob Storage.

El modelo implementa un **espacio vectorial mixto de 25 dimensiones** (características numéricas relativas *ex-ante*, tags estructurales de recinto extraídos mediante NLP y representaciones vectoriales TF-IDF) para agrupar el catálogo en **6 arquetipos estandarizados de demanda** a través de una arquitectura en dos etapas.

---

## 📁 Estructura del Repositorio

```text
clusterizacion-localidades/
├── .env.example                                # Plantilla segura de variables de entorno (Azure)
├── .gitignore                                  # Exclusión de credenciales, datos y entornos
├── .pre-commit-config.yaml                     # Hook pre-commit con nbstripout
├── LICENSE                                     # Licencia MIT del proyecto
├── README.md                                   # Guía general de uso y arquitectura
├── requirements.txt                            # Dependencias locales (sin PySpark)
├── requirements-databricks.txt                 # Dependencias exclusivas para Databricks
├── DOCUMENTACION_MODELO_CLUSTERIZACION.md      # Especificación técnica y matemática exhaustiva (v2.2)
│
├── .github/                                    # Integración Continua (CI)
│   └── workflows/
│       └── ci.yml                              # Pipeline automatizado de GitHub Actions
│
├── data/                                       # Datos locales (ignorado por Git)
│   ├── raw/                                    # Parquets descargados de Azure
│   └── processed/                              # Datasets con features y clusters asignados
│
├── notebooks/                                  # Flujo interactivo paso a paso
│   ├── 00_databricks_raw_data.ipynb            # Extracción y preparación inicial en Databricks
│   ├── 01_eda_clusterizacion.ipynb             # Análisis exploratorio, consistencia y 17 tags
│   └── 02_clustering_espacio_mixto.ipynb       # Espacio mixto (25D), K-Means/GMM y arquetipos (v2.2)
│
├── src/                                        # Módulos Python reutilizables de producción
│   ├── __init__.py
│   ├── azure_utils.py                          # Conexión y descarga segura desde Azure Blob Storage
│   ├── nlp_utils.py                            # Limpieza de marketing y extracción de 17 tags NLP
│   ├── feature_engineering.py                  # Normalización relativa por evento y consistencia
│   └── clustering.py                           # Espacio mixto 25D, clustering bietápico, persistencia e inferencia
│
├── tests/                                      # Suite de pruebas automatizadas y aseguramiento de calidad
│   ├── __init__.py
│   └── test_clustering_golden_set.py           # Golden Set (20 casos), consistencia, persistencia y selector auto
│
├── scripts/                                    # Automatización, diagnóstico y análisis
│   ├── comparar_resultados_clustering.py       # Comparativa cuantitativa y matriz de transición v2.0 vs v2.2
│   ├── diagnostico_y_benchmark_avanzado.py     # Diagnóstico previo, sweep de pesos y benchmark de algoritmos
│   ├── optimizar_k_multizona.py                # Búsqueda formal de k óptimo (Codo Ortogonal + Davies-Bouldin)
│   ├── verificar_k5_perfiles.py                # Inspección de centroides y activación de tags
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

## ⚙️ Configuración Inicial

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
Copia la plantilla de ejemplo y completa con las credenciales de tu cuenta de Azure:
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
```

> [!NOTE]
> `src/azure_utils.py` cuenta con validaciones preventivas (*guards*) que alertarán inmediatamente si las variables conservan los nombres de placeholder sin configurar.

---

## 🧠 Arquitectura de Datos y Modelado

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

3. **Arquitectura en Dos Etapas y Espacio Mixto 25D (`src/clustering.py` - Modelo v2.1 Optimizado):**
   * **Etapa 1 (Determinística):** Aislamiento de funciones de admisión única / tarifa plana a nivel evento ($15,375$ registros, $45.5\%$ del catálogo: Cinemateca, Maloka, museos). Asignación directa a *Admisión Única / Tarifa Plana*.
   * **Etapa 2 (Machine Learning Multi-Zona):** Modelado en espacio mixto de 25 dimensiones sobre el catálogo zonificado ($18,400$ registros, $54.5\%$):
     * $3$ métricas numéricas relativas *ex-ante* (`RobustScaler`).
     * $7$ tags estructurales densos (5 comerciales + 2 verticales) en escala $[0, 1]$.
     * $15$ características TF-IDF reentrenadas exclusivamente sobre multi-zona con ponderación calibrada $\omega_{\text{nlp}} = 0.2$ (evitando la dilución dimensional del bloque continuo).
   * **Algoritmo & Etiquetado:** K-Means ($k=5$, óptimo formal por codo ortogonal y mínimo Davies-Bouldin de $1.2720$) con correspondencia biyectiva de centroides geométricos 1-a-1 mediante el Algoritmo Húngaro.

---

## 🏷️ Los 6 Arquetipos de Demanda (Modelo v2.2 Optimizado)

| Arquetipo Estandarizado | Etapa | Registros | % Catálogo | Ratio Precio | Peso Aforo | Precio Mediano COP | Localidades Típicas Clasificadas |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🎟️ **Admisión Única / Tarifa Plana** | Etapa 1 | 15,375 | **45.5%** | 0.99 | 100.0% | **$13,572** | *Cinemateca Bogotá, Maloka, YAWA, funciones monozona* |
| ⭐ **VIP / Palcos / Premium** | Etapa 2 | 2,912 | **8.6%** | 0.76 | 4.7% | **$135,000** | *Palcos Corporativos, Suites, Mesas VIP, Boxes de lujo* |
| 🎭 **Preferencial / Platea Frontal** | Etapa 2 | 4,697 | **13.9%** | 0.85 | 9.3% | **$94,340** | *Platea 1, Platea Delantera, Sillas Centrales, Preferencial* |
| 🪑 **Platea General / Intermedia** | Etapa 2 | 3,432 | **10.2%** | 0.80 | 37.2% | **$65,150** | *Platea Media, Balcón Delantero, Localidades intermedias* |
| 🏟️ **Grada General / Masiva** | Etapa 2 | 819 | **2.4%** | 0.79 | 81.4% | **$66,000** | *Graderías masivas de estadios, Gradas Norte/Sur completas* |
| 🎟️ **Popular / Balcón / Visibilidad Parcial** | Etapa 2 | 6,540 | **19.4%** | 0.35 | 11.3% | **$50,000** | *Balcón 2do/3er Piso, Grada Alta Posterior, Visibilidad Parcial* |
| **TOTAL CATÁLOGO** | **v2.2** | **33,775** | **100.0%** | — | — | — | *Calidad y consistencia física 100% certificada* |

---

## 🚀 Flujo de Ejecución

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

### Opción D: Inferencia en Producción con Modelo Persistido
```python
import pandas as pd
from src.clustering import cargar_modelo_clustering, predecir_arquetipos_demanda

# Cargar modelo serializado
modelo = cargar_modelo_clustering("data/processed/modelo_clustering_v2_2.joblib")

# Predecir arquetipos con separación bietápica garantizada (monozona vs multi-zona)
df_segmentado = predecir_arquetipos_demanda(df_nuevas_localidades, modelo)
```

---

## 📚 Documentación Técnica Detallada
Para consultar la justificación matemática, fórmulas de normalización, descomposiciones de varianza PCA y pseudocódigo, consulta:
👉 **[DOCUMENTACION_MODELO_CLUSTERIZACION.md](DOCUMENTACION_MODELO_CLUSTERIZACION.md)**
