# Clusterización de Localidades - TuBoleta

Proyecto integral de Data Science y Machine Learning para la segmentación y clasificación automatizada de localidades en espectáculos públicos a partir de datos transaccionales almacenados en formato `.parquet` en Azure Blob Storage.

El modelo implementa un **espacio vectorial mixto de 25 dimensiones** (características numéricas relativas *ex-ante*, tags estructurales de recinto extraídos mediante NLP y representaciones vectoriales TF-IDF) para agrupar el catálogo en **4 arquetipos estandarizados de demanda**.

---

## 📁 Estructura del Repositorio

```text
clusterizacion-localidades/
├── .env.example                                # Plantilla segura de variables de entorno (Azure)
├── .gitignore                                  # Exclusión de credenciales, datos y entornos
├── README.md                                   # Guía general de uso y arquitectura
├── requirements.txt                            # Dependencias y librerías de Python
├── DOCUMENTACION_MODELO_CLUSTERIZACION.md      # Especificación técnica y matemática exhaustiva
│
├── data/                                       # Datos locales (ignorado por Git)
│   ├── raw/                                    # Parquets descargados de Azure
│   └── processed/                              # Datasets con features y clusters asignados
│
├── notebooks/                                  # Flujo interactivo paso a paso
│   ├── 00_databricks_raw_data.ipynb            # Extracción y preparación inicial en Databricks
│   ├── 01_eda_clusterizacion.ipynb             # Análisis exploratorio, consistencia y 17 tags
│   └── 02_clustering_espacio_mixto.ipynb       # Espacio mixto (25D), K-Means/GMM y arquetipos
│
├── src/                                        # Módulos Python reutilizables
│   ├── __init__.py
│   ├── azure_utils.py                          # Conexión y descarga segura desde Azure Blob Storage
│   ├── nlp_utils.py                            # Limpieza de marketing y extracción de 17 tags NLP
│   ├── feature_engineering.py                  # Normalización relativa por evento y consistencia
│   └── clustering.py                           # Construcción de espacio mixto (25D) y K-Means
│
├── scripts/                                    # Automatización de entregables y visualizaciones
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

3. **Arquitectura en Dos Etapas y Espacio Mixto 25D (`src/clustering.py` - Modelo v2.1):**
   * **Etapa 1 (Determinística):** Aislamiento de funciones de admisión única / tarifa plana a nivel evento ($15,375$ registros, $45.5\%$ del catálogo: Cinemateca, Maloka, museos). Asignación directa a *Admisión Única / Tarifa Plana*.
   * **Etapa 2 (Machine Learning Multi-Zona):** Modelado en espacio mixto de 25 dimensiones sobre el catálogo zonificado ($18,400$ registros, $54.5\%$):
     * $3$ métricas numéricas relativas *ex-ante* (`RobustScaler`).
     * $7$ tags estructurales densos (5 comerciales + 2 verticales) en escala $[0, 1]$.
     * $15$ características TF-IDF reentrenadas exclusivamente sobre multi-zona ($\omega_{\text{nlp}} = 1.2$).
   * **Algoritmo & Etiquetado:** K-Means ($k=4$, `n_init=15`) con mapeo de centroides geométricos 1-a-1 en el espacio escalado (Algoritmo Húngaro).

---

## 🏷️ Los 5 Arquetipos de Demanda (Modelo v2.1)

| Arquetipo | Etapa | Registros | % Catálogo | Ratio Precio | Peso Aforo | Localidades Típicas |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Admisión Única / Tarifa Plana** | Etapa 1 | 15,375 | 45.5% | 1.00 | 100.0% | *Cinemateca, Maloka, YAWA, funciones monozona* |
| **VIP / Palcos / Premium** | Etapa 2 | 5,400 | 16.0% | 0.84 | 7.7% | *Palcos, Mesas VIP, Platino, Boxes, Suite* |
| **Preferencial / Platea Frontal** | Etapa 2 | 3,604 | 10.7% | 0.81 | 18.6% | *Platea 1, Platea 2, Preferencial Delantera* |
| **Popular / Visibilidad Parcial / Balcón** | Etapa 2 | 7,311 | 21.6% | 0.38 | 12.7% | *Platea Posterior, Balcón Mayor, Vista Parcial* |
| **Grada General / Masiva** | Etapa 2 | 2,085 | 6.2% | 0.79 | 59.2% | *Graderías masivas de estadios, Cancha General* |

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

---

## 📚 Documentación Técnica Detallada
Para consultar la justificación matemática, fórmulas de normalización, descomposiciones de varianza PCA y pseudocódigo, consulta:
👉 **[DOCUMENTACION_MODELO_CLUSTERIZACION.md](DOCUMENTACION_MODELO_CLUSTERIZACION.md)**
