# Clusterización de Localidades

Proyecto de Data Science y Machine Learning para la segmentación y clusterización de localidades a partir de datos estructurados almacenados en formato `.parquet` en Azure Blob Storage.

---

##  Estructura del Proyecto

```text
clusterizacion-localidades/
├── .env.example            # Plantilla de credenciales (Azure)
├── .gitignore              # Archivos ignorados por Git
├── README.md               # Descripción del proyecto
├── requirements.txt        # Dependencias de Python
├── data/                   # Datos locales (no subidos a Git)
│   ├── raw/
│   └── processed/
├── notebooks/              # Jupyter Notebooks de exploración y modelado
│   └── 01_eda_clusterizacion.ipynb
└── src/                    # Código fuente modular
    ├── __init__.py
    ├── azure_utils.py      # Conexión y descarga de Azure Blob Storage
    └── clustering.py       # Algoritmos de clusterización y métricas
```

---

##  Configuración Inicial

### 1. Crear y activar entorno virtual
```bash
python3 -m venv .venv
source .venv/bin/activate  # En Linux/macOS
# .venv\Scripts\activate   # En Windows
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar credenciales de Azure
Copia el archivo `.env.example` a un archivo `.env`:
```bash
cp .env.example .env
```
Edita `.env` con tus datos de Azure:
```env
AZURE_STORAGE_CONNECTION_STRING="tu_connection_string_real"
AZURE_CONTAINER_NAME="tu_contenedor"
AZURE_BLOB_NAME="tu_archivo.parquet"
```

---

##  Uso

Puedes ejecutar el notebook interactivo en Jupyter:
```bash
jupyter notebook notebooks/01_eda_clusterizacion.ipynb
```
O bien probar directamente la carga desde Python:
```bash
python src/azure_utils.py
```
