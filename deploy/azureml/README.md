# Despliegue en Azure Machine Learning (Online Endpoints)

Este directorio contendrá los scripts y especificaciones YAML para el despliegue del modelo de clusterización como un **Managed Online Endpoint** en Azure Machine Learning.

---

## Directriz de Arquitectura para `score.py`

Cuando se construya el script de inferencia `score.py`, debe cumplir obligatoriamente con la siguiente directriz de gobernanza:

### 1. Ingesta de Vocabulario Desacoplado en `init()`
La función `init()` se ejecuta una sola vez al encender el contenedor de Azure ML. Debe asegurar la lectura de la variable de entorno `AZURE_STOPWORDS_URL`:

```python
import os
import joblib
from src.clustering import cargar_modelo_clustering, predecir_arquetipos_demanda
from src.nlp_utils import cargar_stopwords
import src.nlp_utils as nlp_utils

modelo_global = None

def init():
    global modelo_global
    
    # 1. Cargar artefacto del modelo serializado
    model_dir = os.getenv("AZUREML_MODEL_DIR", "data/processed")
    model_path = os.path.join(model_dir, "modelo_clustering_v2_5.joblib")
    modelo_global = cargar_modelo_clustering(model_path)
    
    # 2. Descargar en caliente el vocabulario gobernado de stopwords (Nivel 1: Blob SAS/URL)
    # Si AZURE_STOPWORDS_URL está configurada, nlp_utils descarga el CSV más reciente desde Blob.
    nlp_utils.STOPWORDS_MARKETING = cargar_stopwords()
```

### 2. Variables de Entorno en el Deployment YAML de Azure ML
En la especificación del despliegue en Azure ML (`deployment.yaml` o Azure ML Studio), configure la variable de entorno:

```yaml
environment_variables:
  AZURE_STOPWORDS_URL: "https://<storage-account>.blob.core.windows.net/<container>/stopwords_marketing.csv?<sas-token>"
```

### 3. Ventaja Operativa
* **Cero Rebuilds de Docker:** Para actualizar la lista de nombres de giras y patrocinios comerciales, simplemente ejecute `python scripts/proponer_stopwords.py --consolidar` (el cual sube el CSV oficial a Blob) y reinicie el servicio/endpoint en Azure ML.
* **Inercia Garantizada:** El modelo en producción (v2.5) permanece inerte y estable ante cambios en stopwords; el nuevo vocabulario servirá para blindar el próximo ciclo de re-entrenamiento (v2.6) contra distorsiones en TF-IDF.
