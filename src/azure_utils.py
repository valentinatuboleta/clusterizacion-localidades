import os
import io
import pandas as pd
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

def get_blob_service_client() -> BlobServiceClient:
    """
    Inicializa y retorna el cliente de Azure Blob Storage desde la variable de entorno.
    """
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string or "TU_ACCOUNT_NAME" in connection_string:
        raise ValueError(
            "Por favor configura una cadena de conexión válida en tu archivo .env (AZURE_STORAGE_CONNECTION_STRING)."
        )
    return BlobServiceClient.from_connection_string(connection_string)

def cargar_parquet_desde_azure(
    container_name: str = None, 
    blob_name: str = None
) -> pd.DataFrame:
    """
    Descarga en memoria un archivo .parquet desde Azure Blob Storage y lo carga en un DataFrame de pandas.
    """
    container_name = container_name or os.getenv("AZURE_CONTAINER_NAME")
    blob_name = blob_name or os.getenv("AZURE_BLOB_NAME")

    if not container_name or not blob_name:
        raise ValueError("AZURE_CONTAINER_NAME y AZURE_BLOB_NAME deben estar definidos en el entorno o como argumentos.")

    blob_service_client = get_blob_service_client()
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    print(f"Descargando blob '{blob_name}' desde el contenedor '{container_name}'...")
    stream = blob_client.download_blob()
    data = stream.readall()

    df = pd.read_parquet(io.BytesIO(data))
    print(f"Descarga exitosa. Registros cargados: {len(df):,} filas, {len(df.columns)} columnas.")
    return df

if __name__ == "__main__":
    try:
        df = cargar_parquet_desde_azure()
        print(df.head())
    except Exception as e:
        print(f"Error cargando datos de Azure: {e}")
