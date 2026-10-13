"""
Script de Smoke Testing local para el microservicio de inferencia (Bloque 2).

Lee deploy/fixtures/ejemplo_evento.json, envia la solicitud POST al endpoint
local /v1/predecir-localidades, imprime las predicciones en una tabla formateada
y valida que la version del modelo contenga '2.5'.
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error


def smoke_test(
    endpoint_url: str = "http://localhost:8000/v1/predecir-localidades",
    fixture_path: str = "deploy/fixtures/ejemplo_evento.json",
    timeout: float = 10.0
) -> int:
    print("=" * 80)
    print("SMOKE TEST LOCAL - SERVICIO FASTAPI CLUSTERING TUBOLETA")
    print("=" * 80)
    print(f" • Fixture objetivo : {fixture_path}")
    print(f" • Endpoint URL     : {endpoint_url}")

    if not os.path.exists(fixture_path):
        print(f"\n[ERROR] No se encontro el archivo fixture en: {fixture_path}")
        return 1

    with open(fixture_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    evento_nombre = payload.get("product") or payload.get("event_name", "N/A")
    venue_nombre = payload.get("site") or payload.get("venue_name", "N/A")
    n_localidades = len(payload.get("localidades", []))
    print(f" • Evento cargado   : {evento_nombre} ({venue_nombre})")
    print(f" • Localidades in   : {n_localidades}")

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint_url,
        data=req_data,
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = response.getcode()
            response_body = response.read().decode("utf-8")
            data = json.loads(response_body)
    except urllib.error.HTTPError as http_err:
        err_msg = http_err.read().decode("utf-8")
        print(f"\n[ERROR HTTP {http_err.code}] {err_msg}")
        return 1
    except urllib.error.URLError as url_err:
        print(f"\n[ERROR CONEXION] No se pudo conectar a {endpoint_url}: {url_err.reason}")
        print(" Asegurate de haber iniciado el servicio con: uvicorn src.api.main:app --reload")
        return 1

    print(f"\n[RESPUESTA HTTP {status_code}] OK")
    modelo_ver = data.get("modelo_version", "")
    print(f" • Version del modelo: {modelo_ver}")
    print(f" • Filas in / out     : {data.get('filas_in')} in / {data.get('filas_out')} out")

    # Assert estricto requerido
    assert "2.5" in modelo_ver, (
        f"Fallo de asercion: '2.5' no encontrado en modelo_version ('{modelo_ver}')"
    )
    print(" [ASSERT PASSED] modelo_version contiene '2.5'")

    predicciones = data.get("predicciones", [])
    print("\n" + "-" * 105)
    print(f"{'LOCALIDAD':<25} | {'ARQUETIPO':<32} | {'MICRO':<6} | {'LABEL AUTO':<16} | {'CONF':<6} | {'FRONTERA'}")
    print("-" * 105)

    for p in predicciones:
        loc = p.get("logical_seat_category", "")[:24]
        arq = p.get("arquetipo", "")[:31]
        mc = p.get("micro_cluster", "")
        lbl = p.get("label_auto", "")[:15]
        conf = f"{p.get('confianza', 0.0):.3f}"
        front = "SI" if p.get("es_frontera") else "NO"
        print(f"{loc:<25} | {arq:<32} | {mc:<6} | {lbl:<16} | {conf:<6} | {front}")

    print("-" * 105)
    print(f"\n Smoke test completado exitosamente: {len(predicciones)} localidades predichas correctamente.\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test local del servicio FastAPI")
    parser.add_argument("--url", default="http://localhost:8000/v1/predecir-localidades", help="URL del endpoint")
    parser.add_argument("--fixture", default="deploy/fixtures/ejemplo_evento.json", help="Ruta al JSON fixture")
    args = parser.parse_args()

    sys.exit(smoke_test(endpoint_url=args.url, fixture_path=args.fixture))
