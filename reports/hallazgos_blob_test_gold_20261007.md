# Reporte Técnico de Auditoría: Hallazgos en Blob de Prueba Capa GOLD

**Destinatario:** Equipo de Data Engineering / Data Platform TuBoleta  
**Remitente:** Equipo de Data Science & Machine Learning  
**Fecha:** 2026-10-07  
**Blob Auditado:** `GOLD/SECUTIX/Training Data/Clustering de Localidades test/`  
**Referencia Cuantitativa:** [`reports/marcha_blanca_20261007.json`](marcha_blanca_20261007.json)  

---

## 1. Resumen Ejecutivo

El blob `GOLD/SECUTIX/Training Data/Clustering de Localidades test/` presenta dos anomalías estructurales severas (duplicación masiva del 89.2% de filas e incoherencia de aforos en el 99.97% de los eventos) que lo hacen inservible para la evaluación estadística de marcha blanca.  
El 99.78% de los registros resulta descartado al aplicar las reglas mínimas de consistencia física y monetaria exigidas por el modelo.  
Se requiere la regeneración completa del lote de prueba bajo criterios de completitud por evento y validación de cuotas antes de proceder con el despliegue a producción.

---

## 2. Evidencia Cuantitativa de Auditoría

Los resultados medidos de forma reproducible durante la ejecución del protocolo de marcha blanca arrojaron la siguiente distribución:

| Métrica / Regla de Validación | Registros | % del Total | Diagnóstico Técnico |
| :--- | :---: | :---: | :--- |
| **Volumetría Cruda Inicial (`filas_in`)** | **706,443** | **100.00%** | Partición total ingerida desde Azure Blob Storage. |
| **Check A: Duplicados exactos en crudo** | **636,577** | **89.20%** | Filas con valores idénticos en todas las columnas de la tabla. |
| **Check C: Eventos con suma(cuota) $\neq$ performance_quota** | **706,235** | **99.97%** | Registros pertenecientes a funciones cuyo aforo no cuadra físicamente. |
| **Regla 1 (Categoría lógica no vacía)** | $0$ | $0.00\%$ | Nombres de localidad válidos. |
| **Regla 2 (`dn_quota > 0`)** | $2,573$ | $0.36\%$ | Localidades con aforo menor o igual a cero. |
| **Regla 3 (`performance_quota > 0`)** | $2,618$ | $0.37\%$ | Eventos con aforo total inválido o en cero. |
| **Regla 4 (Precio estrictamente $> 0$)** | $58,093$ | $8.22\%$ | Filas con tarifa en $\$0$ COP o nula. |
| **Regla 5 (Cantidades $\ge 0$)** | $0$ | $0.00\%$ | Sin devoluciones negativas detectadas. |
| **Regla 6 (Deduplicación por clave de negocio)** | **630,475** | **89.25%** | Duplicados exactos por `(t_performance_id, site, logical_seat_category, dn_quota, precio)`. |
| **Regla 7 (Consistencia de aforo del evento)** | **11,134** | **1.58%** | Localidades de eventos donde la suma de localidades sobrevivientes no iguala `performance_quota`. |
| **Total Eliminadas por Filtro** | **704,893** | **99.78%** | Filas incompatibles con reglas de negocio. |
| **Localidades Válidas Finales (`filas_out`)** | **1,550** | **0.22%** | **Único remanente consistente disponible para inferencia (alerta: -95.41% vs. entrenamiento).** |

---

## 3. Hipótesis de Origen (Para Verificación del Equipo de Datos)

Las siguientes hipótesis se formulan como puntos de chequeo para orientar la depuración en las canalizaciones de datos:

1. **Concatenación no idempotente de ejecuciones:** El volumen de duplicados exactos ($89.2\%$) sugiere que el job de extracción de Secutix o la tarea de copia a la capa Gold se ejecutó múltiples veces volcando registros sobre el mismo prefijo sin cláusula `OVERWRITE` o deduplicación intermedia (`dropDuplicates`).
2. **Submuestreo o extracción fragmentaria por evento:** La discrepancia masiva en la consistencia de aforo ($99.97\%$ de registros en eventos rotos) apunta a que se extrajo una muestra de filas o se aplicó un filtro (por ejemplo, por rango de fechas de venta o canal de taquilla) que omitió localidades pertenecientes al evento, provocando que la suma de cuotas observadas no alcance el aforo total declarado (`performance_quota`).
3. **Población asíncrona de fuentes de cuotas:** Es posible que `performance_quota` se esté extrayendo de una tabla de cabecera de evento actualizada dinámicamente, mientras que `dn_quota` proviene de líneas históricas de configuración de contingente desalineadas en tiempo.

---

## 4. Peticiones Concretas para Data Engineering

Solicitamos coordinar la entrega de un nuevo archivo/prefijo en Azure Blob Storage considerando los siguientes requisitos:

1. **Garantizar Extracción Completa por Evento:** Toda función (`t_performance_id`) incluida en el extracto debe contener la totalidad de sus localidades físicas configuradas en taquilla, sin muestreo parcial.
2. **Deduplicación Nativa en Origen:** Implementar la deduplicación a nivel de pipeline de datos por la clave compuesta:
   ```sql
   PRIMARY KEY (t_performance_id, site, logical_seat_category, dn_quota, base_unit_amt_itx)
   ```
3. **Validación de Integridad de Aforo:** Incorporar una compuerta de calidad de datos (*Data Quality Gate*) previa a la publicación en la capa GOLD que audite:
   $$\sum \text{dn\_quota} == \text{performance\_quota} \quad \forall \; t\_performance\_id$$

---

## 5. Criterio de Aceptación del Nuevo Lote de Prueba

El nuevo blob será considerado apto para certificar la marcha blanca cuando supere sin advertencias los 4 checks de entrada en [`notebooks/03_marcha_blanca_evaluacion.ipynb`](../notebooks/03_marcha_blanca_evaluacion.ipynb):

* **Check A (Duplicados en Crudo):** Tasa de duplicidad $< 1.0\%$.
* **Check B (Conteos de Filtro):** Tasa de supervivencia de filas $> 90.0\%$.
* **Check C (Coherencia de Aforos):** $0$ eventos con diferencias en la suma de cuotas.
* **Check D (Volumetría):** Lote resultante con tamaño comparable al entorno productivo ($| \Delta | < 20\%$ frente al baseline de $33,775$ localidades).

---

## 6. Anexos y Referencias Verificables

* **Reporte JSON Machine-Readable:** [`reports/marcha_blanca_20261007.json`](marcha_blanca_20261007.json)
* **Notebook de Evaluación:** [`notebooks/03_marcha_blanca_evaluacion.ipynb`](../notebooks/03_marcha_blanca_evaluacion.ipynb)
* **Script CLI de Evaluación:** [`scripts/ejecutar_evaluacion_marcha_blanca.py`](../scripts/ejecutar_evaluacion_marcha_blanca.py)
