# Esqueleto de Presentación: Micro-Clusters y Arquetipos de Demanda
## Demostración de Doble Etiquetado en Datos Reales de Marcha Blanca

**Equipo:** BI & Data Science TuBoleta  
**Fecha:** 2026-10-07  
**Modelo Evaluado:** Macro Nivel 1 v2.5 (`modelo_clustering_v2_5.joblib`) + Micro-Clusters Jerárquicos v3.0 (`modelo_jerarquia_v3.joblib`)  
**Datos de Validación:** Lote Shadow Testing Azure Capa GOLD (`GOLD/SECUTIX/Training Data/Clustering de Localidades test/`, N=1,324)  
**Paleta Visual de Referencia:** Navy (`#0B2545`), Steel (`#134074`), Teal (`#009688`), Coral (`#E76F51`), Amber (`#F4A261`), Gris (`#6C757D`). Formato 16:9 (`template.pptx`).

---

### Slide 1: Portada y Contexto: Marcha Blanca de Scoring Productivo
* **Contexto de Operación:** Evaluación en entorno de *Shadow Testing* (Marcha Blanca) sin interferencia sobre sistemas transaccionales productivos.
* **Cero Re-entrenamiento:** Validación estricta de inferencia con modelos congelados (Macro v2.5 y Micro v3.0) ejecutando los envoltorios productivos del repositorio.
* **El Objetivo Central:** Demostrar que cada localidad nueva de TuBoleta recibe simultáneamente una **doble identidad**: técnica (micro-cluster y `label_auto`) y de negocio (arquetipo macro de demanda).
* **Referencia / Artefacto:** [`reports/marcha_blanca_20261007.json`](marcha_blanca_20261007.json) y [`notebooks/03_marcha_blanca_evaluacion.ipynb`](../notebooks/03_marcha_blanca_evaluacion.ipynb).
* **Notas de Orador:**  
  > *"Hoy no venimos a presentar un experimento de laboratorio, sino los resultados de la marcha blanca sobre datos transaccionales reales descargados de Azure. Ambos modelos corrieron congelados, sin re-entrenar una sola línea, demostrando cómo opera el pipeline en producción real."*

---

### Slide 2: El Modelo en Dos Niveles: De la Granularidad Técnica al Arquetipo de Negocio
* **Nivel 1 (Macro de Negocio):** 6 Arquetipos de Demanda que dictan la estrategia de pricing y asignación de contingentes (`Admisión Única`, `VIP/Palcos`, `Preferencial`, `Platea General`, `Popular/Balcón`, `Grada Masiva`).
* **Nivel 2 (Micro-Cluster Técnico):** Particiones especializadas dentro de cada arquetipo que capturan patrones léxicos y arquitectónicos finos mediante sub-espacios dedicados.
* **Invarianza de Rollup 1:1:** Cada micro-cluster pertenece estrictamente a un único arquetipo macro; no existen solapamientos ni ambigüedad jerárquica.
* **El Puente Semántico:** El campo `label_auto` traduce el clúster matemático a una descripción legible construida a partir de los atributos dominantes observados.
* **Referencia / Ilustración:** [`reports/figures/fig_arbol_arquetipos_microclusters.png`](figures/fig_arbol_arquetipos_microclusters.png).
* **Notas de Orador:**  
  > *"Fíjense en la estructura de árbol: el negocio habla en términos de 6 arquetipos para planear eventos, pero el motor de datos opera en micro-clusters. El puente entre la ingeniería y el negocio es `label_auto`, garantizando que todo agrupamiento técnico tenga un sentido comercial transparente."*

---

### Slide 3: El Catálogo Canónico de Micro-Clusters (v3.0)
* **Composición Formal:** Catálogo canónico de micro-clusters gobernado por métricas de pureza léxica (`purity_tag`, `purity_term`) y frecuencia.
* **Estructura por Macro-Arquetipo (18 Micro-Clusters Activos):**
  * `Admisión Única`: 1 nodo terminal determinístico (`AU-0`, 45.5% del catálogo histórico).
  * `Popular / Balcón`: 4 micro-clusters (`POP-0` a `POP-3`) separando balcones, pisos altos y visibilidad parcial.
  * `VIP / Palcos`: 3 micro-clusters (`VIP-0` a `VIP-2`) discriminando palcos puros frente a áreas corporativas híbridas.
  * `Platea General`: 4 micro-clusters (`PGI-0` a `PGI-3`) estratificando lunetas, plateas intermedias y sectores generales.
  * `Preferencial / Frontal`: 3 micro-clusters (`PPF-0` a `PPF-2`) con pureza de platea superior al 93%.
  * `Grada General`: 3 micro-clusters (`GGM-0` a `GGM-2`) consolidando tiquetes masivos y sectores de pie.
* **Descalificación por Calidad:** `VIP-3` y `PPF-3` fueron excluidos del catálogo activo al no alcanzar la compuerta de tamaño mínimo ($\ge 3\%$), protegiendo al modelo de sub-clusters residuales degenerados.
* **Referencia / Artefacto:** [`data/processed/cluster_catalog_v3.csv`](../data/processed/cluster_catalog_v3.csv).
* **Notas de Orador:**  
  > *"El catálogo v3.0 no fue producto de intuición: cada sub-espacio se optimizó independientemente con curvas Codo-Davies Bouldin. Excluimos deliberadamente dos grupos degenerados que tenían solo 3 filas en pruebas tempranas para mantener un catálogo industrial robusto con estabilidad bootstrap superior a 0.88 en todas las particiones clave."*

---

### Slide 4: Cómo se Asigna: La Mecánica de Inferencia en 3 Pasos
* **Paso 1: Filtro de Consistencia y Normalización Ex-Ante:** Aplicación de las 9 reglas de consistencia de datos, canonización de tipología de venue (`type_site`) y cálculo de percentiles relativos por evento.
* **Paso 2: Inferencia Nivel 1 (Macro v2.5):** Determinación de monozona vs. multi-zona; si es multi-zona, proyección en espacio mixto 36D y asignación biyectiva al arquetipo macro.
* **Paso 3: Especialización Nivel 2 (Micro v3.0):** El arquetipo macro invoca exclusivamente su sub-modelo K-Means especializado, calculando el `micro_cluster_id`, proyectando `label_auto` y auditando frontera.
* **Propagación Segura de Incertidumbre:** Si la localidad cae en zona de frontera en Nivel 1, el indicador `segmento_incierto=True` se propaga hacia el micro-cluster para alertar a los modelos downstream.
* **Referencia / Artefacto:** `src/jerarquia.py` (`predecir_microclusters`) y [`scripts/ejecutar_evaluacion_marcha_blanca.py`](../scripts/ejecutar_evaluacion_marcha_blanca.py).
* **Notas de Orador:**  
  > *"El scoring corre en milisegundos y en estricta cascada: primero garantizamos calidad física del dato, luego v2.5 asigna el arquetipo general, y finalmente el sub-modelo especializado desciende al micro-cluster. Si hay duda geométrica, el sistema no inventa certeza: prende la alarma de frontera."*

---

### Slide 5: Portada de la Galería y Densidad de Asignaciones
* **Las 6 Héroes Representativas:** Un ejemplo nítido y libre de frontera por cada uno de los 6 macro-arquetipos, demostrando la consistencia del etiquetado sobre datos reales de Azure GOLD.
* **Prueba de Densidad por Recinto:** Matriz de distribución de los **Top 10 recintos** por volumen sobre las **1,324 localidades limpias**, evidenciando cómo el modelo distribuye de forma coherente localidades en recintos monozona (Maloka, Planetario) y multizona (Movistar Arena, Teatro Jorge Eliécer Gaitán, Teatro Mayor).
* **Volumen Certificado:** 100% de las 1,324 localidades reciben doble etiquetado sin fallos ni valores nulos.
* **Referencias / Ilustraciones:** [`reports/figures/fig_tarjetas_ejemplos_marcha.png`](figures/fig_tarjetas_ejemplos_marcha.png) y [`reports/figures/fig_matriz_recintos_arquetipos.png`](figures/fig_matriz_recintos_arquetipos.png).
* **Notas de Orador:**  
  > *"Aquí abrimos la galería con las 6 tarjetas héroes: una por cada arquetipo macro. Y a la derecha, la matriz de recintos demuestra la densidad de la marcha blanca sobre las 1,324 filas: recintos culturales absorben tarifas planas, mientras arenas y teatros despliegan la estratificación completa de VIP, preferenciales y balcones."*

---

### Slide 6: Un Evento por Tamaño — Estrato Micro (≤ 500)
* **Evento Real:** `TRIBUTO A RAPHAEL YO ME LLAMO`
* **Recinto:** `BATUTA CALDAS - MANIZALES`
* **Aforo Total:** 240 personas (Estrato objetivo: `micro`, Estrato real: `micro`).
* **Localidades Evaluadas (5):**
  1. `PRIMER PISO` (\$81,818 COP) $\rightarrow$ Platea General / Intermedia (`PGI-0`, `hibrido_k0`, Conf: 30.7%).
  2. `CENTRO` (\$63,636 COP) $\rightarrow$ Platea General / Intermedia (`PGI-1`, `hibrido_k1`, Conf: 27.7%).
  3. `BALCÓN - GRADERÍA DERECHA` (\$45,454 COP) $\rightarrow$ Popular / Balcón (`POP-2`, `hibrido_k2`, Conf: 16.2%).
  4. `BALCÓN` (\$45,454 COP) $\rightarrow$ Popular / Balcón (`POP-2`, `hibrido_k2`, Conf: 14.9%, **FRONTERA**).
  5. `BALCÓN - GRADERÍA IZQUIERDA` (\$45,454 COP) $\rightarrow$ Popular / Balcón (`POP-2`, `hibrido_k2`, Conf: 16.2%).
* **Referencia / Ilustración:** [`reports/figures/fig_evento_micro.png`](figures/fig_evento_micro.png).
* **Notas de Orador:**  
  > *"Mismo modelo, cinco escaleras: de un teatro de 400 butacas a un estadio — la jerarquía interna se respeta en todos los tamaños. En este auditorio de 240 butacas en Manizales no se fuerzan palcos inexistentes: desciende limpiamente de Platea General a Balcón Popular, capturando con honestidad una frontera en el balcón central difuso."*

---

### Slide 7: Un Evento por Tamaño — Estrato Pequeño (500 – 2.000)
* **Evento Real:** `HANS ZIMMER VS JOHN WILLIAMS`
* **Recinto:** `TEATRO JORGE ELIECER GAITAN` (Bogotá)
* **Aforo Total:** 1,650 personas (Estrato objetivo: `pequeno`, Estrato real: `pequeno`).
* **Localidades Evaluadas (6):**
  1. `PLATEA CENTRO` (\$200,000 COP) $\rightarrow$ Preferencial / Platea Frontal (`PPF-2`, `Platea`, Conf: 55.6%).
  2. `PLATEA LATERAL` (\$176,000 COP) $\rightarrow$ Preferencial / Platea Frontal (`PPF-2`, `Platea`, Conf: 69.7%).
  3. `PLATEA POSTERIOR` (\$152,000 COP) $\rightarrow$ Preferencial / Platea Frontal (`PPF-0`, `Platea`, Conf: 45.1%).
  4. `PALCO` (\$124,000 COP) $\rightarrow$ VIP / Palcos / Premium (`VIP-2`, `hibrido_k2`, Conf: 44.0%).
  5. `BALCON DELANTERO` (\$100,000 COP) $\rightarrow$ Popular / Balcón (`POP-0`, `hibrido_k0`, Conf: 4.7%, **FRONTERA**).
  6. `BALCON POSTERIOR` (\$79,920 COP) $\rightarrow$ Popular / Balcón (`POP-0`, `hibrido_k0`, Conf: 38.5%).
* **Referencia / Ilustración:** [`reports/figures/fig_evento_pequeno.png`](figures/fig_evento_pequeno.png).
* **Notas de Orador:**  
  > *"Mismo modelo, cinco escaleras: de un teatro de 400 butacas a un estadio — la jerarquía interna se respeta en todos los tamaños. En el Teatro Gaitán, la platea frontal se estratifica en Centro ($200k), Lateral ($176k) y Posterior ($152k), mientras que el balcón delantero ($100k) se enciende como frontera honesta frente a la zona preferencial."*

---

### Slide 8: Un Evento por Tamaño — Estrato Mediano (2.000 – 8.000)
* **Evento Real:** `EL UNIPERSONAL DE LUCHO MELLERA`
* **Recinto:** `MOVISTAR ARENA` (Bogotá, Formato Mediano)
* **Aforo Total:** 6,435 personas (Estrato objetivo: `mediano`, Estrato real: `mediano`).
* **Localidades Evaluadas (7):**
  1. `TRIBUNA FAN SUR` (\$350,000 COP) $\rightarrow$ VIP / Palcos / Premium (`VIP-1`, `hibrido_k1`, Conf: 25.0%, **HÍBRIDO**).
  2. `PLATEA 101 - 103` (\$300,000 COP) $\rightarrow$ Preferencial / Platea Frontal (`PPF-2`, `Platea`, Conf: 52.9%).
  3. `PLATEA 104 - 106` (\$235,000 COP) $\rightarrow$ Preferencial / Platea Frontal (`PPF-0`, `Platea`, Conf: 25.7%).
  4. `SEGUNDO PISO 202 - 204` (\$180,000 COP) $\rightarrow$ Popular / Balcón (`POP-0`, `hibrido_k0`, Conf: 5.5%, **FRONTERA**).
  5. `SEGUNDO PISO 207 - 213` (\$180,000 COP) $\rightarrow$ Popular / Balcón (`POP-0`, `hibrido_k0`, Conf: 26.9%).
  6. `SEGUNDO PISO 205 - 206` (\$180,000 COP) $\rightarrow$ Popular / Balcón (`POP-0`, `hibrido_k0`, Conf: 31.3%).
  7. `SEGUNDO PISO 201 / 209 VISTA PARCIAL` (\$100,000 COP) $\rightarrow$ Popular / Balcón (`POP-1`, `hibrido_k1`, Conf: 48.8%).
* **Referencia / Ilustración:** [`reports/figures/fig_evento_mediano.png`](figures/fig_evento_mediano.png).
* **Notas de Orador:**  
  > *"Mismo modelo, cinco escaleras: de un teatro de 400 butacas a un estadio — la jerarquía interna se respeta en todos los tamaños. En un formato mediano de arena, Tribuna Fan lidera como VIP híbrido, las plateas ocupan el segmento preferencial, y el segundo piso se clasifica limpiamente en popular, aislando como frontera la fila de transición."*

---

### Slide 9: Un Evento por Tamaño — Estrato Grande (8.000 – 20.000)
* **Evento Real:** `CARLOS VIVES MEDELLÍN`
* **Recinto:** `DAVIARENA` (Medellín)
* **Aforo Total:** 12,979 personas (Estrato objetivo: `grande`, Estrato real: `grande`).
* **Localidades Evaluadas (12):**
  1. `TRIBUNA FAN 1` (\$699,000 COP) $\rightarrow$ VIP / Palcos (`VIP-1`, `hibrido_k1`, Conf: 21.1%).
  2. `PALCOS` (\$650,000 COP) $\rightarrow$ VIP / Palcos (`VIP-1`, `hibrido_k1`, Conf: 38.2%).
  3. `PLATEA DIAMANTE` (\$549,000 COP) $\rightarrow$ Preferencial (`PPF-2`, `Platea`, Conf: 19.1%).
  4. `PRIMAVERA LOUNGE` (\$449,000 COP) $\rightarrow$ VIP / Palcos (`VIP-2`, `hibrido_k2`, Conf: 44.1%).
  5. `PISO 3 302 - 303` (\$399,000 COP) $\rightarrow$ VIP / Palcos (`VIP-2`, `hibrido_k2`, Conf: 19.0%).
  6. `PLATEA VIP` (\$369,000 COP) $\rightarrow$ Preferencial (`PPF-0`, `Platea`, Conf: 13.6%, **FRONTERA**).
  7. `PLATEA GENERAL` (\$259,000 COP) $\rightarrow$ Popular (`POP-3`, `hibrido_k3`, Conf: 10.1%, **FRONTERA**).
  8. `PISO 3 305 - 311` (\$229,000 COP) $\rightarrow$ Popular (`POP-2`, `hibrido_k2`, Conf: 38.1%).
  9. `PISO 3 304 Y 312` (\$199,000 COP) $\rightarrow$ Popular (`POP-1`, `hibrido_k1`, Conf: 51.9%).
  10. `PISO 4 407 - 413` (\$157,000 COP) $\rightarrow$ Popular (`POP-0`, `hibrido_k0`, Conf: 47.4%).
  11. `PISO 4 404 - 406` (\$129,000 COP) $\rightarrow$ Popular (`POP-1`, `hibrido_k1`, Conf: 46.6%).
  12. `PISO 4 402 - 403` (\$99,000 COP) $\rightarrow$ Popular (`POP-1`, `hibrido_k1`, Conf: 44.2%).
* **Referencia / Ilustración:** [`reports/figures/fig_evento_grande.png`](figures/fig_evento_grande.png).
* **Notas de Orador:**  
  > *"Mismo modelo, cinco escaleras: de un teatro de 400 butacas a un estadio — la jerarquía interna se respeta en todos los tamaños. Con 13 mil personas y 12 localidades, la escalera completa convive sin colapsar: Palcos y Lounges en VIP, Platea Diamante en Preferencial, y la gradación de pisos 3 y 4 en Balcón Popular, con frontera en Platea General."*

---

### Slide 10: Un Evento por Tamaño — Estrato Estadio (> 20.000)
* **Evento Real:** `THE LAST DANCE - EL ULTIMO BAILE DEL REY DE AMÉRICA`
* **Recinto:** `ESTADIO METROPOLITANO` (Barranquilla)
* **Aforo Total:** 44,586 personas (Estrato objetivo: `estadio`, Estrato real: `estadio`).
* **Localidades Evaluadas (8):**
  1. `OCCIDENTAL ALTA` (\$178,500 COP) $\rightarrow$ VIP / Palcos / Premium (`VIP-0`, `hibrido_k0`, Conf: 4.7%, **FRONTERA**).
  2. `OCCIDENTAL BAJA` (\$161,500 COP) $\rightarrow$ VIP / Palcos / Premium (`VIP-0`, `hibrido_k0`, Conf: 32.3%).
  3. `ORIENTAL ALTA` (\$144,500 COP) $\rightarrow$ VIP / Palcos / Premium (`VIP-0`, `hibrido_k0`, Conf: 10.4%, **FRONTERA**).
  4. `ORIENTAL BAJA` (\$136,000 COP) $\rightarrow$ VIP / Palcos / Premium (`VIP-0`, `hibrido_k0`, Conf: 33.8%).
  5. `SUR ALTA` (\$102,000 COP) $\rightarrow$ Popular / Balcón (`POP-2`, `hibrido_k2`, Conf: 22.0%).
  6. `NORTE ALTA` (\$102,000 COP) $\rightarrow$ Popular / Balcón (`POP-2`, `hibrido_k2`, Conf: 22.0%).
  7. `SUR BAJA` (\$93,500 COP) $\rightarrow$ Popular / Balcón (`POP-2`, `hibrido_k2`, Conf: 35.5%).
  8. `NORTE BAJA` (\$93,500 COP) $\rightarrow$ Popular / Balcón (`POP-2`, `hibrido_k2`, Conf: 35.5%).
* **Referencia / Ilustración:** [`reports/figures/fig_evento_estadio.png`](figures/fig_evento_estadio.png).
* **Notas de Orador:**  
  > *"Mismo modelo, cinco escaleras: de un teatro de 400 butacas a un estadio — la jerarquía interna se respeta en todos los tamaños. En el Metropolitano de Barranquilla con casi 45 mil butacas, la topología deportiva pura se respeta: Occidental y Oriental son tratadas como VIP/Palcos con fronteras en las bandejas altas, y las tribunas Norte y Sur descienden a Popular."*

---

### Slide 11: Galería Segmento Premium (VIP y Preferencial Frontal)
* **Evidencia de Granularidad Interna (≤ 8 Tarjetas):** Desglose de localidades de alta gama asignadas a `VIP / Palcos / Premium` y `Preferencial / Platea Frontal`.
* **Mismo Arquetipo, Distintos Micro-Clusters:**
  * Dentro de VIP: discriminación entre palcos puros (`VIP-2`), palcos teatro/suites (`VIP-1`) y áreas corporativas mixtas (`VIP-0`).
  * Dentro de Preferencial: separación entre plateas frontales puras (`PPF-2`), plateas centrales (`PPF-1`) y plateas posteriores (`PPF-0`).
* **Barras de Confianza Calibradas:** Semáforo visual en cada tarjeta (Verde $\ge 70\%$, Ámbar $50-70\%$, Rojo $< 50\%$).
* **Referencia / Ilustración:** [`reports/figures/fig_galeria_premium.png`](figures/fig_galeria_premium.png).
* **Notas de Orador:**  
  > *"Mismo arquetipo, distintos micro-clusters: la granularidad interna es la feature de backend; el arquetipo es el lenguaje de negocio. Vean cómo dos localidades llamadas 'Platea' reciben micro-clusters diferentes según su cercanía al escenario y su nivel de tarificación."*

---

### Slide 12: Galería Segmento Masivos (Platea General, Popular / Balcón y Grada)
* **Evidencia en Sectores de Mayor Volumen (≤ 12 Tarjetas):** Agrupamiento de localidades masivas cubriendo `Platea General / Intermedia`, `Popular / Balcón / Visibilidad Parcial` y `Grada General / Masiva`.
* **Descomposición Fina de Micro-Clusters:**
  * En Popular/Balcón: discriminación entre balcones clásicos (`POP-0`), pisos altos (`POP-1`), pisos combinados (`POP-2`) y sectores periféricos (`POP-3`).
  * En Platea General: lunetas intermedias (`PGI-1`, `PGI-2`) frente a sectores generales de platea (`PGI-3`).
  * En Grada Masiva: tiquetes masivos (`GGM-0`) y gradas de pie (`GGM-1`, `GGM-2`).
* **Referencia / Ilustración:** [`reports/figures/fig_galeria_masivos.png`](figures/fig_galeria_masivos.png).
* **Notas de Orador:**  
  > *"Mismo arquetipo, distintos micro-clusters: la granularidad interna es la feature de backend; el arquetipo es el lenguaje de negocio. En los sectores masivos, donde se concentra la mayor venta de boletos, el sub-KMeans separa claramente un balcón de teatro de un piso 4 de arena."*

---

### Slide 13: Galería de Casos Especiales (Admisión Única, Fronteras Estadísticas e Híbridos)
* **Admisión Única (`AU-0`):** Cobertura determinística al 100% de confianza sobre eventos monozona y tarifas planas (`MEET & GREET`, `GALERÍA INESPERADA`).
* **Fronteras Estadísticas Honestas (`caso="frontera"`):**
  * Localidades en el límite difuso entre arquetipos (`PLATEA` en El Ensueño con 0.16% de confianza; `PLATEA POSTERIOR` con 0.30%).
  * El sistema las marca con borde punteado, asigna la mejor estimación y levanta la bandera `es_frontera=True` y `segmento_incierto=True`.
* **Híbridos Léxicos (`caso="hibrido"`):** Localidades donde los atributos textuales combinan múltiples conceptos (`label_auto`: `hibrido_k1`).
* **Referencia / Ilustración:** [`reports/figures/fig_galeria_especiales.png`](figures/fig_galeria_especiales.png).
* **Notas de Orador:**  
  > *"Aquí vemos cómo el sistema sabe lo que no sabe: las tarjetas con borde punteado son fronteras estadísticas honestas donde la confianza cae por debajo del 1% y el sistema activa la bandera de incertidumbre. Y en verde, la admisión única determinística resuelve el 48% del lote con certeza absoluta."*

---

### Slide 14: ¿Dónde Cayeron las 1,324? Frecuencia de Asignaciones por Micro-Cluster
* **Distribución Observada en Marcha Blanca (N=1,324):**
  * `AU-0` (Admisión Única): **647 localidades (48.87%)** — dominante debido a la alta presencia de eventos culturales monozona en el blob de prueba.
  * Sub-espacios de Platea General: `PGI-2` (82, 6.19%), `PGI-0` (71, 5.36%), `PGI-1` (52, 3.93%), `PGI-3` (49, 3.70%).
  * Sub-espacios de Preferencial: `PPF-1` (70, 5.29%), `PPF-2` (61, 4.61%), `PPF-0` (26, 1.96%).
  * Sub-espacios de Popular: `POP-2` (66, 4.98%), `POP-0` (28, 2.11%), `POP-1` (24, 1.81%), `POP-3` (3, 0.23%).
  * Sub-espacios VIP: `VIP-0` (38, 2.87%), `VIP-1` (33, 2.49%), `VIP-2` (33, 2.49%).
  * Sub-espacios de Grada: `GGM-1` (27, 2.04%), `GGM-0` (11, 0.83%), `GGM-2` (3, 0.23%).
* **Diagnóstico de Sesgo Muestral:** Los 18 micro-clusters activos cuentan con asignaciones en la marcha blanca; los conteos bajos en `POP-3` o `GGM-2` son reflejo del sesgo del subconjunto limpio de este blob específico, no una falla del modelo.
* **Referencia / Ilustración:** [`reports/figures/fig_frecuencia_microclusters.png`](figures/fig_frecuencia_microclusters.png).
* **Notas de Orador:**  
  > *"La muestra se concentra en Admisión Única (AU-0 con 48.9%) y Platea General (PGI-2 con 6.2%); los huecos en POP-3 o GGM-2 reflejan el sesgo del subset limpio del blob, no del modelo — con el blob regenerado esta curva se estabiliza."*

---

### Slide 15: Evidencia Completa y Anexo Verificable
* **Transparencia Total de los Datos:** Las **1,324 asignaciones completas de la marcha blanca** están consolidadas y disponibles para auditoría directa en formato Excel y Parquet.
* **Anexo Verificable Interactivo:**
  * Archivo: [`reports/anexo_asignaciones_marcha_blanca.xlsx`](anexo_asignaciones_marcha_blanca.xlsx) (75.3 KB).
  * 9 columnas normalizadas: `localidad`, `recinto`, `precio`, `type_site`, `micro_cluster_id`, `label_auto`, `arquetipo_demanda`, `score_confianza`, `es_frontera`.
  * Hoja única con **autofiltro activado**, **encabezados congelados** y confianza formateada en porcentaje.
* **Notas de Orador:**  
  > *"No nos quedamos en diapositivas resumen: las 1,324 asignaciones completas, filtrables y con sus 9 variables clave, quedan compiladas en el archivo Excel anexo para que cualquier miembro del equipo de negocio pueda filtrar por evento, recinto o cluster con un solo clic."*

---

### Slide 16: Métricas de Calidad y Confiabilidad en Marcha Blanca
* **Score de Confianza Geométrico Medio:** **0.6314 (63.14%)**, superando ampliamente el umbral operacional mínimo de 0.50.
* **Tasa de Localidades en Frontera:** **14.27%**, dentro del rango de diseño previsto ($< 20.0\%$), garantizando estabilidad en el 85.7% del inventario restante.
* **Tasa de Texto Vacío Post-Filtro:** **0.00%** (100% de localidades con categorización semántica completa).
* **Cobertura Transaccional:** **100%** de cobertura sobre las localidades consistentes procesadas.
* **Referencia / Artefacto:** [`reports/marcha_blanca_20261007.json`](marcha_blanca_20261007.json) (sección `metricas_evaluacion`).
* **Notas de Orador:**  
  > *"A nivel de calidad matemática, el modelo superó todas sus compuertas: confianza geométrica de 63%, apenas 14% de registros en frontera —muy por debajo del límite de alarma del 20%— y 100% de cobertura. Más aún: los 5 sub-espacios multi-zona tienen documentada una estabilidad bootstrap-ARI superior a 0.88 en réplicas al 80%, lo que garantiza reproducibilidad ante re-entrenamientos."*

---

### Slide 17: Límites Honestos del Lote y Gobernanza de Datos
* **Validación Funcional Exitosa, No Estadística ($N=1,324$):**
  * El pipeline procesó 706,443 registros brutos; tras aplicar las 9 reglas de consistencia física, sobrevivieron únicamente **1,324 localidades limpias (99.81% de reducción)**.
  * El lote demostró que el software, las APIs y los envoltorios funcionan de punta a punta, pero **su tamaño no permite certificar drift poblacional ni representatividad estadística** frente al baseline de 33,775 localidades.
* **Causas Raíz Identificadas en Capa GOLD:**
  * **90.11% de duplicados idénticos en crudo** (636,577 filas).
  * **99.97% de eventos con aforo roto** en crudo (muestreo parcial o extracción fragmentaria por fecha de venta).
  * Ausencia de pipeline completo de Databricks (18,420 filas con patrones técnicos sin depurar aguas arriba).
* **Criterios de Aceptación Acordados para el Nuevo Blob:** Tasa de duplicidad $< 1\%$, supervivencia $> 90\%$, 0 eventos con suma de cuota rota, y 0 patrones de producto no comerciales en crudo.
* **Referencia / Artefacto:** [`reports/hallazgos_blob_test_gold_20261007.md`](hallazgos_blob_test_gold_20261007.md).
* **Notas de Orador:**  
  > *"Debemos ser transparentes con el comité: 1,324 filas certifican que el código no se rompe y que las reglas filtran la basura, pero no constituyen una prueba estadística válida. El blob de prueba que recibimos tenía 90% de duplicados y funciones cortadas a la mitad. Ya entregamos el reporte formal de auditoría a Data Engineering con 5 criterios de aceptación para el nuevo lote."*

---

### Slide 18: Próximos Pasos y Hoja de Ruta
* **Paso Inmediato: Entrega de Nuevo Lote GOLD Completo:** Coordinación con Data Engineering para ingerir un blob extraído sin muestreo, con deduplicación por clave primaria en origen y compuerta $\sum \text{cuotas} == \text{aforo}$.
* **Evaluación Automatizada de Shadow Testing Final:** Ejecución del script `ejecutar_evaluacion_marcha_blanca.py` sobre el nuevo lote para auditar PSI de drift ($< 0.10$).
* **Resolución de Deuda Metodológica en v2.6:**
  * Las reglas 8 y 9 unificaron el serving con Spark.
  * Queda abierta la política comercial para localidades con tarifa en \$0 COP (cortesías/prensa) para alinear formalmente el re-entrenamiento histórico con el filtro estricto.
* **Publicación de Feature Store:** Exposición de `micro_cluster_id` y `arquetipo_demanda` como variables categóricas listas para modelos de propensión, pricing dinámico y elasticidad.
* **Referencia / Artefacto:** [`DOCUMENTACION_MODELO_CLUSTERIZACION.md`](../DOCUMENTACION_MODELO_CLUSTERIZACION.md) (Sección 10.3 y 11).
* **Notas de Orador:**  
  > *"La hoja de ruta es clara: una vez Data Engineering publique el blob curado con las 5 condiciones pactadas, ejecutaremos el protocolo final de drift. Con eso, y tras definir la política comercial para precios en cero en la versión 2.6, los micro-clusters quedarán listos para integrarse al pricing dinámico de la compañía."*
