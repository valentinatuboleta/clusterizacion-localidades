# Archivo Histórico de Artefactos de Modelo

Este directorio contiene los payloads serializados históricos de versiones anteriores del modelo macro de clusterización:
* `modelo_clustering_v2_2.joblib`: Payload de la era v2.2 (25 dimensiones).
* `modelo_clustering_v2_3.joblib`: Payload de la era v2.3 (35 dimensiones, 9 categorías de venue v1).
* `modelo_clustering_v2_4.joblib`: Payload de la era v2.4 (36 dimensiones, 10 categorías de venue v2 preliminar).

**Producción Vigente:**
El único artefacto vigente de producción macro es [`data/processed/modelo_clustering_v2_5.joblib`](../../data/processed/modelo_clustering_v2_5.joblib) (36 dimensiones, 10 categorías canónicas, 136 validaciones humanas al 100%), complementado por el modelo jerárquico [`data/processed/modelo_jerarquia_v3.joblib`](../../data/processed/modelo_jerarquia_v3.joblib) (19 micro-clusters canónicos).
