"""
Esquemas Pydantic para el API REST de clustering de localidades (Bloque 1).
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator


class LocalidadInput(BaseModel):
    logical_seat_category: str = Field(
        ...,
        min_length=1,
        description="Nombre o categoría de la localidad física"
    )
    dn_quota: float = Field(
        ...,
        gt=0,
        description="Aforo o capacidad asignada a la localidad (debe ser mayor a cero)"
    )
    med_base_unit_amt_itx: Optional[float] = Field(
        None,
        gt=0,
        description="Precio unitario mediano (debe ser mayor a cero si se proporciona)"
    )
    precio: Optional[float] = Field(
        None,
        gt=0,
        description="Alias opcional para med_base_unit_amt_itx"
    )
    net_sold_p_qty: Optional[float] = Field(
        0.0,
        ge=0,
        description="Boletas netas vendidas pagadas (>= 0)"
    )
    net_sold_c_qty: Optional[float] = Field(
        0.0,
        ge=0,
        description="Boletas netas vendidas de cortesía (>= 0)"
    )

    @model_validator(mode="after")
    def validar_precio_positivo(self) -> "LocalidadInput":
        p = self.med_base_unit_amt_itx if self.med_base_unit_amt_itx is not None else self.precio
        if p is None or p <= 0:
            raise ValueError(
                "Debe proporcionarse med_base_unit_amt_itx o precio con un valor estrictamente mayor a 0."
            )
        if self.med_base_unit_amt_itx is None:
            self.med_base_unit_amt_itx = self.precio
        return self


class EventoInput(BaseModel):
    t_performance_id: Optional[Union[int, str]] = Field(
        1,
        description="Identificador único del evento / función"
    )
    product: Optional[str] = Field(
        None,
        description="Nombre oficial del producto / evento"
    )
    event_name: Optional[str] = Field(
        None,
        description="Alias para product"
    )
    site: Optional[str] = Field(
        None,
        description="Nombre del recinto o venue"
    )
    venue_name: Optional[str] = Field(
        None,
        description="Alias para site"
    )
    performance_quota: Optional[float] = Field(
        None,
        gt=0,
        description="Aforo total de la función (si se omite, se calcula sumando el dn_quota de las localidades)"
    )
    localidades: List[LocalidadInput] = Field(
        ...,
        min_length=1,
        description="Lista de localidades a predecir (mínimo 1 localidad)"
    )


class PrediccionItemOutput(BaseModel):
    logical_seat_category: str
    arquetipo: str
    micro_cluster: str
    label_auto: str
    confianza: float
    es_frontera: bool
    segmento_incierto: bool
    arquetipo_demanda: Optional[str] = None
    micro_cluster_id: Optional[str] = None
    score_confianza: Optional[float] = None


class PrediccionOutput(BaseModel):
    modelo_version: str
    filas_in: int
    filas_out: int
    reporte_filtro: Dict[str, Any]
    predicciones: List[PrediccionItemOutput]


class HealthOutput(BaseModel):
    status: str
    modelo_clustering_version: str
    modelo_jerarquia_version: str
    modelo_version: str
