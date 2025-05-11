"""
Ejemplo de integración con FastAPI para el endpoint de exportación.
"""
from fastapi import FastAPI, Request, HTTPException, Query
from typing import Optional

from exporters.facade.export_facade import ExportFacade

app = FastAPI()

@app.get("/export/{format}")
async def export_prediction(
    request: Request,
    format: str,
    price: float = Query(..., description="Precio predicho a exportar"),
    model: str = Query("LinearRegression", description="Modelo usado para la predicción"),
    currency: str = Query("EUR", description="Moneda del precio"),
    use_adapter: bool = Query(False, description="¿Usar patrón Adapter? (alternativa: Decorator)"),
    validate: bool = Query(True, description="¿Validar datos?"),
    log: bool = Query(True, description="¿Añadir logging?"),
    metadata: bool = Query(True, description="¿Añadir metadatos?")
):
    """
    Endpoint para exportar una predicción a diferentes formatos.

    Implementa un patrón Facade que encapsula los patrones Adapter y Decorator.
    """
    if not price:
        raise HTTPException(status_code=400, detail="No hay predicción para exportar")

    return ExportFacade.export_prediction(
        format=format,
        price=price,
        request=request,
        model=model,
        currency=currency,
        use_adapter_pattern=use_adapter,
        add_validation=validate,
        add_logging=log,
        add_metadata=metadata
    )

@app.get("/export/formats")
async def get_supported_formats():
    """
    Devuelve los formatos de exportación soportados.
    """
    return {
        "formats": ExportFacade.get_supported_formats()
    }
