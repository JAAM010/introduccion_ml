from pydantic import BaseModel


# 🔹 Modelo 1: Predicción de precios de vivienda
class ViviendaInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str


# 🔹 Modelo 2: Predicción de riesgo financiero
class RiesgoInput(BaseModel):
    edad: int
    ingresos_mensuales: float
    historial_credito: str
    num_productos_financieros: int
