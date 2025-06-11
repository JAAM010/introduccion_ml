from dotenv import load_dotenv
from fastapi import FastAPI

from .router.riesgo import router as riesgo_router
from .router.vivienda import router as vivienda_router


app = FastAPI(title="ML API - Modelos especializados")

# Incluir las rutas desde routers.py
app.include_router(vivienda_router)
app.include_router(riesgo_router)


@app.get("/")
def root():
    return {"message": "API activa", "endpoints": ["/predict/vivienda", "/predict/riesgo"]}
