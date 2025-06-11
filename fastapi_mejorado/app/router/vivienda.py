import os

from fastapi import APIRouter
import pandas as pd
import mlflow.pyfunc
from dotenv import load_dotenv

from app.schemas import ViviendaInput

# Cargar .env si lo necesitas (opcional)
load_dotenv()

# Configurar MLflow URI si aplica
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8888"))

router = APIRouter()


modelo_vivienda = mlflow.pyfunc.load_model("models:/entrenamiento_vivienda/2")


@router.post("/vivienda")
def predict_vivienda(data: ViviendaInput):
    df = pd.DataFrame([data.model_dump()])
    prediction = modelo_vivienda.predict(df)
    return {"model": "modelo_vivienda", "prediction": float(prediction[0])}
