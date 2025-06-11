import os

from fastapi import APIRouter
import pandas as pd
import mlflow.pyfunc
from dotenv import load_dotenv

from app.schemas import RiesgoInput

# Cargar .env si lo necesitas (opcional)
load_dotenv()

# Configurar MLflow URI si aplica
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8888"))

router = APIRouter()

modelo_riesgo = mlflow.pyfunc.load_model("models:/Apples_Demand_first_attempt/1")


@router.post("/riesgo")
def predict_riesgo(data: RiesgoInput):
    df = pd.DataFrame([data.model_dump()])
    prediction = modelo_riesgo.predict(df)
    return {"model": "modelo_riesgo", "riesgo_probabilidad": float(prediction[0])}
