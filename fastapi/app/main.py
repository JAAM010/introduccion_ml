import os

from fastapi import FastAPI
import pandas as pd
import mlflow.pyfunc
from dotenv import load_dotenv

from app.schemas import InputData

# Cargar .env si lo necesitas (opcional)
load_dotenv()

# Configurar MLflow URI si aplica
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8888"))

# Cargar el modelo desde MLflow Registry
MODEL_URI = "models:/entrenamiento_vivienda/2"
model = mlflow.pyfunc.load_model(MODEL_URI)

# Crear la app
app = FastAPI(title="Servicio de Predicción de Vivienda")


@app.get("/")
def root():
    return {"message": "API de predicción activa"}


@app.post("/predict")
def predict(data: InputData):
    # Convertir a DataFrame
    df = pd.DataFrame([data.model_dump()])
    # Predecir
    pred = model.predict(df)
    return {"prediction": float(pred[0])}
