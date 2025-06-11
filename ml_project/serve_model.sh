#!/bin/bash


echo "Cargando las variables de entorno desde el archivo .env..."
export MLFLOW_TRACKING_URI=postgresql+psycopg2://mlflow:mlflow@localhost:5432/mlflow_db
export MLFLOW_ARTIFACT_URI=s3://mlflow
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=mlflow
export AWS_SECRET_ACCESS_KEY=mlflow_p@ssw0rd

echo "Ejecutando el comando de despliegue..."
mlflow models serve -m models:/entrenamiento_vivienda/1 -p 8890 --no-conda &

echo "Despliegue completado exitosamente."
