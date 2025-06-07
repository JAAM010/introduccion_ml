import os
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

from src.data_loader import fetch_housing_data, load_housing_data
from src.preprocessing import build_preprocessor
from src.metrics import print_metrics
from src.logger import get_logger


logger = get_logger(__name__)

dotenv_path = os.path.join(os.path.dirname(__file__), "src", ".env")
load_dotenv(dotenv_path)


def main():
    logger.info("Iniciando el proceso de entrenamiento del modelo...")
    # URL del servidor de Mlflow
    mlflow.set_tracking_uri("http://localhost:8888")

    try:
        # Nombre del experimento en Mlflow
        mlflow.create_experiment("entrenamiento_vivienda", artifact_location="s3://mlflow")
        mlflow.set_experiment("entrenamiento_vivienda")
    except mlflow.exceptions.MlflowException as e:
        # Si el experimento ya existe, lo configuramos
        logger.warning(f"El experimento ya existe: {e}")
        mlflow.set_experiment("entrenamiento_vivienda")

    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"train_run_{fecha}", nested=True):
        logger.info("Iniciando el seguimiento del experimento en Mlflow...")

        logger.info("Cargando datos...")
        fetch_housing_data()
        df = load_housing_data()
        logger.info("Datos cargados correctamente.")

        logger.info("Preprocesando datos...")
        X = df.drop("median_house_value", axis=1)
        y = df["median_house_value"]

        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Crear pipeline completo
        preprocessor = build_preprocessor(X_train)

        pipeline = Pipeline([("preprocessing", preprocessor), ("regressor", LinearRegression())])

        logger.info("Entrenando el modelo...")
        # Entrenar pipeline
        pipeline.fit(X_train, y_train)
        logger.info("Modelo entrenado correctamente.")

        # Registrar información del modelo en Mlflow
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("dataset", "housing.csv")
        mlflow.log_param("split_ratio", 0.2)

        logger.info("Evaluando el modelo...")
        # Evaluar modelo
        y_pred = pipeline.predict(X_test)
        rmse, r2 = print_metrics(y_test, y_pred)

        # Registrar métricas en Mlflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        logger.info("Evaluación completada.")

        mlflow.sklearn.log_model(pipeline, "modelo_entrenado")

        logger.info("Guardando el modelo entrenado...")
        # Guardar pipeline completo
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/modelo_entrenado.pkl")


if __name__ == "__main__":
    main()
