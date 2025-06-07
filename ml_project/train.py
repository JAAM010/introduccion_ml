# nativas de python
import os
import joblib

# terceros
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# mis librerias
from src.data_loader import fetch_housing_data, load_housing_data
from src.preprocessing import build_preprocessor
from src.metrics import print_metrics
from src.logger import get_logger


logger = get_logger(__name__)


def main():
    logger.info("Iniciando el proceso de entrenamiento del modelo...")

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

    logger.info("Evaluando el modelo...")
    # Evaluar modelo
    y_pred = pipeline.predict(X_test)
    rmse, r2 = print_metrics(y_test, y_pred)
    logger.info("Evaluación completada.")

    logger.info("Guardando el modelo entrenado...")
    # Guardar pipeline completo
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/modelo_entrenado.pkl")


if __name__ == "__main__":
    main()
