import os
import joblib
from datetime import datetime

import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import optuna
from dotenv import load_dotenv
import tempfile
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import fetch_housing_data, load_housing_data
from src.preprocessing import build_preprocessor
from src.metrics import print_metrics
from src.logger import get_logger

logger = get_logger(__name__)

dotenv_path = os.path.join(os.path.dirname(__file__), "src", ".env")
load_dotenv(dotenv_path)


# Cargar y preparar datos una vez
fetch_housing_data()
df = load_housing_data()
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
preprocessor = build_preprocessor(X_train)


def create_plot(x_values, y_values, xlabel, ylabel, title):
    # Crear gráfico de RMSE vs número de trial
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o", linestyle="-", color="green")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    return plt


def objective(trial, parent_run_id):
    # Sugerir hiperparámetros
    alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
    solver = trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])

    pipeline = Pipeline([("preprocessing", preprocessor), ("regressor", Ridge(alpha=alpha, solver=solver))])

    with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True, parent_run_id=parent_run_id):

        logger.info(f"Trial {trial.number}: alpha={alpha}, solver={solver}")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse, r2 = print_metrics(y_test, y_pred)

        mlflow.log_params({"alpha": alpha, "solver": solver})
        mlflow.log_metrics({"rmse": rmse, "r2": r2})

        # Registrar información del modelo en Mlflow
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("dataset", "housing.csv")
        mlflow.log_param("split_ratio", 0.2)

        # Registrar el modelo con firma
        signature = infer_signature(X_train, y_pred)
        input_example = X_train.iloc[0:1]
        mlflow.sklearn.log_model(pipeline, "model_trial", signature=signature, input_example=input_example)

        return rmse  # Optuna minimiza esta métrica


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
    with mlflow.start_run(run_name=f"optuna_search_{fecha}") as parent_run:
        parent_run_id = parent_run.info.run_id

        study = optuna.create_study(
            direction="minimize", study_name=f"Ridge_Optimization_{fecha}", storage="postgresql://mlflow:mlflow@localhost:5432/optuna_db"
        )
        study.optimize(lambda trial: objective(trial, parent_run_id), n_trials=20, n_jobs=2)

        logger.info("Búsqueda completada.")
        logger.info(f"Mejores hiperparámetros: {study.best_params}")
        mlflow.log_params(study.best_params)

        # Reentrenar el mejor modelo y loguear artefactos en el parent_run
        best_alpha = study.best_params["alpha"]
        best_solver = study.best_params["solver"]
        best_trial = study.best_trial

        best_pipeline = Pipeline([("preprocessing", preprocessor), ("regressor", Ridge(alpha=best_alpha, solver=best_solver))])
        best_pipeline.fit(X_train, y_train)
        y_pred_best = best_pipeline.predict(X_test)
        rmse_best, r2_best = print_metrics(y_test, y_pred_best)

        signature = infer_signature(X_train, y_pred_best)
        input_example = X_train.iloc[0:1]

        mlflow.log_metric("rmse_best", rmse_best)
        mlflow.log_metric("r2_best", r2_best)
        mlflow.log_param("best_trial_number", best_trial.number)
        mlflow.sklearn.log_model(best_pipeline, "best_model", signature=signature, input_example=input_example)

        trials = study.trials_dataframe(attrs=("number", "value", "params"))

        for i, row in trials.iterrows():
            trial_number = int(row["number"])
            rmse_value = row["value"]
            mlflow.log_metric(f"trial_{trial_number}_rmse", rmse_value)

        # Guardar un resumen CSV de todos los trials y registrarlo como artefacto
        # Crear CSV con resumen de resultados
        trials_csv_path = os.path.join(tempfile.gettempdir(), "trials_summary.csv")
        trials.to_csv(trials_csv_path, index=False)

        # Registrar el CSV como artefacto del parent_run
        mlflow.log_artifact(trials_csv_path, artifact_path="summary")

        # Crear gráfico de RMSE vs número de trial
        fig = create_plot(trials["number"], trials["value"], "Número de Trial", "RMSE", "RMSE por número de trial")
        fig_path = os.path.join(tempfile.gettempdir(), "optuna_rmse_vs_trial.png")
        fig.savefig(fig_path)
        fig.close()

        mlflow.log_artifact(fig_path, artifact_path="summary")


if __name__ == "__main__":
    main()
