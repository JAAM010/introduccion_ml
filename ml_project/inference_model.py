import requests

import pandas as pd

# URL del modelo servido por MLflow
endpoint = "http://localhost:8890/invocations"

# Crear el input como DataFrame
data = pd.DataFrame(
    [
        {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY",
        }
    ]
)

payload = {
    "dataframe_split": {
        "columns": [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "ocean_proximity",
        ],
        "data": [[-122.23, 37.88, 41, 880, 129, 322, 126, 8.3252, "NEAR BAY"]],
    }
}

# Convertir a formato MLflow (dataframe_split)
payload = {"dataframe_split": {"columns": data.columns.tolist(), "data": data.values.tolist()}}

# Enviar POST request
response = requests.post(endpoint, json=payload, headers={"Content-Type": "application/json"})

# Mostrar resultado
if response.status_code == 200:
    print("Predicción:")
    print(response.json())
else:
    print("Error en la predicción:")
    print(response.status_code, response.text)
