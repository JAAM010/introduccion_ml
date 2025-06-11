import requests

import pandas as pd

# URL del modelo servido por MLflow
endpoint = "http://localhost:8890/invocations"


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

# Enviar POST request
response = requests.post(endpoint, json=payload, headers={"Content-Type": "application/json"})

# Mostrar resultado
if response.status_code == 200:
    print("Predicción:")
    print(response.json())
else:
    print("Error en la predicción:")
    print(response.status_code, response.text)
