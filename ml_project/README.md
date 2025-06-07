# 🧠 ML Project - Pipeline de Regresión para Precios de Vivienda

Este proyecto implementa un flujo profesional en Python utilizando `scikit-learn` y `Pipeline` para entrenar un modelo de regresión lineal sobre el dataset de California Housing.

## 📁 Estructura del Proyecto

```
ml_project/
├── datasets/
│   └── housing/
├── models/
│   └── modelo_entrenado.pkl
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── metrics.py
├── train.py
├── README.md
└── requirements.txt
```

## 🚀 Uso

```bash
pip install -r requirements.txt
python train.py
```

El modelo entrenado se guarda automáticamente en la carpeta `models/`.

## 🔍 Evaluación

Se imprime en consola:

- RMSE (Root Mean Squared Error)
- R² (coeficiente de determinación)
