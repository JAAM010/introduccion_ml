# Introducción a Python para Inteligencia Artificial y Machine Learning

Este repositorio contiene los materiales, ejemplos y ejercicios del curso **"Introducción a Python para IA y ML"**, diseñado para cubrir las mejores prácticas de desarrollo con Python en proyectos de analítica avanzada y el despliegue correcto de modelos en entornos productivos.

## 🧭 Objetivos del curso

1. Conocer los estándares de desarrollo para proyectos de ML/IA con Python.
2. Implementar pipelines de Machine Learning estructurados, reproducibles y mantenibles.

## 🗂️ Contenido por sesión

| Sesión | Tema | Enlace |
|--------|------|--------|
| 0 | Ciclo de vida de un proyecto ML | [notebooks/sesion_0_lifecycle.ipynb](notebooks/sesion_0_lifecycle.ipynb) |
| 1 | Fundamentos de desarrollo para ML con Python | [notebooks/sesion_1_intro.ipynb](notebooks/sesion_1_intro.ipynb) |
| 2 | Pipeline de entrenamiento con buenas prácticas | [notebooks/sesion_2_pipeline.ipynb](notebooks/sesion_2_pipeline.ipynb) |
| 3 | Gestión de experimentos con MLflow | [notebooks/sesion_3_mlflow.ipynb](notebooks/sesion_3_mlflow.ipynb) |
| 4 | Introducción al despliegue en AWS SageMaker | [notebooks/sesion_4_sagemaker.ipynb](notebooks/sesion_4_sagemaker.ipynb) |
| 5 | Alternativas de despliegue y buenas prácticas de MLOps | [notebooks/sesion_5_mlops.ipynb](notebooks/sesion_5_mlops.ipynb) |

## 📁 Estructura del repositorio

```bash
.
├── notebooks/           # Notebooks por sesión
├── src/                 # Código fuente reutilizable
├── config/              # Configuraciones (YAML, .env, etc.)
├── tests/               # Tests automatizados
├── diagrams/            # Diagramas y visuales de apoyo
├── pyproject.toml       # Definición del entorno con Poetry
└── README.md
```

## 🧰 Requisitos

- Python 3.10+
- [Poetry](https://python-poetry.org/)

## 📦 Instalación rápida

```bash
git clone https://github.com/hrodriguezgi/introduccion_ml.git
cd introduccion_ml
poetry install
```
