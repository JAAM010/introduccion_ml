# Introducción a Python para Inteligencia Artificial y Machine Learning

Este repositorio contiene los materiales, ejemplos y ejercicios del curso **"Introducción a Python para IA y ML"**, diseñado para cubrir las mejores prácticas de desarrollo con Python en proyectos de analítica avanzada y el despliegue correcto de modelos en entornos productivos.

## 🧭 Objetivos del curso

1. Conocer los estándares de desarrollo para proyectos de ML/IA con Python.
2. Implementar pipelines de Machine Learning estructurados, reproducibles y mantenibles.

## 🗂️ Contenido por sesión

| Sesión | Tema | Enlace |
|--------|------|--------|
| 0 | Ciclo de vida de un proyecto ML | [notebooks/0_lifecycle.ipynb](notebooks/0_lifecycle.ipynb) |
| 1 | Fundamentos de desarrollo para ML con Python | [notebooks/1_fundamentos.ipynb](notebooks/1_fundamentos.ipynb) |
| 2 | Entornos virtuales en Python | [notebooks/2_entornos_virtuales.ipynb](notebooks/2_entornos_virtuales.ipynb) |
| 3 | Pipeline de entrenamiento con buenas prácticas | [notebooks/3_pipeline.ipynb](notebooks/3_pipeline.ipynb) |
| 4 | Gestión de experimentos con MLflow | [notebooks/4_mlflow.ipynb](notebooks/4_mlflow.ipynb) |


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
