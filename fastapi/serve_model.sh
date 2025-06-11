#!/bin/bash


echo "Ejecutando el comando de despliegue..."
uvicorn app.main:app --reload --port 8000 &

echo "Despliegue completado exitosamente."
