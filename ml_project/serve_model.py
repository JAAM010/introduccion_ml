import os
import subprocess

from dotenv import load_dotenv

from src.logger import get_logger

logger = get_logger(__name__)

dotenv_path = os.path.join(os.path.dirname(__file__), "src", ".env")
load_dotenv(dotenv_path)


def main():
    logger.info("Iniciando el proceso de despliegue del modelo...")

    # Comando para ejecutar el script de despliegue
    command = ["mlflow", "models", "serve", "-m", "models:/entrenamiento_vivienda/1", "-p", "8890", "--no-conda"]

    try:
        logger.info("Ejecutando el comando de despliegue...")
        subprocess.run(command, check=True)
        logger.info("Despliegue completado exitosamente.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error al ejecutar el comando de despliegue: {e}")
        raise


if __name__ == "__main__":
    main()
