import os
import logging


def get_logger(name: str, log_file: str = "../logs/train.log") -> logging.Logger:
    # Crear carpeta de logs si no existe
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configuración del logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Evitar duplicados
    if not logger.handlers:
        # formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(message)s")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Handler para archivo
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
