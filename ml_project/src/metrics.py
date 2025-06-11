from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from .logger import get_logger

logger = get_logger(__name__)


def print_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"R²: {r2:.2f}")
    return rmse, r2
