import os
import tarfile
import urllib.request as request

import pandas as pd

from .logger import get_logger

logger = get_logger(__name__)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url: str = HOUSING_URL, housing_path: str = HOUSING_PATH) -> None:
    logger.info(f"Fetching housing data from {housing_url}...")
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    request.urlretrieve(housing_url, tgz_path)
    logger.info(f"Extracting data to {housing_path}...")
    with tarfile.open(tgz_path, "r:gz") as f:
        f.extractall(path=housing_path)


def load_housing_data(housing_path: str = HOUSING_PATH) -> pd.DataFrame:
    csv_path = os.path.join(housing_path, "housing.csv")
    logger.info(f"Loading housing data from {csv_path}...")
    return pd.read_csv(csv_path)
