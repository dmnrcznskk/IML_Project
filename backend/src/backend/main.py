import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from backend.ml.data.fetchers.yahoo_fetcher import fetch_history
from backend.ml.data.mappers.map_ohlcv_to_features import map_ohlcv_to_features
from backend.ml.data.create_target import create_market_target

from backend.ml.architectures.random_forest_tree_class import (
    MusaRandomForestTreeClassifier,
)


def start_api() -> None:
    """
    Uruchamia serwer FastAPI
    """
    pass


def start_dev() -> None:
    '''
    Uruchamianie dla developerów backendu
    '''
   pass