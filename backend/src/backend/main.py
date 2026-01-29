import tensorflow as tf
from backend.data.pipeline import DataPipeline
import os

def start_api() -> None:
    """
    Uruchamia serwer FastAPI
    """
    pass


def start_dev() -> None:
    '''
    Uruchamianie dla developerów backendu
    '''
    print(">>> Start: Pobieranie i przetwarzanie danych... <<<")

    # Inicjalizacja pipeline'u (z włączonym balansowaniem klas)
    pipeline = DataPipeline(balance_data=True, return_as_tuple=True)

    # Pobranie gotowych ramek danych
    train_df, val_df, test_df = pipeline.get_data()

    print(train_df)