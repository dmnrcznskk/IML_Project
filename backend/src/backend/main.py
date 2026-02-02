import tensorflow as tf
from backend.data.pipeline import DataPipeline
import os
from backend.utils.conosle_app import ConsoleApp


def start_api() -> None:
    """
    Uruchamia serwer FastAPI
    """
    pass


def start_dev() -> None:
    """
    Uruchamianie dla developerów backendu
    """

    app = ConsoleApp()
    app.run()
