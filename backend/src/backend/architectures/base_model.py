from abc import ABC, abstractmethod
from typing import Dict, Any
from numpy import ndarray
from pandas import DataFrame
import numpy as np


class BaseModel(ABC):
    """
    Bazowa klasa dla modeli klasyfikujących znaki
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self.is_trained = False

    def train(
        self, train_data: DataFrame, val_data: DataFrame, config: Dict[str, Any]
    ) -> None:
        """
        Funkcja trenująca model wewnątrz

        :param train_data: Description
        :type train_data: DataFrame
        :param val_data: Description
        :type val_data: DataFrame
        :param config: Description
        :type config: Dict[str, Any]
        """
        pass

    @abstractmethod
    def predict_proba(self, image: ndarray) -> ndarray:
        """
        Zwraca macierz prawdopodobieństw o kształcie (N, num_classes).
        Dla pojedynczego zdjęcia N=1.
        """
        pass

    def predict(self, image: ndarray) -> ndarray:
        """
        Przyjmuje obraz (np. numpy array) i zwraca predykcję w formie numeru klasy.
        """
        probs = self.predict_proba(image)
        return np.argmax(probs, axis=1)

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Zapisuje model w odpowiednim dla niego formacie w podanym katalogu.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> None:
        """
        Metoda fabryczna (Factory Method).
        Tworzy instancję klasy, ładuje do niej wskazany model oraz zwracą obiekt tej klasy z załadowanymi wartościami.
        """
        pass
