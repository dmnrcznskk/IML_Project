from abc import ABC, abstractmethod
from typing import Dict, Any
from numpy import ndarray
import numpy as np
from typing import Tuple


class BaseModel(ABC):
    """
    Bazowa klasa dla modeli klasyfikujących znaki
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self.is_trained = False

    def train(
        self,
        train_data: Tuple[ndarray, ndarray],
        val_data: Tuple[ndarray, ndarray],
        config: Dict[str, Any],
    ) -> None:
        pass

    @abstractmethod
    def predict_proba(self, image: ndarray) -> ndarray:
        """
        Funkcja do przewidywania z prawdopodobieństwem

        Args:
            image (ndarray): załadowny obraz za pomocą cv2

        Returns:
            ndarray: macierz prawdopodobieństw o kształcie (N, num_classes) Dla pojedynczego zdjęcia N=1.

        """
        pass

    def predict(self, image: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Przyjmuje obraz (np. numpy array) i zwraca predykcję w formie numeru klasy oraz pewności wyboru tej klasy.
        """
        probs = self.predict_proba(image)

        predictions = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)

        return predictions, confidence

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Zapisuje model w podnaj ścieżce w formacie .keras
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> any:
        """
        Metoda fabryczna (Factory Method).
        Tworzy instancję klasy, ładuje do niej wskazany model oraz zwracą obiekt tej klasy z załadowanymi wartościami.
        """
        pass
