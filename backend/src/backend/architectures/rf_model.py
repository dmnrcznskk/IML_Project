import json
import os
from typing import Dict, Any, Optional, Tuple

import cv2
import joblib
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier

from backend.architectures.base_model import BaseModel


class TrafficSignRF(BaseModel):
    """
    Model klasyfikacyjny oparty na lesie losowym (Random Forest).
    """

    def __init__(
        self,
        input_shape: Tuple[int, int] = (32, 32),
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
        )

    def _flatten_data(self, X: ndarray) -> ndarray:
        """Zamienia (N, H, W, 3) na (N, H*W*3)"""
        return X.reshape(X.shape[0], -1)

    def train(
        self,
        train_data: Tuple[ndarray, ndarray],
        val_data: Tuple[ndarray, ndarray],
        config: Dict[str, Any],
    ) -> None:
        """Funkcja trenujaca model

        Args:
            train_data (Tuple[ndarray, ndarray]): Zbiór treningowy w postaci krotki (X_train, y_train).
            val_data (Tuple[ndarray, ndarray]): Zbiór walidacyjny w tym samym formacie co train_data.
            ~~config~~ (Dict[str, Any]): Pozostałość po kalsie bazowej, nieużywana tutaj
        """
        X_train, y_train = train_data
        X_val, y_val = val_data

        X_train_flat = self._flatten_data(X_train)
        X_val_flat = self._flatten_data(X_val)

        print(f"Trenowanie Random Forest ({len(X_train)} próbek)...")

        self.model.fit(X_train_flat, y_train)
        self.is_trained = True
        val_score = self.model.score(X_val_flat, y_val)
        print(f"Trening zakończony. Accuracy na zbiorze walidacyjnym: {val_score:.4f}")

    def predict_proba(self, image: ndarray) -> ndarray:
        #TODO zrobienie osobnej funkcji do obróbki danych
        target_size = (self.input_shape[1], self.input_shape[0])
        img = cv2.resize(image, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = img.astype("float32") / 255.0

        img_flat = img_norm.reshape(1, -1)

        return self.model.predict_proba(img_flat)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_path = path if path.endswith(".joblib") else path + ".joblib"
        joblib.dump(self.model, model_path)

        metadata = {
            "is_trained": self.is_trained,
            "input_shape": self.input_shape,
        }
        with open(model_path.replace(".joblib", ".json"), "w") as f:
            json.dump(metadata, f)

        print(f"Model RF zapisany w: {model_path}")

    @classmethod
    def load(cls, path: str):
        model_path = path if path.endswith(".joblib") else path + ".joblib"
        json_path = model_path.replace(".joblib", ".json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Nie znaleziono modelu RF: {model_path}")

        loaded_model = joblib.load(model_path)

        with open(json_path, "r") as f:
            meta = json.load(f)

        instance = cls(input_shape=tuple(meta["input_shape"]))
        instance.model = loaded_model
        instance.is_trained = meta["is_trained"]

        return instance
