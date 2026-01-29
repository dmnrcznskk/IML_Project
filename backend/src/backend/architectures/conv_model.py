import os
import cv2
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple
from numpy import ndarray
from backend.architectures.base_model import BaseModel
import json


class TrafficSignConvNN(BaseModel):
    """
    Konwolucyjna sieć neuronowa (CNN).
    W tej wersji metoda 'train' oczekuje już przetworzonych danych (X, y),
    a nie surowego DataFrame.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int] = (32, 32),
        num_classes: int = 43,
        create_model: bool = True,
    ) -> None:
        """
        Args:
            input_shape (Tuple[int, int]): rozmiar zdjęcia wejściowego (H, W)
            num_classes (int): ilość klas
        """
        super().__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

        if create_model:
            self.model = self._create_model(input_shape, num_classes)
        else:
            self.model = None

    def _create_model(
        self, input_shape: Tuple[int, int], num_classes: int
    ) -> tf.keras.Sequential:
        """
        Wewnętrzna metoda do tworzenia architektury modelu.
        """
        model = tf.keras.models.Sequential(name="TrafficSignConvNN")
        k_input_shape = (input_shape[0], input_shape[1], 3)

        # Blok 1
        model.add(
            tf.keras.layers.Conv2D(
                32, (3, 3), padding="same", input_shape=k_input_shape
            )
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        # Blok 2
        model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        # Blok 3
        model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        # Głowica klasyfikująca
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(num_classes))
        model.add(tf.keras.layers.Activation("softmax"))

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def train(
        self,
        train_data: Tuple[ndarray, ndarray],
        val_data: Tuple[ndarray, ndarray],
        config: Dict[str, Any],
    ) -> None:
        """
        Trenuje model przy użyciu przygotowanych danych (macierzy NumPy).

        Args:
            train_data (Tuple[ndarray, ndarray]): Zbiór treningowy w postaci krotki (X_train, y_train).
            val_data (Tuple[ndarray, ndarray]): Zbiór walidacyjny w tym samym formacie co train_data.
            config (Dict[str, Any]): Słownik konfiguracji treningu (wymagane klucze np.: 'epochs', 'batch_size').
        """
        epochs = config.get("epochs", 10)
        batch_size = config.get("batch_size", 32)

        X_train, y_train = train_data
        X_val, y_val = val_data

        print(f"Rozpoczynam trening na {len(X_train)} próbkach...")

        self.model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
        )
        self.is_trained = True

    def predict_proba(self, image: ndarray) -> ndarray:
        """
        Zwraca prawdopodobieństwa dla pojedynczego obrazu.
        Pamiętaj: Image musi być w formacie BGR (jeśli wczytany przez cv2)
        """
        target_size = (self.input_shape[1], self.input_shape[0])

        img_resized = cv2.resize(image, target_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype("float32") / 255.0
        img_batch = np.expand_dims(img_norm, axis=0)

        return self.model.predict(img_batch, verbose=0)

    def save(self, path: str) -> None:
        if not path.endswith(".keras"):
            path += ".keras"

        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.model.save(path)

        metadata = {
            "is_trained": self.is_trained,
            "input_shape": self.input_shape,
        }

        json_path = path.replace(".keras", ".json")

        with open(json_path, "w") as f:
            json.dump(metadata, f)

        print(f"Model zapisany w: {path}")
        print(f"Metadane zapisane w: {json_path}")

    @classmethod
    def load(cls, path: str):
        if not path.endswith(".keras"):
            path += ".keras"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Nie znaleziono modelu: {path}")

        keras_model = tf.keras.models.load_model(path)
        json_path = path.replace(".keras", ".json")

        loaded_is_trained = False
        loaded_input_shape = (32, 32)

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                metadata = json.load(f)
                loaded_is_trained = metadata.get("is_trained", False)
                if "input_shape" in metadata:
                    loaded_input_shape = tuple(metadata["input_shape"])
        else:
            raise FileNotFoundError(f"Nie znaleziono pliku metadanych: {json_path}")

        instance = cls(input_shape=loaded_input_shape, create_model=False)
        instance.model = keras_model
        instance.is_trained = loaded_is_trained

        return instance
