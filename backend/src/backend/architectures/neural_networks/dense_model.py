import tensorflow as tf
from typing import Tuple
from backend.architectures.neural_networks.keras_base import KerasBaseModel


class TrafficSignDenseNN(KerasBaseModel):
    """
    Sieć neuronowa typu Dense oparta na modelu bazowym KerasBaseModel
    """

    def _create_model(
        self, input_shape: Tuple[int, int], num_classes: int
    ) -> tf.keras.Model:
        model = tf.keras.models.Sequential(name="TrafficSignDenseNN")
        k_input_shape = (input_shape[0], input_shape[1], 3)

        model.add(tf.keras.layers.Flatten(input_shape=k_input_shape))
        model.add(tf.keras.layers.Dense(512, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(256, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model
