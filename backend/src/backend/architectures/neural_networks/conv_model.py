from typing import Tuple

import tensorflow as tf

from backend.architectures.neural_networks.keras_base import KerasBaseModel


class TrafficSignConvNN(KerasBaseModel):
    """
    Konwolucyjna sieć neuronowa opartna na modelu bazowym KerasBaseModel
    """

    def _create_model(
        self, input_shape: Tuple[int, int], num_classes: int
    ) -> tf.keras.Model:
        model = tf.keras.models.Sequential(name="TrafficSignConvNN")
        k_input_shape = (input_shape[0], input_shape[1], 3)

        model.add(tf.keras.layers.Input(shape=k_input_shape))

        # Blok 1
        model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same"))
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
