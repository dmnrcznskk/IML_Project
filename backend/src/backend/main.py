import tensorflow as tf
from backend.data.pipeline import DataPipeline
import os


def start_api() -> None:
    """
    Uruchamia serwer FastAPI
    """
    pass


import os
import shutil
import cv2
import numpy as np
import pandas as pd
from backend.architectures.neural_networks.conv_model import (
    TrafficSignConvNN,
)  # Upewnij się, że import pasuje
from backend.mappers.map_classes import get_classes_to_names
from backend.train_and_evaluate.evaluate.evaluator import ModelEvaluator
from backend.utils.image_loader import load_images_from_paths

def start_dev() -> None:
    """
    Uruchamianie dla developerów backendu
    """
    print(">>> Demo ewaluacji <<<")
    
    pipeline = DataPipeline(balance_data=False, return_as_tuple=False)
    _, _, test_df = pipeline.get_data()

    test_df = test_df.sample(50)
    
    print(f"Przetwarzanie {len(test_df)} zdjęć...")

    X_images = load_images_from_paths(test_df['Path'].values, target_size=(32, 32))
    y_test = test_df['ClassId'].values

    # Zabezpieczenie, gdyby nic się nie wczytało
    if len(X_images) == 0:
        print("Nie udało się wczytać żadnych zdjęć.")
        return

    model = TrafficSignConvNN(create_model=True) 
    
    names = get_classes_to_names()
    evaluator = ModelEvaluator(class_names=names)
    
    results = evaluator.evaluate(model, X_images, y_test, show_plot=True)
    
    print("Koniec demo.")