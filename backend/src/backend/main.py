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
    print(">>> TEST CALLBACKÓW START <<<")
    
    # 1. Pobieramy dane (zwraca DataFrame)
    # Wyłączamy balansowanie, żeby było szybciej
    pipeline = DataPipeline(balance_data=False, return_as_tuple=False)
    train_df, val_df, _ = pipeline.get_data()

    # 2. Bierzemy MIKRO próbkę (np. 50 zdjęć treningowych i 20 walidacyjnych)
    train_sample = train_df.sample(50)
    val_sample = val_df.sample(20)
    
    print(f"Przygotowanie danych: Train={len(train_sample)}, Val={len(val_sample)}")

    # 3. Ładujemy obrazki z dysku (używając Twojego nowego utils)
    X_train = load_images_from_paths(train_sample['Path'].values)
    y_train = train_sample['ClassId'].values
    
    X_val = load_images_from_paths(val_sample['Path'].values)
    y_val = val_sample['ClassId'].values

    # 4. Konfiguracja treningu "na niby"
    config = {
        "epochs": 2,          # Tylko 2 epoki! Tyle wystarczy, żeby odpalić callbacki.
        "batch_size": 8,      # Mały batch
        "patience": 5,        # Parametr dla EarlyStopping (nieistotny przy 2 epokach, ale wymagany przez config)
        "log_dir": "logs_test",           # Opcjonalnie: osobny katalog na testy
        "checkpoint_dir": "models/test"   # Opcjonalnie: osobny katalog na testy
    }

    # 5. Inicjalizacja i trening
    model = TrafficSignConvNN(create_model=True)
    
    print("Rozpoczynam próbny trening...")
    model.train(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        config=config
    )
    
    print(">>> TEST ZAKOŃCZONY <<<")
    print("Sprawdź teraz foldery 'logs_test' oraz 'models/test'!")