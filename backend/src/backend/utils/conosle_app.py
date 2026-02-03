import os
import sys
import pandas as pd
import numpy as np
from backend.architectures.rf_model import TrafficSignRF
from backend.architectures.neural_networks.conv_model import TrafficSignConvNN
from backend.architectures.neural_networks.dense_model import TrafficSignDenseNN
from backend.data.pipeline import DataPipeline
from backend.utils.image_loader import load_images_from_paths
from backend.mappers.map_classes import get_classes_to_names


class ConsoleApp:
    def __init__(self):
        self.model = None
        self.pipeline = DataPipeline(balance_data=True, return_as_tuple=True)
        self.running = True

    def create_model_workflow(self):
        print("\n--- TWORZENIE NOWEGO MODELU ---")
        print("Dostępne modele:")
        print("1. Random Forest (RF)")
        print("2. Convolutional Neural Network (CNN)")
        print("3. Dense Neural Network (MLP)")

        model_choice = input("Wybierz model (1-3): ").strip()

        if model_choice == "1":
            n_estimators = int(input("Liczba estymatorów (domyślnie 100): ") or 100)
            max_depth_in = input("Max depth (Enter dla braku limitu): ").strip()
            max_depth = int(max_depth_in) if max_depth_in else None
            self.model = TrafficSignRF(n_estimators=n_estimators, max_depth=max_depth)
        elif model_choice == "2":
            self.model = TrafficSignConvNN(create_model=True)
        elif model_choice == "3":
            self.model = TrafficSignDenseNN(create_model=True)
        else:
            print("!! Błędny wybór.")
            return

        print(f">> Utworzono model: {type(self.model).__name__}")

    def train_model_workflow(self):
        print("\n=== TRENING MODELU ===")
        if self.model is None:
            print("!! Błąd: Najpierw stwórz model.")
            return

        print("1. Pobieranie danych...")
        try:
            (X_train_raw, y_train), (X_val_raw, y_val), _ = self.pipeline.get_data()  # type: ignore
        except Exception as e:
            print(f"!! Błąd pipeline: {e}")
            return

        train_paths = X_train_raw[:, -1] if X_train_raw.ndim > 1 else X_train_raw
        val_paths = X_val_raw[:, -1] if X_val_raw.ndim > 1 else X_val_raw

        print(f"2. Wczytywanie obrazów ({len(train_paths)} tr, {len(val_paths)} val)...")
        X_train = load_images_from_paths(train_paths, target_size=(32, 32))
        X_val = load_images_from_paths(val_paths, target_size=(32, 32))

        if X_train is None or X_val is None:
            print("!! Błąd wczytywania obrazów.")
            return

        X_train = X_train[..., ::-1].astype("float32") / 255.0
        X_val = X_val[..., ::-1].astype("float32") / 255.0

        try:
            if isinstance(self.model, TrafficSignRF):
                self.model.train((X_train, y_train), (X_val, y_val), config={})
            else:
                epochs = int(input("Liczba epok (5): ") or 5)
                batch_size = int(input("Batch size (32): ") or 32)
                self.model.train((X_train, y_train), (X_val, y_val), config={"epochs": epochs, "batch_size": batch_size})
            print(">> Trening zakończony.")
        except Exception as e:
            print(f"!! Błąd podczas treningu: {e}")

    def save_model_workflow(self):
        print("\n--- ZAPISYWANIE MODELU ---")
        if self.model is None:
            print("!! Brak modelu do zapisania.")
            return

        save_name = input("Nazwa pliku do zapisu (bez rozszerzenia, np. 'my_model'): ").strip()
        if not save_name:
            print("!! Nie podano nazwy pliku.")
            return

        if isinstance(self.model, TrafficSignRF):
            save_path = f"models/{save_name}.joblib"
        else:
            save_path = f"models/{save_name}.keras"

        try:
            print(f">> Zapisywanie modelu do {save_path}...")
            self.model.save(save_path)
            print(">> Model zapisany pomyślnie.")
        except Exception as e:
            print(f"!! Błąd zapisu: {e}")

    def load_workflow(self):
        print("\n--- ŁADOWANIE MODELU ---")
        path = input("Podaj ścieżkę do modelu (np. models/my_model.keras): ").strip()

        if not os.path.exists(path):
            print(f"!! Plik {path} nie istnieje.")
            return

        try:
            if path.endswith(".joblib"):
                print(">> Wykryto Random Forest")
                self.model = TrafficSignRF.load(path)
            elif path.endswith(".keras"):
                print(">> Wykryto sieć neuronową (.keras)")
                print("Jaki to typ sieci?")
                print("1. CNN (TrafficSignConvNN)")
                print("2. Dense (TrafficSignDenseNN)")
                net_type = input("Wybór (1/2): ").strip()

                if net_type == "2":
                    self.model = TrafficSignDenseNN.load(path)
                else:
                    self.model = TrafficSignConvNN.load(path)
            else:
                print("!! Nieznany format pliku.")
                return

            print(">> Model załadowany pomyślnie.")
        except Exception as e:
            print(f"!! Błąd ładowania: {e}")

    def predict_workflow(self):
        print("\n--- PREDYKCJA ---")
        if self.model is None:
            print("!! Najpierw załaduj lub wytrenuj model!")
            return

        img_path = input("Podaj ścieżkę do obrazka (jpg/png): ").strip()
        if not os.path.exists(img_path):
            print("!! Plik nie istnieje.")
            return

        image = load_images_from_paths(img_path, target_size=(32, 32))

        if image is None or image.size == 0:
            print("!! Błąd wczytywania obrazu.")
            return

        try:
            pred_idx, confidence = self.model.predict(image)

            if isinstance(pred_idx, np.ndarray):
                pred_idx = int(pred_idx[0]) if pred_idx.size > 0 else 0
            else:
                pred_idx = int(pred_idx)

            if isinstance(confidence, np.ndarray):
                confidence = float(confidence[0]) if confidence.size > 0 else 0.0
            else:
                confidence = float(confidence)

            names = get_classes_to_names()
            class_name = names.get(pred_idx, f"Class {pred_idx}")

            print(f"\n>>> WYNIK: {class_name} (ID: {pred_idx})")
            print(f">>> Pewność: {confidence:.2%}")
        except Exception as e:
            print(f"!! Błąd podczas predykcji: {e}")

    def show_menu(self):
        model_name = type(self.model).__name__ if self.model else "BRAK"

        print("========================================")
        print("   TRAFFIC SIGN DEV TOOL")
        print("========================================")
        print(f"Załadowany model: {model_name}")
        print("----------------------------------------")
        print("1. Stwórz nowy model")
        print("2. Trenuj aktualny model")
        print("3. Zapisz model do pliku")
        print("4. Załaduj model z pliku")
        print("5. Przetestuj model (Predykcja)")
        print("6. Wyczyść ekran")
        print("7. Wyjście")
        print("----------------------------------------")

    def run(self):
        while self.running:
            self.show_menu()
            choice = input("Wybierz opcję: ")

            match choice:
                case "1":
                    self.create_model_workflow()
                case "2":
                    self.train_model_workflow()
                case "3":
                    self.save_model_workflow()
                case "4":
                    self.load_workflow()
                case "5":
                    self.predict_workflow()
                case "6":
                    os.system("cls" if os.name == "nt" else "clear")
                case "7":
                    self.running = False
                case _:
                    pass
