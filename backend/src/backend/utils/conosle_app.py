import os
import sys
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

    def train_workflow(self):
        print("\n--- TRENING MODELU ---")
        print("Dostępne modele:")
        print("1. Random Forest (RF)")
        print("2. Convolutional Neural Network (CNN)")
        print("3. Dense Neural Network (MLP)")

        model_choice = input("Wybierz model (1-3): ").strip()

        if model_choice not in ["1", "2", "3"]:
            print("!! Błędny wybór.")
            return

        save_name = input(
            "Nazwa pliku do zapisu (bez rozszerzenia, np. 'my_model'): "
        ).strip()

        if not save_name:
            print("!! Nie podano nazwy pliku.")
            return

        print("\n1. Pobieranie danych...")
        train_df, val_df, test_df = self.pipeline.get_data()
        print(
            f"   Treningowe: {len(train_df)} | Walidacyjne: {len(val_df)} | Testowe: {len(test_df)}"
        )

        try:
            if model_choice == "1":
                # Random Forest
                n_estimators = int(input("Liczba estymatorów (domyślnie 100): ") or 100)
                max_depth_in = input("Max depth (Enter dla braku limitu): ").strip()
                max_depth = int(max_depth_in) if max_depth_in else None

                self.model = TrafficSignRF(
                    n_estimators=n_estimators, max_depth=max_depth
                )
                self.model.train(train_df, val_df, config={})  # type: ignore
                save_path = f"models/{save_name}.joblib"

            else:
                # Sieci neuronowe
                epochs = int(input("Podaj liczbę epok (domyślnie 5): ") or 5)
                batch_size = int(input("Podaj batch size (domyślnie 32): ") or 32)

                config = {
                    "epochs": epochs,
                    "batch_size": batch_size,
                }

                if model_choice == "2":
                    self.model = TrafficSignConvNN(create_model=True)
                elif model_choice == "3":
                    self.model = TrafficSignDenseNN(create_model=True)

                self.model.train(
                    train_data=train_df,  # type: ignore
                    val_data=val_df,  # type: ignore
                    config=config,
                )
                save_path = f"models/{save_name}.keras"

            print(f"4. Zapisywanie modelu do {save_path}...")
            self.model.save(save_path)

        except Exception as e:
            print(f"!! Błąd podczas treningu: {e}")
            return

        input("\nNaciśnij ENTER, aby wrócić do menu...")

    def load_workflow(self):
        print("\n--- ŁADOWANIE MODELU ---")
        path = input("Podaj ścieżkę do modelu (np. models/my_model.keras): ").strip()

        if not os.path.exists(path):
            print(f"!! Plik {path} nie istnieje.")
            input("ENTER...")
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

            names = get_classes_to_names()
            class_name = names[int(pred_idx)]

            print(f"\n>>> WYNIK: {class_name} (ID: {pred_idx})")
            print(f">>> Pewność: {confidence:.2%}")
        except Exception as e:
            print(f"!! Błąd podczas predykcji: {e}")

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def show_menu(self):
        model_name = type(self.model).__name__ if self.model else "BRAK"

        print("========================================")
        print("   TRAFFIC SIGN DEV TOOL")
        print("========================================")
        print(f"Załadowany model: {model_name}")
        print("----------------------------------------")
        print("1. Trenuj nowy model")
        print("2. Załaduj model z pliku")
        print("3. Przetestuj model (Predykcja)")
        print("4. Wyjście")
        print("5. Wyczyść ekran")
        print("----------------------------------------")

    def run(self):
        while self.running:
            self.show_menu()
            choice = input("Wybierz opcję: ")

            match choice:
                case "1":
                    self.train_workflow()
                case "2":
                    self.load_workflow()
                case "3":
                    self.predict_workflow()
                case "4":
                    self.running = False
                case "5":
                    self.clear_screen()
                case _:
                    pass
