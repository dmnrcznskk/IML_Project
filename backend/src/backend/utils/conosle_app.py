import os

import numpy as np

from backend.architectures.neural_networks.conv_model import TrafficSignConvNN
from backend.architectures.neural_networks.dense_model import TrafficSignDenseNN
from backend.architectures.rf_model import TrafficSignRF
from backend.data.pipeline import DataPipeline
from backend.evaluate.evaluator import ModelEvaluator
from backend.mappers.map_classes import get_classes_to_names
from backend.utils.image_loader import load_images_from_paths


class ConsoleApp:
    def __init__(self):
        self.model = None
        self.pipeline = DataPipeline(balance_data=True, return_as_tuple=True)
        self.running = True
        
        print(">> Inicjalizacja pipeline i pobieranie danych (może to chwilę potrwać)...")
        try:
            self.train_data_raw, self.val_data_raw, self.test_data_raw = self.pipeline.get_data() # type: ignore
            print(">> Dane załadowane do pamięci cache.")
        except Exception as e:
            print(f"!! Błąd podczas wstępnego pobierania danych: {e}")
            self.train_data_raw = self.val_data_raw = self.test_data_raw = None

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

        if self.train_data_raw is None or self.val_data_raw is None:
            print("!! Błąd: Brak danych treningowych w cache.")
            return

        print("1. Korzystanie z danych z cache...")
        X_train_raw, y_train = self.train_data_raw
        X_val_raw, y_val = self.val_data_raw

        #TODO zrobienie osobnej funkcji do obróbki danych
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

            show_all = input("\nCzy wyświetlić rozpiskę wszystkich klas? (t/n): ").lower() == 't'
            if show_all:
                threshold = float(input("Pokaż klasy powyżej (np. 0.01 dla 1%): ") or 0.01)
                probs = self.model.predict_proba(image)
                if isinstance(probs, np.ndarray):
                    probs = probs.flatten()
                    
                    print(f"\nKlasy powyżej {threshold:.2%}:")
                    found = False
                    sorted_indices = np.argsort(probs)[::-1]
                    for idx in sorted_indices:
                        p = probs[idx]
                        if p >= threshold:
                            c_name = names.get(idx, f"Class {idx}")
                            print(f"- {c_name} (ID: {idx}): {p:.2%}")
                            found = True
                    if not found:
                        print("- Brak klas spełniających kryterium.")

        except Exception as e:
            print(f"!! Błąd podczas predykcji: {e}")

    def evaluation_workflow(self):
        print("\n=== EWALUACJA MODELU (TEST SET) ===")
        if self.model is None:
            print("!! Brak modelu do ewaluacji.")
            return

        if self.test_data_raw is None:
            print("!! Błąd: Brak danych testowych w cache.")
            return

        print("1. Korzystanie z danych testowych z cache...")
        X_test_raw, y_test = self.test_data_raw
        if X_test_raw.size == 0:
            print("!! Brak danych testowych.")
            return

        test_paths = X_test_raw[:, -1] if X_test_raw.ndim > 1 else X_test_raw
        print(f"2. Wczytywanie obrazów ({len(test_paths)})...")
        X_test = load_images_from_paths(test_paths, target_size=(32, 32))

        if X_test is None:
            print("!! Błąd wczytywania obrazów.")
            return

        try:
            names = get_classes_to_names()
            evaluator = ModelEvaluator(class_names=names)
            
            print("3. Uruchamianie ewaluacji...")
            evaluator.evaluate(self.model, X_test, y_test, show_plot=False)
            
        except Exception as e:
            print(f"!! Błąd podczas ewaluacji: {e}")

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
        print("6. Ewaluacja modelu (Zbiór testowy)")
        print("7. Wyczyść ekran")
        print("8. Wyjście")
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
                    self.evaluation_workflow()
                case "7":
                    os.system("cls" if os.name == "nt" else "clear")
                case "8":
                    self.running = False
                case _:
                    pass
