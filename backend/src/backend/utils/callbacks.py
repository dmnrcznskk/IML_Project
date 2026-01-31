import os
import tensorflow as tf
from typing import List, Dict, Any

def get_callbacks(config: Dict[str, Any], model_name: str) -> List[tf.keras.callbacks.Callback]:
    """
    Tworzy listę callbacków do treningu modelu Keras.

    Args:
        config (Dict[str, Any]): Konfiguracja treningu. Oczekiwane klucze (opcjonalne):
            - 'patience': int (ile epok bez poprawy czekać, domyślnie 10)
            - 'log_dir': str (gdzie zapisywać logi, domyślnie 'logs')
            - 'checkpoint_dir': str (gdzie zapisać najlepszy model, domyślnie 'models/checkpoints')
        model_name (str): Nazwa modelu (używana do nazywania plików).

    Returns:
        List[tf.keras.callbacks.Callback]: Lista callbacków.
    """
    callbacks = []

    # Pobieranie parametrów z configu (z wartościami domyślnymi)
    patience = config.get("patience", 10)
    base_log_dir = config.get("log_dir", "logs")
    checkpoint_dir = config.get("checkpoint_dir", "models/checkpoints")
    
    # 1. Tworzenie ścieżek
    # Logi TensorBoard: logs/TrafficSignConvNN/20250131-1430
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base_log_dir, model_name, timestamp)
    
    # Ścieżka do zapisu modelu: models/checkpoints/TrafficSignConvNN_best.keras
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"{model_name}_best.keras")

    # 2. ModelCheckpoint - zapisuje TYLKO najlepszy model
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_accuracy",   # Patrzymy na dokładność walidacyjną
        save_best_only=True,      # Nadpisuj tylko jeśli wynik jest lepszy
        mode="max",               # Chcemy jak największą dokładność
        verbose=1
    )
    callbacks.append(checkpoint_cb)

    # 3. EarlyStopping - przerywa trening jak nie ma postępów
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",       # Patrzymy na stratę walidacyjną (bardziej stabilna niż accuracy)
        patience=patience,        # Ile epok czekać na poprawę
        restore_best_weights=True,# Po przerwaniu przywróć wagi z najlepszego momentu
        verbose=1
    )
    callbacks.append(early_stopping_cb)

    # 4. ReduceLROnPlateau - zmniejsza learning rate jak utknie
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,               # Zmniejsz LR o połowę
        patience=patience // 2,   # Czekaj połowę tego co EarlyStopping
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr_cb)

    # 5. TensorBoard - wykresy
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1          # Loguj histogramy wag co epokę
    )
    callbacks.append(tensorboard_cb)

    # 6. CSVLogger - zapisuje historię do pliku tekstowego
    csv_log_path = os.path.join(base_log_dir, model_name, f"history_{timestamp}.csv")
    os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
    csv_logger_cb = tf.keras.callbacks.CSVLogger(csv_log_path)
    callbacks.append(csv_logger_cb)

    print(f"[Callbacks] Logi TensorBoard: {log_dir}")
    print(f"[Callbacks] Model Checkpoint: {model_path}")

    return callbacks