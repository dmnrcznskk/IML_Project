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

    patience = config.get("patience", 10)
    base_log_dir = config.get("log_dir", "logs")
    checkpoint_dir = config.get("checkpoint_dir", "models/checkpoints")
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base_log_dir, model_name, timestamp)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"{model_name}_best.keras")

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_accuracy", 
        save_best_only=True, 
        mode="max",    
        verbose=1
    )
    callbacks.append(checkpoint_cb)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping_cb)

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5, 
        patience=patience // 2,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr_cb)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1 
    )
    callbacks.append(tensorboard_cb)

    csv_log_path = os.path.join(base_log_dir, model_name, f"history_{timestamp}.csv")
    os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
    csv_logger_cb = tf.keras.callbacks.CSVLogger(csv_log_path)
    callbacks.append(csv_logger_cb)

    print(f"[Callbacks] Logi TensorBoard: {log_dir}")
    print(f"[Callbacks] Model Checkpoint: {model_path}")

    return callbacks