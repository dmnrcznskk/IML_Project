import os
from typing import List, Union, Optional, Tuple

import cv2
import numpy as np


def load_images_from_paths(
    paths: Union[List[str], np.ndarray, str], 
    target_size: Optional[Tuple[int, int]] = (32, 32)
) -> np.ndarray:
    """
    Wczytuje zdjęcia z podanych ścieżek i zwraca jako macierz numpy.
    
    Args:
        paths: Pojedyncza ścieżka (str) lub lista/tablica ścieżek.
        target_size: Docelowy rozmiar (H, W). Domyślnie (32, 32). 
                     Ustaw None, jeśli chcesz oryginalny rozmiar.
    
    Returns:
        np.ndarray: Tablica 4D (N, H, W, 3) dla listy lub 3D (H, W, 3) dla pojedynczego.
    """
    
    # Normalizacja wejścia: jeśli pojedynczy string, zamień na listę
    is_single = isinstance(paths, str) or (isinstance(paths, np.ndarray) and paths.ndim == 0)
    
    if is_single:
        path_list = [str(paths)]
    elif isinstance(paths, np.ndarray):
        path_list = paths.tolist()
    else:
        path_list = paths

    loaded_images = []

    for path in path_list:
        if not isinstance(path, str):
            print(f"[SKIP] Nieprawidłowa ścieżka (nie string): {path}")
            continue

        if not os.path.exists(path):
            print(f"[SKIP] Plik nie istnieje: {path}")
            continue

        # Wczytanie (OpenCV ładuje w BGR)
        img = cv2.imread(path)

        if img is None:
            print(f"[SKIP] OpenCV nie mógł odczytać pliku: {path}")
            continue
        
        # Resize (jeśli wymagany)
        if target_size is not None:
            try:
                img = cv2.resize(img, target_size)
            except cv2.error:
                print(f"[SKIP] Błąd resize dla: {path}")
                continue

        loaded_images.append(img)

    if not loaded_images:
        # Zwracamy pustą tablicę o odpowiednim kształcie, żeby kod się nie wywalił
        return np.array([])

    final_array = np.array(loaded_images)

    # Jeśli wejście było pojedynczą ścieżką, zwróć pojedynczy obraz (3D), a nie batch (4D)
    if is_single and len(final_array) > 0:
        return final_array[0]

    return final_array