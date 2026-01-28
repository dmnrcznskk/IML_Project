import os
import pandas as pd
from PIL import Image
from typing import Dict, Any, Union

from backend.mappers.map_classes import get_belgium_mapping, get_polish_mapping

def find_target_folder(base_path: str, target_name: str) -> str:
    """
    Rekurencyjnie przeszukuje katalog w poszukiwaniu folderu o zadanej nazwie.

    Args:
        base_path (str): Ścieżka początkowa wyszukiwania.
        target_name (str): Nazwa szukanego folderu (np. 'Training').

    Returns:
        str: Pełna ścieżka do znalezionego folderu.

    Raises:
        FileNotFoundError: Gdy folder nie zostanie znaleziony.
    """
    if os.path.basename(base_path) == target_name:
        return base_path

    if os.path.isdir(os.path.join(base_path, target_name)):
        return os.path.join(base_path, target_name)

    for root, dirs, _ in os.walk(base_path):
        if target_name in dirs:
            return os.path.join(root, target_name)
            
    raise FileNotFoundError(f"Nie znaleziono folderu '{target_name}' wewnątrz {base_path}")

def to_german_standard(path: str, mapping_dict: Dict[Union[str, int], int], subset_name: str = "Training") -> pd.DataFrame:
    """
    Mapuje strukturę folderów i plików dowolnego datasetu do ujednoliconego formatu dataframe,
    zgodnego ze standardem GTSRB.

    Args:
        path (str): Ścieżka do głównego katalogu datasetu.
        mapping_dict (Dict[Union[str, int], int]): Słownik mapujący nazwy folderów/klas na ID GTSRB.
        subset_name (str): Nazwa podfolderu z danymi (np. 'Training' lub 'classification').

    Returns:
        pd.DataFrame: Ramka danych zawierająca kolumny: Width, Height, Roi.X1/Y1/X2/Y2, ClassId, Path.
    """
    real_path = find_target_folder(path, subset_name)
    data = []
    
    try:
        class_folders = sorted(os.listdir(real_path))
    except FileNotFoundError:
        return pd.DataFrame()
    
    for folder_name in class_folders:
        folder_full_path = os.path.join(real_path, folder_name)
        
        if not os.path.isdir(folder_full_path):
            continue

        german_id = None

        if folder_name in mapping_dict:
            german_id = mapping_dict[folder_name]
        else:
            try:
                folder_id_int = int(folder_name)
                if folder_id_int in mapping_dict:
                    german_id = mapping_dict[folder_id_int]
            except ValueError:
                pass 

        if german_id is None:
            continue

        for file in os.listdir(folder_full_path):
            if file.lower().endswith(('.ppm', '.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_full_path, file)
                
                # Pobieramy wymiary obrazka
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                except Exception:
                    continue
                
                data.append({
                    "Width": w,
                    "Height": h,
                    "Roi.X1": -1,
                    "Roi.Y1": -1,
                    "Roi.X2": -1,
                    "Roi.Y2": -1,
                    "ClassId": german_id,
                    "Path": img_path,
                })

    return pd.DataFrame(data)