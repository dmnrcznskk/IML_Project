import os

from PIL import Image


def map_ppm_to_png(src_root_path: str, dst_root_path: str = "png_dataset") -> str:
    """
    Konwertuje wszystkie pliki obrazów .ppm w podanej strukturze katalogów na format .png.
    Zachowuje oryginalną strukturę folderów w nowej lokalizacji.

    Args:
        src_root_path (str): Ścieżka korzenia do datasetu z plikami PPM.
        dst_root_path (str): Nazwa/ścieżka folderu wyjściowego dla plików PNG.

    Returns:
        str: Ścieżka do katalogu zawierającego przekonwertowane pliki.
    """
    for dirpath, _, filenames in os.walk(src_root_path):
        relative_path = os.path.relpath(dirpath, src_root_path)
        target_dir = os.path.join(dst_root_path, relative_path)
        os.makedirs(target_dir, exist_ok=True)

        for file in filenames:
            if file.lower().endswith(".ppm"):
                src_file = os.path.join(dirpath, file)
                dst_file = os.path.join(target_dir, os.path.splitext(file)[0] + ".png")
                
                try:
                    with Image.open(src_file) as img:
                        img.save(dst_file)
                except Exception:
                    pass
    
    return dst_root_path