from typing import Dict, Union

def get_belgium_mapping() -> Dict[int, int]:
    """
    Zwraca mapowanie identyfikatorów klas belgijskich na niemieckie (GTSRB).

    Returns:
        Dict[int, int]: Słownik mapujący ID belgijskie (klucz) na ID GTSRB (wartość).
    """
    return {
        32: 4, 31: 9, 24: 10, 17: 11, 61: 12, 19: 13, 21: 14, 25: 16,
        22: 17, 13: 18, 3: 19, 4: 20, 5: 21, 0: 22, 2: 23, 16: 24,
        10: 25, 11: 26, 7: 28, 8: 29, 35: 33, 34: 35, 36: 36, 37: 40,
    }

def get_polish_mapping() -> Dict[str, int]:
    """
    Zwraca mapowanie polskich nazw znaków (np. 'A1', 'B20') na identyfikatory GTSRB.

    Returns:
        Dict[str, int]: Słownik mapujący polskie kody znaków na ID GTSRB.
    """
    return {
        'A1': 20, 'A2': 19, 'A17': 28, 'A30': 18, 'A7': 13,
        'B1': 15, 'B2': 17, 'B20': 14, 'C12': 40, 'D1': 12,
    }

def get_classes_to_names() -> Dict[int, str]:
    """
    Zwraca słownik opisujący znaczenie poszczególnych klas GTSRB w języku polskim.

    Returns:
        Dict[int, str]: Słownik mapujący ID klasy na jej polską nazwę.
    """
    return {
        0: 'Ograniczenie 20km/h',
        1: 'Ograniczenie 30km/h',
        2: 'Ograniczenie 50km/h',
        3: 'Ograniczenie 60km/h',
        4: 'Ograniczenie 70km/h',
        5: 'Ograniczenie 80km/h',
        6: 'Koniec ogr. 80km/h',
        7: 'Ograniczenie 100km/h',
        8: 'Ograniczenie 120km/h',
        9: 'Zakaz wyprzedzania',
        10: 'Zakaz wyprz. (ciężarowe)',
        11: 'Pierwszeństwo na skrzyż.',
        12: 'Droga z pierwszeństwem',
        13: 'Ustąp pierwszeństwa',
        14: 'Stop',
        15: 'Zakaz ruchu',
        16: 'Zakaz wjazdu (ciężarowe)',
        17: 'Zakaz wjazdu',
        18: 'Inne niebezpieczeństwo',
        19: 'Niebezp. zakręt w lewo',
        20: 'Niebezp. zakręt w prawo',
        21: 'Seria zakrętów',
        22: 'Nierówna droga',
        23: 'Śliska nawierzchnia',
        24: 'Zwężenie z prawej',
        25: 'Roboty drogowe',
        26: 'Sygnalizacja świetlna',
        27: 'Piesi',
        28: 'Dzieci',
        29: 'Rowerzyści',
        30: 'Oszronienie / Śnieg',
        31: 'Dzikie zwierzęta',
        32: 'Koniec wszystkich zakazów',
        33: 'Nakaz skrętu w prawo',
        34: 'Nakaz skrętu w lewo',
        35: 'Nakaz jazdy prosto',
        36: 'Prosto lub w prawo',
        37: 'Prosto lub w lewo',
        38: 'Nakaz jazdy z prawej',
        39: 'Nakaz jazdy z lewej',
        40: 'Ruch okrężny (Rondo)',
        41: 'Koniec zakazu wyprzedzania',
        42: 'Koniec zakazu wyprz. (ciężarowe)'
    }