import tensorflow as tf
from backend.data.pipeline import DataPipeline
import os

def start_api() -> None:
    """
    Uruchamia serwer FastAPI
    """
    pass


def start_dev() -> None:
    '''
    Uruchamianie dla developerów backendu
    '''
    print(">>> Start: Pobieranie i przetwarzanie danych... <<<")

    # Inicjalizacja pipeline'u (z włączonym balansowaniem klas)
    pipeline = DataPipeline(balance_data=True)

    # Pobranie gotowych ramek danych
    train_df, val_df, test_df = pipeline.get_data()

    print("\n" + "="*60)
    print(f" PODSUMOWANIE DANYCH")
    print("="*60)
    
    # 1. Zbiór Treningowy
    print(f"\n[ZBIÓR TRENINGOWY] Liczba próbek: {len(train_df)}")
    print("-" * 30)
    print(train_df.head())
    print(f"\nLiczba unikalnych klas: {train_df['ClassId'].nunique()}")
    
    # 2. Zbiór Walidacyjny
    print(f"\n[ZBIÓR WALIDACYJNY] Liczba próbek: {len(val_df)}")
    print("-" * 30)
    print(val_df.head())

    # 3. Zbiór Testowy
    print(f"\n[ZBIÓR TESTOWY] Liczba próbek: {len(test_df)}")
    print("-" * 30)
    print(test_df.head())

    print("\n" + "="*60)
    print(">>> Zakończono sukcesem <<<")