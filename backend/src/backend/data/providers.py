import os
import kagglehub
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, List
from sklearn.model_selection import train_test_split

from backend.mappers.normalize import to_german_standard
from backend.mappers import map_classes
from backend.mappers.img_format import map_ppm_to_png

class BaseDatasetProvider(ABC):
    """
    Abstrakcyjna klasa bazowa definiująca interfejs dla dostawców danych znaków drogowych.
    """

    @abstractmethod
    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Pobiera dane, przetwarza je i dzieli na zbiory.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Krotka ramek danych (treningowa, walidacyjna, testowa).
        """
        pass

    def _download(self, handle: str) -> str:
        """
        Pobiera dataset z serwisu Kaggle.

        Args:
            handle (str): Identyfikator datasetu na Kaggle.

        Returns:
            str: Ścieżka do pobranego katalogu z danymi.
        """
        return kagglehub.dataset_download(handle)


class GermanDatasetProvider(BaseDatasetProvider):
    """
    Implementacja dostawcy dla niemieckiego zbioru GTSRB.
    """

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Pobiera niemiecki dataset, ładuje pliki CSV i konwertuje ścieżki.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Zbiory treningowy, walidacyjny i testowy.
        """
        path = self._download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
        train_df, test_df, _ = self._load_csv_files(path)
        
        train_df = self._convert_paths(path, train_df)
        test_df = self._convert_paths(path, test_df)
        
        train_split, val_split = train_test_split(
            train_df, test_size=0.2, random_state=50, stratify=train_df["ClassId"]
        )
        
        return train_split, val_split, test_df

    def _load_csv_files(self, dataset_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Wczytuje pliki Train.csv, Test.csv i Meta.csv z katalogu datasetu.

        Args:
            dataset_dir (str): Ścieżka do głównego katalogu danych.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Ramki danych dla zbioru treningowego, testowego i metadanych.

        Raises:
            FileNotFoundError: Gdy brakuje któregoś z wymaganych plików CSV.
        """
        train_csv = os.path.join(dataset_dir, "Train.csv")
        test_csv = os.path.join(dataset_dir, "Test.csv")
        meta_csv = os.path.join(dataset_dir, "Meta.csv")

        for path in [train_csv, test_csv, meta_csv]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Nie znaleziono wymaganego pliku: {path}")

        return pd.read_csv(train_csv), pd.read_csv(test_csv), pd.read_csv(meta_csv)

    def _convert_paths(self, dataset_path: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zamienia względne ścieżki w kolumnie Path na ścieżki absolutne.

        Args:
            dataset_path (str): Ścieżka bazowa datasetu.
            df (pd.DataFrame): Ramka danych do przetworzenia.

        Returns:
            pd.DataFrame: Ramka danych z poprawionymi ścieżkami.
        """
        df["Path"] = df["Path"].apply(lambda row: os.path.join(dataset_path, row))
        return df


class PolishDatasetProvider(BaseDatasetProvider):
    """
    Implementacja dostawcy dla polskiego zbioru znaków drogowych.
    """

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Pobiera polski dataset i mapuje klasy na standard niemiecki.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Zbiory treningowy, walidacyjny i testowy.
        """
        path = self._download("chriskjm/polish-traffic-signs-dataset")
        df = to_german_standard(path, map_classes.get_polish_mapping(), "classification")
        
        train_val, test_split = train_test_split(
            df, test_size=0.1, random_state=50, stratify=df["ClassId"]
        )
        train_split, val_split = train_test_split(
            train_val, test_size=0.1, random_state=50, stratify=train_val["ClassId"]
        )
        
        return train_split, val_split, test_split


class BelgiumDatasetProvider(BaseDatasetProvider):
    """
    Implementacja dostawcy dla belgijskiego zbioru znaków drogowych.
    """

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Pobiera belgijski dataset, konwertuje obrazy PPM na PNG i mapuje klasy.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Zbiory treningowy, walidacyjny i testowy.
        """
        path = self._download("mahadevkonar/belgiumts-dataset")
        data_path = map_ppm_to_png(path)
        
        train_df = to_german_standard(data_path, map_classes.get_belgium_mapping(), "Training")
        test_df = to_german_standard(data_path, map_classes.get_belgium_mapping(), "Testing")
        
        train_split, val_split = train_test_split(
            train_df, test_size=0.1, random_state=50, stratify=train_df["ClassId"]
        )
        
        return train_split, val_split, test_df