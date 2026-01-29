import pandas as pd
import numpy as np
from typing import List, Tuple, Union

from backend.data.providers import BaseDatasetProvider, GermanDatasetProvider, PolishDatasetProvider, BelgiumDatasetProvider
from backend.data import balance

class DataPipeline:
    """
    Zarządza procesem pozyskiwania danych, łączenia różnych zbiorów oraz ich balansowaniem.
    Zwraca dataframe lub krotki numpy array, gotowe do dalszego przetwarzania przez modele.
    """

    def __init__(self, balance_data: bool = True, return_as_tuple: bool = False):
        """
        Inicjalizuje pipeline danych.

        Args:
            balance_data (bool): Czy stosować oversampling (balansowanie) klas w zbiorze treningowym.
            return_as_tuple (bool): Czy zwracać dane jako krotki (X, y) typu np.ndarray (dla scikit-learn).
        """
        self.balance_data = balance_data
        self.return_as_tuple = return_as_tuple
        
        # Lista dostawców danych, z których będziemy korzystać
        self.providers: List[BaseDatasetProvider] = [
            GermanDatasetProvider(),
            PolishDatasetProvider(),
            BelgiumDatasetProvider()
        ]

    def _merge_dataframes(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Łączy listę dataframe danych w jeden duzy.

        Args:
            dfs (List[pd.DataFrame]): Lista dataframów do połączenia.

        Returns:
            pd.DataFrame: Połączony dataframe.
        """
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def get_data(self) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        """
        Pobiera dane od wszystkich dostawców, łączy je i opcjonalnie balansuje zbiór treningowy.

        Returns:
            Union:
            - Jeśli return_as_tuple=False: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] (Train, Val, Test)
            - Jeśli return_as_tuple=True: Tuple[Tuple[X_train, y_train], Tuple[X_val, y_val], Tuple[X_test, y_test]]
              gdzie X i y to np.ndarray.
        """
        train_dfs: List[pd.DataFrame] = []
        val_dfs: List[pd.DataFrame] = []
        test_dfs: List[pd.DataFrame] = []

        for provider in self.providers:
            p_train, p_val, p_test = provider.get_data()
            train_dfs.append(p_train)
            val_dfs.append(p_val)
            test_dfs.append(p_test)

        full_train_df = self._merge_dataframes(train_dfs)
        full_val_df = self._merge_dataframes(val_dfs)
        full_test_df = self._merge_dataframes(test_dfs)

        if self.balance_data:
            full_train_df = balance.balance_dataframe(full_train_df)

        if self.return_as_tuple:
            def split_xy_numpy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
                if df.empty:
                    return np.array([]), np.array([])
                
                # Zakładamy, że 'ClassId' to kolumna docelowa (y)
                y = df['ClassId'].to_numpy()
                # Reszta kolumn to cechy (X)
                X = df.drop(columns=['ClassId']).to_numpy()
                return X, y

            return split_xy_numpy(full_train_df), split_xy_numpy(full_val_df), split_xy_numpy(full_test_df)

        return full_train_df, full_val_df, full_test_df