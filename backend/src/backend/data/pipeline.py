import pandas as pd
from typing import List, Tuple

from backend.data.providers import BaseDatasetProvider, GermanDatasetProvider, PolishDatasetProvider, BelgiumDatasetProvider
from backend.data import balance

class DataPipeline:
    """
    Zarządza procesem pozyskiwania danych, łączenia różnych zbiorów oraz ich balansowaniem.
    Zwraca dataframe, gotowe do dalszego przetwarzania.
    """

    def __init__(self, balance_data: bool = True):
        """
        Inicjalizuje pipeline danych.

        Args:
            balance_data (bool): Czy stosować oversampling (balansowanie) klas w zbiorze treningowym.
        """
        self.balance_data = balance_data
        
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

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Pobiera dane od wszystkich dostawców, łączy je i opcjonalnie balansuje zbiór treningowy.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            Krotka zawierająca odpowiednio:
            - Zbiór treningowy (Train)
            - Zbiór walidacyjny (Val)
            - Zbiór testowy (Test)
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

        return full_train_df, full_val_df, full_test_df