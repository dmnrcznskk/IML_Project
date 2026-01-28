import pandas as pd
from sklearn.utils import resample

def balance_dataframe(df: pd.DataFrame, column: str = 'ClassId') -> pd.DataFrame:
    """
    Balansuje dataframe poprzez oversampling klas mniejszościowych do liczebności najliczniejszej klasy.

    Args:
        df (pd.DataFrame): Ramka danych do zbalansowania.
        column (str): Nazwa kolumny zawierającej etykiety klas.

    Returns:
        pd.DataFrame: Zbalansowana ramka danych z przemieszanymi wierszami.
    """
    if df.empty:
        return df

    class_counts = df[column].value_counts()
    if len(class_counts) == 0:
        return df

    max_size = class_counts.max()
    balanced_groups = []

    for _, group in df.groupby(column):
        if len(group) < max_size:
            upsampled_group = resample(
                group,
                replace=True,
                n_samples=max_size,
                random_state=50
            )
            balanced_groups.append(upsampled_group)
        else:
            balanced_groups.append(group)

    balanced_df = pd.concat(balanced_groups, ignore_index=True)
    return balanced_df.sample(frac=1, random_state=50).reset_index(drop=True)