import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def reduce_to32bits(df, silence: bool = True):
    """
    This function reduces the memory usage of a dataframe by converting the data types of the columns to 32 bits.
    :param df: dataframe to be reduced
    :return: dataframe with 32 bits data types
    """
    df_ = df.copy()

    start_mem = df_.memory_usage().sum() / 1024**2
    if not silence:
        print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df_.columns:
        if df_[col].dtype.name.startswith("float"):
            df_[col] = df_[col].astype("float32")
        elif df_[col].dtype.name.startswith("int"):
            df_[col] = df_[col].astype("int32")

    end_mem = df_.memory_usage().sum() / 1024**2
    if not silence:
        print(f"Memory usage after optimization is: {end_mem:.2f} MB")
        print(f"Decreased by {(100 * (start_mem - end_mem) / start_mem):.1f}%")

    return df_


def get_columns_with_low_std(dataframe: pd.DataFrame):
    """
    Returns a list of column names from the given dataframe that have a standard deviation less than or equal to 1.

    Parameters:
    dataframe (pd.DataFrame): The input dataframe.

    Returns:
    list: A list of column names with low standard deviation.
    """
    low_std_columns = []
    for column in dataframe.columns:
        if dataframe[column].std() <= 1:
            low_std_columns.append(column)
    return low_std_columns


def add_date_vars(df: pd.DataFrame) -> pd.DataFrame:
    df_ = df.copy()
    # fourier transforms
    df_features, features_added = get_date_features(df_)
    df_features, added_features = bulk_add_fourier_features(
        df_features,
        ["month", "hour", "dayofyear"],
        max_values=[12, 24, 365],
        n_fourier_terms=1,
        use_32_bit=True,
    )
    df_features = (
        df_features.drop(features_added, axis="columns")
        # .drop('Wind_power_MW', axis='columns')
    )

    return df_features, added_features


def get_date_features(
    df: pd.DataFrame, use_names: bool = False
) -> Tuple[pd.DataFrame, list]:
    df_ = df.copy()
    df_["weekday"] = df_.index.weekday
    df_["week"] = df_.index.isocalendar().week
    df_["day"] = df_.index.day
    df_["hour"] = df_.index.hour
    df_["date"] = df_.index.date
    df_["month"] = df_.index.month
    df_["year"] = df_.index.year
    df_["dayofyear"] = df_.index.day_of_year

    features_created = [
        "weekday",
        "week",
        "day",
        "hour",
        "date",
        "month",
        "year",
        "dayofyear",
    ]

    if use_names:
        df_["weekday_name"] = df_.index.day_name()
        df_["month_name"] = df_.index.month_name()

        # Making ordered categoricals to make for sorted plots
        df_["month_name"] = pd.Categorical(
            df_["month_name"],
            categories=[
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ],
            ordered=True,
        )
        df_["weekday_name"] = pd.Categorical(
            df_["weekday_name"],
            categories=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            ordered=True,
        )

    return df_, features_created


def bulk_add_fourier_features(
    df: pd.DataFrame,
    columns_to_encode: List[str],
    max_values: List[int],
    n_fourier_terms: int = 1,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Adds Fourier Terms for all the specified seasonal cycle columns, like month, week, hour, etc.

    Args:
        df (pd.DataFrame): The dataframe which has the seasonal cyycles which has to be encoded
        columns_to_encode (List[str]): The column names which has the seasonal cycle
        max_values (List[int]): The list of maximum values the seasonal cycles can attain in the
            same order as the columns to encode. for eg. for month, max_value is 12.
            If not given, it will be inferred from the data, but if the data does not have at least a
            single full cycle, the inferred max value will not be appropriate. Defaults to None
        n_fourier_terms (int): Number of fourier terms to be added. Defaults to 1
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.
    Raises:
        warnings.warn: Raises a warning if max_value is None

    Returns:
        [Tuple[pd.DataFrame, List]]: Returns a tuple of the new dataframe and a list of features which were added
    """
    assert len(columns_to_encode) == len(
        max_values
    ), "`columns_to_encode` and `max_values` should be of same length."
    added_features = []
    for column_to_encode, max_value in zip(columns_to_encode, max_values):
        df, features = add_fourier_features(
            df,
            column_to_encode,
            max_value,
            n_fourier_terms=n_fourier_terms,
            use_32_bit=use_32_bit,
        )
        added_features += features
    return df, added_features


def add_fourier_features(
    df: pd.DataFrame,
    column_to_encode: str,
    max_value: Optional[int] = None,
    n_fourier_terms: int = 1,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Adds Fourier Terms for the specified seasonal cycle column, like month, week, hour, etc.

    Args:
        df (pd.DataFrame): The dataframe which has the seasonal cyycles which has to be encoded
        column_to_encode (str): The column name which has the seasonal cycle
        max_value (int): The maximum value the seasonal cycle can attain. for eg. for month, max_value is 12.
            If not given, it will be inferred from the data, but if the data does not have at least a
            single full cycle, the inferred max value will not be appropriate. Defaults to None
        n_fourier_terms (int): Number of fourier terms to be added. Defaults to 1
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.
    Raises:
        warnings.warn: Raises a warning if max_value is None

    Returns:
        [Tuple[pd.DataFrame, List]]: Returns a tuple of the new dataframe and a list of features which were added
    """
    assert (
        column_to_encode in df.columns
    ), "`column_to_encode` should be a valid column name in the dataframe"
    assert is_numeric_dtype(
        df[column_to_encode]
    ), "`column_to_encode` should have numeric values."
    if max_value is None:
        max_value = df[column_to_encode].max()
        raise warnings.warn(
            "Inferring max cycle as {} from the data. This may not be accuracte if data is less than a single seasonal cycle."
        )
    fourier_features = _calculate_fourier_terms(
        df[column_to_encode].astype(int).values,
        max_cycle=max_value,
        n_fourier_terms=n_fourier_terms,
    )
    feature_names = [
        f"{column_to_encode}_sin_{i}" for i in range(1, n_fourier_terms + 1)
    ] + [f"{column_to_encode}_cos_{i}" for i in range(1, n_fourier_terms + 1)]
    df[feature_names] = fourier_features
    if use_32_bit:
        df[feature_names] = df[feature_names].astype("float32")
    return df, feature_names


def _calculate_fourier_terms(
    seasonal_cycle: np.ndarray, max_cycle: int, n_fourier_terms: int
):
    """Calculates Fourier Terms given the seasonal cycle and max_cycle"""
    sin_X = np.empty((len(seasonal_cycle), n_fourier_terms), dtype="float64")
    cos_X = np.empty((len(seasonal_cycle), n_fourier_terms), dtype="float64")
    for i in range(1, n_fourier_terms + 1):
        sin_X[:, i - 1] = np.sin((2 * np.pi * seasonal_cycle * i) / max_cycle)
        cos_X[:, i - 1] = np.cos((2 * np.pi * seasonal_cycle * i) / max_cycle)
    return np.hstack([sin_X, cos_X])
