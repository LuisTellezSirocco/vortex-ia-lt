"""
QUALITY of WIND Time Series

Uses the utility script 'vtxpy' to avoid importing 
things from the module vortexpy. You can use this metric outside 
kaggle by importing all_sources_stats, add_all_deduced_stat.
"""
import math
from typing import Tuple

import numpy as np
import pandas as pd
from vtxpy import add_all_deduced_stats, all_sources_stats, compute_quality

kaggle_vars = {
    # WIND & POWER
    "M": {
        "absolute bias (%)": (0.0, 2.0, 3),
        "EMD": (0.0, 0.3, 3),
        "absolute bias_q0.25 (%)": (0.0, 2.0, 1),
        "absolute bias_q0.5 (%)": (0.0, 2.0, 1),
        "absolute bias_q0.75 (%)": (0.0, 3.0, 1),
        "absolute bias_q0.98 (%)": (0.0, 5.0, 1),
        "rmse": (0.0, 1.5, 1),
        "rmse_hourly": (0.0, 1.3, 1),
        "rmse_daily": (0.0, 1.0, 1),
        "rmse_monthly": (0.0, 0.2, 1),
        "rmse (%)": (0.0, 25.0, 1),
        "rmse_hourly (%)": (0.0, 20.0, 1),
        "rmse_daily (%)": (0.0, 12.0, 1),
        "rmse_monthly (%)": (0.0, 3.0, 1),
        "r2": (1.0, 0.75, 2),
        "r2_hourly": (1.0, 0.8, 2),
        "r2_daily": (1.0, 0.9, 1),
        "r2_monthly": (1.0, 0.95, 5),
    },
    "power": {
        "r2": (1.0, 0.7, 1),
        "r2_hourly": (1.0, 0.75, 1),
        "r2_daily": (1.0, 0.8, 1),
        "r2_monthly": (1.0, 0.9, 1),
        "rmse": (0.0, 1000.0, 1),
        "rmse_hourly": (0.0, 800.0, 1),
        "rmse_daily": (0.0, 200.0, 1),
        "rmse_monthly": (0.0, 100.0, 1),
        "rmse_daily (%)": (0.0, 20.0, 1),
        "rmse_monthly (%)": (0.0, 5.0, 1),
    },
    # Wind Rose
    "U": {
        "EMD": (0.0, 0.3, 1),
        "absolute bias_q0.25": (0.0, 0.2, 1),
        "absolute bias_q0.5": (0.0, 0.2, 1),
        "absolute bias_q0.75": (0.0, 0.2, 1),
        "r2": (1.0, 0.75, 1),
        "r2_hourly": (1.0, 0.8, 1),
        "r2_daily": (1.0, 0.9, 1),
        "r2_monthly": (1.0, 0.95, 1),
    },
    "V": {
        "EMD": (0.0, 0.3, 1),
        "absolute bias_q0.25": (0.0, 0.2, 1),
        "absolute bias_q0.5": (0.0, 0.2, 1),
        "absolute bias_q0.75": (0.0, 0.2, 1),
        "r2": (1.0, 0.75, 1),
        "r2_hourly": (1.0, 0.8, 1),
        "r2_daily": (1.0, 0.9, 1),
        "r2_monthly": (1.0, 0.95, 1),
    },
    "Dir": {
        "absolute bias": (0.0, 3.0, 5),
        "rmse": (0.0, 25.0, 5),
        "r2": (1.0, 0.9, 1),
        "r2_hourly": (1.0, 0.92, 1),
        "r2_daily": (1.0, 0.96, 1),
        "r2_monthly": (1.0, 0.98, 1),
    },
    "TAB_pop": {"rmse": (0.0, 0.1, 3)},
    "A_sector": {
        "r2": (1.0, 0.9, 3),
        "rmse": (0.0, 0.4, 1),
        "weighted_rmse": (0.0, 0.3, 3),
        "weighted_MAE": (0.0, 0.4, 1),
    },
    "k": {
        "absolute bias": (0.0, 0.05, 1),
        "absolute bias (%)": (0.0, 2.0, 1),
    },
    "k_sector": {
        "r2": (1.0, 0.7, 1),
        "rmse": (0.0, 0.2, 1),
        "weighted_rmse": (0.0, 0.2, 1),
        "weighted_MAE": (0.0, 0.3, 1),
    },
    # Diurnal and seasonal cycles
    "M_mth": {
        "r2": (1.0, 0.95, 2),
        "rmse": (0.0, 0.1, 2),
    },
    "M_hr": {"r2": (1.0, 0.95, 2), "rmse": (0.0, 0.1, 2)},
    "M_mthhr": {"r2": (1.0, 0.9, 2), "rmse": (0.0, 0.15, 2)},
    "M_norm_mth": {
        "rmse": (0.0, 1.0, 1),
    },
    "M_norm_hr": {"rmse": (0.0, 1.0, 1)},
    "M_norm_mthhr": {"rmse": (0.0, 2.0, 1)},
    # Subhourly variability
    "M-noise": {
        "absolute bias": (0, 0.1, 3),
        "rmse": (0, 0.2, 1),
        "EMD": (0, 0.1, 1),
    },
    "M-ramp": {
        "EMD": (0, 0.25, 1),
    },
}


def assess_submisison(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
) -> Tuple[pd.DataFrame, float]:
    sources_df = {"solution": solution, "submission": submission}
    sources = {}
    for n, df in sources_df.items():
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        sources[n] = df[["M", "Dir"]].to_xarray()

    vars_list = ["M", "Dir", "power", "U", "V"]
    extra_vals = {
        "tabfile": {},
        "weibull": {"num_sectors": 16},
        "season_day": {
            "normalized": "all",
            "stat": "mean",
            "min_days": 20,
        },
        "ramps": {},
        "noise": {},
    }

    st = all_sources_stats(
        sources,
        name_ref="solution",
        vars_list=vars_list,
        extra_validations=extra_vals,
        hourly=True,
        daily=True,
        monthly=True,
        yearly=True,
    )
    stats_d = add_all_deduced_stats(st)

    dfq, quality = compute_quality(stats_d, my_vars=kaggle_vars)
    return dfq, quality


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
) -> float:
    """
    Quality Index Based on Multiple Wind Validations

    Wind and Power (time-domain and distribution)
    Wind Rose
    Seasonality and diurnal cycle
    Sub-hourly variability
    """

    dfq, quality = assess_submisison(solution=solution, submission=submission)
    return quality


def _prepare_df_to_asses_submision(
    df: pd.DataFrame, U_col_name: str = None, V_col_name: str = None
):
    """El propósito de esta función es preparar el dataframe para que sea
    compatible con la función assess_submisison, esto es, que tenga las columnas
    'time', 'M' y 'Dir'. Si no existen las columnas 'M' y 'Dir', se calcularán
    a partir de las componentes U y V.
    """
    assert isinstance(df, pd.DataFrame), "data debe ser un DataFrame de pandas"

    df_ = df.copy()

    # Comprobamos si existe una columna 'time', si no existe, vamos a comprobar si es el indice
    if "time" not in df_.columns:
        # comprobamos si el indice es de tipo fecha, en dicho caso, lo renombraremos
        if isinstance(df_.index, pd.DatetimeIndex):
            # check if index name is None
            if df_.index.name is None:
                df_.index.name = "time"
            else:
                df_ = df_.reset_index(drop=False)
        else:
            raise ValueError(
                "data debe tener una columna 'time' o ser un DataFrame con un indice de tipo fecha"
            )
    else:
        # Aseguramos que la columna 'time' sea de tipo fecha
        df_["time"] = pd.to_datetime(df_["time"])

    if "M" not in df_.columns or "Dir" not in df_.columns:
        assert (
            U_col_name is not None
        ), "Si no existe la columna 'M', debe proporcionarse el nombre de la columna de la componente U"
        assert (
            V_col_name is not None
        ), "Si no existe la columna 'M', debe proporcionarse el nombre de la columna de la componente V"
        assert U_col_name in df_.columns, f"La columna {U_col_name} no existe en data"
        assert V_col_name in df_.columns, f"La columna {V_col_name} no existe en data"

    # Comprobamos que existan las columnas 'M' y 'Dir'
    if "M" not in df_.columns:
        # Tenemos que calcular la velocidad a partir de las componentes U y V
        df_["M"] = (df_[U_col_name] ** 2 + df_[V_col_name] ** 2) ** 0.5

    if "Dir" not in df_.columns:
        # Tenemos que calcular la dirección a partir de las componentes U y V
        df_["Dir"] = np.arctan2(df_[U_col_name], df_[V_col_name]) * 180 / math.pi + 180

    return df_[["time", "M", "Dir"]].reset_index().rename(columns={"index": "sample"})
