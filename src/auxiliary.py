import os
import re
from math import pi as _pi
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

pd.options.mode.chained_assignment = None
# default='warn', but to avoid SettingWithCopyWarning

INPUT_PATH: str = "input"
OUTPUT_PATH: str = "output"
os.makedirs(INPUT_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Get repository root path
ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# I/O FUNCTIONS

__read_csv: callable = lambda f: pd.read_csv(  # noqa: E731
    f,
    parse_dates=["time"],
    infer_datetime_format=True,  # date_format="%Y-%m-%d %H:%M:%S",
    index_col="sample",
)

TRAIN: pd.DataFrame = __read_csv(os.path.join(ROOT_PATH, INPUT_PATH, "train.csv"))
TEST: pd.DataFrame = __read_csv(os.path.join(ROOT_PATH, INPUT_PATH, "test.csv"))
TARGET_HEIGHT: float = 70.0
TARGET_COLS: List[str] = ["M", "Dir"]
USE_ONLY_U_V_AS_FEATURES: bool = True
USE_ONLY_U_V_AS_TARGET: bool = True


# #####################################################################
# MAIN AUXILIARY FUNCTIONS
# #####################################################################


def get_test_x() -> pd.DataFrame:
    return _get_data(train=False, obs=False)


def get_train_x() -> pd.DataFrame:
    return _get_data(train=True, obs=False)


def get_train_y() -> pd.DataFrame:
    return _get_data(train=True, obs=True)


# AUXILIARY FUNCTIONS


def _get_data(
    train: bool,
    obs: bool,
    _use_only_u_v_as_features: bool = None,
    _use_only_u_v_as_target: bool = None,
) -> pd.DataFrame:
    _use_only_u_v_as_features, _use_only_u_v_as_target = ___set_bool_vars(
        _use_only_u_v_as_features, _use_only_u_v_as_target
    )
    if obs:
        assert train, "The label (y) only exists for the train set!"
        df: pd.DataFrame = TRAIN[
            [_c for _c in TRAIN.columns if _c in ("time", *TARGET_COLS)]
        ]
    else:
        df: pd.DataFrame = TRAIN.drop(columns=TARGET_COLS, axis=1) if train else TEST

        if _use_only_u_v_as_features:
            df = df[
                ["time"] + [col for col in df.columns if col.startswith(("U_", "V_"))]
            ]

    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    if not obs:
        ds = __times_df_to_ds(df)
        ds = ds.combine_first(ds.interp(lev=TARGET_HEIGHT).expand_dims("lev"))
        if not _use_only_u_v_as_features:
            ds = __add_wind_variables_to_ds(ds)
    else:
        ds = __obs_df_to_ds(df)
        ds = __add_wind_components_to_ds(ds)
        if _use_only_u_v_as_target:
            ds = ds[["U", "V"]]

    df = __ds_to_df(ds)
    return df


def ___set_bool_vars(_use_only_u_v_as_features, _use_only_u_v_as_target):
    if _use_only_u_v_as_features is None:
        _use_only_u_v_as_features = USE_ONLY_U_V_AS_FEATURES
    if _use_only_u_v_as_target is None:
        _use_only_u_v_as_target = USE_ONLY_U_V_AS_TARGET
    return _use_only_u_v_as_features, _use_only_u_v_as_target


def __ds_to_df(ds: xr.Dataset) -> pd.DataFrame:
    df = ds.to_dataframe().reset_index()
    df_pivoted = df.pivot(index="time", columns="lev", values=list(ds.data_vars))
    df_pivoted.columns = ["{}_{}".format(var, lev) for var, lev in df_pivoted.columns]
    return df_pivoted


def __add_wind_components_to_ds(ds: xr.Dataset) -> xr.Dataset:
    """
    Add zonal (U) and meridional (V) wind components to the dataset.
    """
    # calculate U
    if "U" not in ds:
        if "M" in ds and "Dir" in ds:
            u = -ds["M"] * np.sin(ds["Dir"] * np.pi / 180.0)
            ds["U"] = u.rename("U")
        else:
            raise ValueError("Cannot obtain U (no Dir or M in dataset)")

    # calculate V
    if "V" not in ds:
        if "M" in ds and "Dir" in ds:
            v = -ds["M"] * np.cos(ds["Dir"] * np.pi / 180.0)
            ds["V"] = v.rename("V")
        else:
            raise ValueError("Cannot obtain V (no Dir or M in dataset)")
    return ds


def __add_wind_variables_to_ds(ds: xr.Dataset) -> xr.Dataset:
    """
    Add wind magnitude and direction variables to a dataset of wind components.
    """
    ds["M"] = (ds["U"] ** 2 + ds["V"] ** 2) ** (1 / 2)
    # ds['M'] = xr.DataArray(ds['M'], dims=ds['U'].dims, coords=ds['U'].coords)
    ds["Dir"] = np.arctan2(ds["U"], ds["V"]) * 180 / _pi + 180
    # ds['Dir'] = xr.DataArray(ds['Dir'], dims=ds['U'].dims, coords=ds['U'].coords)
    return ds


# MARTA's TRANSFORMATION FUNCTION


def __times_df_to_ds(df: pd.DataFrame) -> xr.Dataset:
    df.reset_index(inplace=True)
    if USE_ONLY_U_V_AS_FEATURES:
        vars_no_levs = []
        vars_with_levs = ["U", "V"]
    else:
        vars_no_levs = ["RMOL", "UST", "PBLH", "HFX"]
        vars_with_levs = ["U", "V", "W", "T", "P", "RH", "SD"]
    levs = [float(x.split("_")[1]) for x in df.columns if "U_" in x]

    ds_no_levs = df[["time", *vars_no_levs]].set_index("time").to_xarray()
    list_with_levs = []
    for lev in levs:
        this_vars = {v + f"_{lev:.2f}": v for v in vars_with_levs}
        ds_this_lev = df[["time", *this_vars.keys()]].set_index("time").to_xarray()
        ds_this_lev = ds_this_lev.expand_dims({"lev": [lev]})
        ds_this_lev = ds_this_lev.rename_vars(this_vars)
        list_with_levs.append(ds_this_lev)
    ds_levs = xr.concat(list_with_levs, dim="lev")
    ds_all = xr.merge([ds_levs, ds_no_levs])
    return ds_all.transpose("time", "lev")


# GERARD's TRANSFORMATION FUNCTION


def __obs_df_to_ds(df: pd.DataFrame) -> xr.Dataset:
    ds = df.reset_index().set_index("time").to_xarray()
    ds = ds.expand_dims({"lev": [TARGET_HEIGHT]})
    return ds.transpose("time", "lev")


#  ARNAU's TRANSFORMATION FUNCTION


def __df_to_ds__depr__(df: pd.DataFrame) -> xr.Dataset:
    for col in df.columns:
        if "_" in col:
            new_col = col.replace("_", "@")
            df.rename(columns={col: new_col}, inplace=True)
    heights_pattern = r"\d+\.\d+|\d+"
    heights_list = set(
        float(re.search(heights_pattern, name).group())
        for name in df.columns.tolist()
        if re.search(heights_pattern, name)
    )
    unique_heights_list = sorted(list(heights_list))
    vars_pattern = r"([A-Za-z_]+)@\d+"
    vars_list = [
        re.match(vars_pattern, col).group(1)
        for col in df.columns.tolist()
        if re.match(vars_pattern, col)
    ]
    unique_vars_list = sorted(set(vars_list))
    time = pd.to_datetime(df["time"])
    lev = unique_heights_list
    coords = {"time": ("time", time), "lev": ("lev", lev)}
    data_info = {}

    for _vars in unique_vars_list:
        cols = [f"{_vars}@{_l}" for _l in unique_heights_list]
        data_info[_vars] = df[df.columns.intersection(cols)].values

    nan_filled_data_info = {
        var: np.full((len(time), len(lev)), np.nan) for var in unique_vars_list
    }
    for var in unique_vars_list:
        for i, height in enumerate(lev):
            col_name = f"{var}@{height:.2f}"
            if col_name in df.columns:
                nan_filled_data_info[var][:, i] = df[col_name].values
    return xr.Dataset(
        data_vars={
            var: xr.Variable(data=nan_filled_data_info[var], dims=["time", "lev"])
            for var in unique_vars_list
        },
        coords=coords,
    )


# AND PLOTTING FUNCTION...


def plot_wind_speed_hist():
    kwargs = {"_use_only_u_v_as_features": False, "_use_only_u_v_as_target": False}
    obs: pd.Series = _get_data(train=True, obs=True, **kwargs)[f"M_{TARGET_HEIGHT}"]
    train: pd.Series = _get_data(train=True, obs=False, **kwargs)[f"M_{TARGET_HEIGHT}"]
    test: pd.Series = _get_data(train=False, obs=False, **kwargs)[f"M_{TARGET_HEIGHT}"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    merged = pd.DataFrame({"train": train, "obs": obs})
    sns.kdeplot(
        data=merged["train"],
        label="train",
        fill=True,
        color="orange",
        alpha=0.3,
        ax=axes[0],
    )
    sns.kdeplot(
        data=merged["obs"],
        label="obs",
        fill=True,
        color="lightblue",
        alpha=0.3,
        ax=axes[0],
    )
    axes[0].legend()
    axes[0].set_xlabel("M [$ms^{-1}$]")
    axes[0].set_title("X vs. Y in train split")

    merged = pd.DataFrame({"train": train, "test": test})
    sns.kdeplot(
        data=merged["train"],
        label="train",
        fill=True,
        color="orange",
        alpha=0.3,
        ax=axes[1],
    )
    sns.kdeplot(
        data=merged["test"],
        label="test",
        fill=True,
        color="lightblue",
        alpha=0.3,
        ax=axes[1],
    )
    axes[1].legend()
    axes[1].set_xlabel("M [$ms^{-1}$]")
    axes[1].set_title("Train vs. test for X")

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join("output", "wind_speed_eda_kdes.png"))


if __name__ == "__main__":
    # _train_x = get_train_x()
    # _train_y = get_train_y()
    # _test_x = get_test_x()
    # print("\nFEATURES\n")
    # print("TRAIN", _train_x)
    # print("TEST", _test_x)
    # print("\nLABELS\n")
    # print("TRAIN", _train_y)
    plot_wind_speed_hist()
