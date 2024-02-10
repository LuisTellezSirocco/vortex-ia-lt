import xarray as xr
import pandas as pd
import numpy as np
import math
import pandas.api.types
from scipy.stats import wasserstein_distance
from typing import Dict, Union, List, Any, Callable, Tuple
from scipy import stats
from scipy import interpolate
from statistics import mode
import functools
from scipy import optimize
from scipy.special import gamma

from vortex_module.calculate_score import kaggle_vars as vars_kaggle

vSet = Union[xr.Dataset, pd.DataFrame]
vArray = Union[xr.DataArray, pd.Series]
vData = Union[vSet, vArray]

circular_vars = ['Dir', 'Dir_rep']
meaningless_zero = circular_vars + ['U', 'V']
cumulative_vars = ['energy']
vars_aggregated_using_sum = ['energy', 'total_veer']

def calculate_index(metric, val_at_zero, val_at_ten):
    return (10 / (val_at_ten - val_at_zero)) * (metric - val_at_zero)

def view_limits_quality(my_vars=None):
    if my_vars is None:
        my_vars = vars_kaggle

    limits = []
    for v in my_vars:
        for metric in my_vars[v]:
            perfect, good, weight = my_vars[v][metric]
            limits.append({'var': v, 'metric': metric, 'perfect': perfect,
                          'good': good, 'weight': weight})
    return pd.DataFrame(limits)


def compute_quality(df: pd.DataFrame, my_vars=None) \
        -> Tuple[pd.DataFrame, float]:
    if my_vars is None:
        my_vars = vars_kaggle

    missing_samples = []
    limits = []
    for v in my_vars:
        for metric in my_vars[v]:
            perfect, good, weight = my_vars[v][metric]
            try:
                value = float(df[df['var'] == v][metric])
            except KeyError:
                missing_samples.append([v, metric])
                continue
            if np.isnan(value):
                continue
            index = calculate_index(value, val_at_zero=good,
                                    val_at_ten=perfect)
            limits.append({'var': v, 'metric': metric, 'perfect': perfect,
                           'good': good, 'weight': weight,
                           'value': value, 'quality_index': index,
                           'quality_weighted': weight * index})
    df_quality = pd.DataFrame(limits)

    if len(missing_samples) > 0:
        print('WARNING: There are missing variables for full quality computation')
        print(missing_samples)

    quality_sum = df_quality['quality_weighted'].sum()
    weight = df_quality['weight'].sum()
    quality = quality_sum / weight

    return df_quality, quality


def compare_stats(ref: xr.DataArray, new: xr.DataArray, var: str, dim='time',
                  emd=True, quantiles=True, quantiles_errors=True,
                  hourly=False, daily=False, monthly=False, yearly=False,
                  lfit=True, weights=None, only_dist_validation=False,
                  **kwargs) \
        -> Dict[str, float]:
    """
    Compare statistics of two time series DataArrays (one variable,
    a single point, no multiple levs).

    It returns a dictionary with metrics and statistics, such as 'mean',
    'bias', 'rmse', 'r2', 'length', ...  Extra statistics can be activated
    or deactivated using booleans.

    Parameters
    ----------
    ref: xr.DataArray
    new: xr.DataArray
    var: str
        You need to pass the name of the variable because some vars
        need special treatment (energy is summed instead of averaged;
        Dir is a circular variable,...)
    dim: str
        Typically, the dimension we compute the metrics on will be 'time',
        but we can use this function for other dimensions, such as 'sector'
        for example.
    emd: bool
        Earth-mover distance (ie, how different are the distributions?)
    quantiles: bool
        Save values of some quantiles of the time series: 0.02, 0.25,
        0.5, 0.75, 0.98
    quantiles_errors: bool
        Save the quantiles of the timeseries of the errors: 0.02, 0.25,
        0.5, 0.75, 0.98
    hourly: bool
        Resample to hourly and compute some statistics. USUALLY SLOW
    daily: bool
        Resample to daily and compute some statistics.
    monthly: bool
        Resample to monthly and compute some statistics.
    yearly: bool
        Resample to yearly and compute some statistics.
    lfit: bool
        Do linear fit and save the values for slope and intercept
    weights: xr.DataArray or None
        You can pass an array of weights to compute weighted rmse and
        MAE. It is useful when dim is not time, but some aggregated
        concept like 'sector' or 'Mbin' that can have a population.
    only_dist_validation: bool
        True if we don't want to validate in dim domain. For example,
        if we are validating wind speeds for a certain sector, we
        don't want to validate the rmse or correlation, only
        distribution metrics.

    Returns
    -------
    Dict[str, float]
        Dictionary of results with many metrics
    """

    ref = ref.squeeze(drop=True)
    new = new.squeeze(drop=True)

    try:
        too_short = len(ref) < 2
        too_short = len(new) < 2
    except TypeError:
        # TypeError: len() of unsized object
        too_short = True

    if too_short:
        print('too short compare stats')
        return {}

    results = {}
    samples = len(new.values)
    results['length'] = samples

    bias_length = samples - len(ref.values)
    if bias_length != 0 and not only_dist_validation:
        print('cannot do comparison in ' + dim + ' domain')
        only_dist_validation = True

    if only_dist_validation:
        results['bias_length'] = bias_length

    # MEAN VALUES
    if var in circular_vars:
        # just in case there are nans, we make sure we do the circular mean
        mean_ref = stats.circmean(ref.dropna(dim=dim, how='any').values,
                                  high=360.0)
        mean_new = stats.circmean(new.dropna(dim=dim, how='any').values,
                                  high=360.0)

        std_ref = stats.circstd(ref.dropna(dim=dim, how='any').values,
                                high=360.0)
        std_new = stats.circstd(new.dropna(dim=dim, how='any').values,
                                high=360.0)
    else:
        mean_ref = ref.mean(dim=dim).values
        mean_new = new.mean(dim=dim).values

        std_ref = ref.std(dim=dim).values
        std_new = new.std(dim=dim).values

    results['mean'] = float(mean_new)
    bias = mean_new - mean_ref

    if float(bias) == np.nan:
        raise ValueError('NaN values in the statistics calculation.')

    results['bias'] = float(bias)
    if var not in meaningless_zero:
        factor = mean_ref / mean_new
        results['factor'] = float(factor)

    if var in cumulative_vars:
        sum_ref = ref.sum(dim=dim).values
        sum_new = new.sum(dim=dim).values
        results['sum'] = float(sum_new)
        bias_sum = sum_new - sum_ref
        results['bias_sum'] = float(bias_sum)

    results['std'] = float(std_new)
    bias_std = std_new - std_ref
    results['std_bias'] = float(bias_std)

    # DISTRIBUTION
    if emd:
        emd = wasserstein_distance(ref, new)
        results['EMD'] = float(emd)

    if quantiles:
        for quant in [0.02, 0.25, 0.5, 0.75, 0.98]:
            q_ref = float(ref.quantile(quant, dim=dim).values)
            q_new = float(new.quantile(quant, dim=dim).values)
            results['q' + str(quant)] = q_new
            results['bias_q' + str(quant)] = q_new - q_ref

    max_ref = float(ref.max(dim=dim).values)
    results['max'] = float(new.max(dim=dim).values)
    results['bias_max'] = results['max'] - max_ref

    if var in circular_vars:
        for stat, bias in results.items():
            if 'bias' in stat:
                if bias > 180:
                    bias = bias - 360
                elif bias < -180:
                    bias = bias + 360
                results[stat] = bias

    # TIME DOMAIN
    if only_dist_validation:
        return results

    bias = new - ref

    if var in circular_vars:
        bias = bias.where(bias > -180, bias + 360)
        bias = bias.where(bias < 180, bias - 360)
        new = bias + ref

    error = xr.apply_ufunc(np.abs, bias, dask='allowed')

    results['MAE'] = float(error.mean(dim=dim).values)
    rmse = np.sqrt((bias ** 2).mean(dim=dim).values)
    results['rmse'] = float(rmse)

    if var in cumulative_vars:
        # sum of absolute errors
        results['SAE'] = float(error.sum(dim=dim).values)

    if weights is not None:
        # WEIGHTED VARS: population
        weighted_rmse = np.sqrt((bias ** 2).weighted(
            weights=weights).mean(dim=dim).values)
        results['weighted_rmse'] = float(weighted_rmse)

        weighted_mae = np.sqrt(error.weighted(
            weights=weights).mean(dim=dim).values)
        results['weighted_MAE'] = float(weighted_mae)

    new_series = new.to_series()
    ref_series = ref.to_series()
    r2 = new_series.corr(ref_series) ** 2
    results['r2'] = float(r2)
    
    if lfit:
        try:
            result = stats.linregress(ref_series, new_series)
        except ValueError:
            print('Could not do a linear fit')
        else:
            results['lfit_slope'] = float(result.slope)
            results['lfit_intercept'] = float(result.intercept)

    if quantiles_errors:
        for quant in [0.02, 0.25, 0.5, 0.75, 0.98]:
            q_bia = float(bias.quantile(quant, dim=dim).values)
            results['td_bias_q' + str(quant)] = q_bia

    if quantiles_errors:
        for quant in [0.02, 0.25, 0.5, 0.75, 0.98]:
            q_err = float(error.quantile(quant, dim=dim).values)
            results['td_error_q' + str(quant)] = float(q_err)

    if dim == 'time':
        some_times = new.coords['time'].values[:200]
        num_ts_in_one_hour = 60 // deduce_frequency(some_times)

        resamplings = {
            'hourly': hourly, 'daily': daily,
            'monthly': monthly, 'yearly': yearly,
        }
        for name_freq, do_freq in resamplings.items():
            if do_freq:
                # Do frequency hourly/daily/monthly as requested
                if name_freq == 'hourly' and num_ts_in_one_hour == 1:
                    for st in ['length', 'r2', 'rmse', 'MAE', 'SAE', 'EMD']:
                        if st in results:
                            results[st + '_hourly'] = results[st]
                else:
                    results_f = stats_in_resampling(ref, new, var, name_freq,
                                                    num_ts_in_one_hour)
                    results = {**results, **results_f}

    return results


def convert_tabfiles_to_population(tabfiles_sources: Dict[str, xr.DataArray],
                                   max_bin: float = None,
                                   min_samples_bin: int = 15) \
        -> Dict[str, xr.DataArray]:
    if min_samples_bin is None:
        min_samples_bin = 15

    # We must make sure all tabfiles end at the same wind speed
    if max_bin is None:
        max_bin = max([windbins.coords['speed'].values[-1]
                       for windbins in tabfiles_sources.values()])
        # if max_bin is 37.5, we need bins 0.5, 1.5,...37.5
    center_bins_m = np.arange(0.5, max_bin + 1, 1)

    populations_bins = {}
    for s, windbins_s in tabfiles_sources.items():
        # reindex all windbins to same reference speeds
        windbins_s = windbins_s.reindex(speed=center_bins_m)
        numcases = windbins_s.sum(dim=['speed', 'direction'])
        # set to 0 all NaN and too empty bins (less than `min_samples_bin`)
        windbins_s = windbins_s.where(windbins_s > min_samples_bin, 0.)
        # convert number of cases to percentage (because we may have
        # passed sources that cover different periods)
        population_s = 100 * windbins_s.astype('float') / numcases
        population_s = population_s.assign_attrs({'long_name':
                                                      'Bin Population (%)'})
        populations_bins[s] = population_s

    return populations_bins


def compare_tabfiles(sources: Dict[str, xr.Dataset],
                     name_ref: str = 'ref',
                     info: Dict[str, Any] = None,
                     num_sectors: int = 16,
                     max_bin: float = None,
                     min_samples_bin: int = 15,
                     validate_relative: bool = False,
                     **kwargs) -> pd.DataFrame:

    tabfiles_sources = {s: distribution_da(ds, num_sectors=num_sectors,
                                           max_bin=max_bin)
                        for s, ds in sources.items()}

    populations_bins = convert_tabfiles_to_population(
        tabfiles_sources, max_bin=max_bin, min_samples_bin=min_samples_bin)

    stats_list = []

    if info is None:
        info = {}

    for source, population in populations_bins.items():
        if source == name_ref:
            continue

        # Stack speed and direction on a single dimension
        ref_values = populations_bins[name_ref].stack(
            windbins=("speed", "direction")).dropna(dim='windbins')
        now_values = population.stack(
            windbins=("speed", "direction")).dropna(dim='windbins')

        # Compare POPULATIONS of bins (TAB_pop)
        stats_s = compare_stats(ref_values, now_values, 'TAB_pop',
                                dim='windbins')
        info_s = {**info, 'source': source, 'var': 'TAB_pop', **stats_s}
        stats_list.append(info_s)

        if validate_relative:
            # Considering only bins with at least 1% population,
            # validate the relative population of the current source
            # ie, if name_ref says 2% cases happen in a certain bin and
            # the source says 3%, its relative population is 150% and
            # will be validated versus the 100% "relative population" of name_ref
            ref_values = ref_values.where(ref_values > 1).dropna(dim='windbins')
            now_values = now_values.sel(windbins=ref_values.coords['windbins'])
            now_values = now_values.dropna(dim='windbins')
            relpop = 100 * now_values / ref_values.values
            relpop_ref = 100 * ref_values / ref_values.values

            # Compare relative POPULATIONS of bins (TAB_relpop)
            if len(relpop) < 3 or len(relpop_ref) < 3:
                continue

            stats_r = compare_stats(relpop, relpop_ref, 'TAB_relpop',
                                    dim='windbins')
            info_rel = {**info, 'source': source, 'var': 'TAB_relpop',
                        **stats_r}
            stats_list.append(info_rel)

    df = pd.DataFrame(stats_list)

    return df

def compare_weibulls(sources: Dict[str, xr.Dataset],
                     name_ref: str = 'ref',
                     info: Dict[str, Any] = None, 
                     num_sectors: int = None, 
                     method: str = 'vtx_operative', **kwargs):

    if 'k' in sources[name_ref] and 'A' in sources[name_ref]:
        weibull_sources = sources
        sources = None
        if groups is not None:
            raise ValueError('Cannot compare weibull by groups if you '
                             'pass directly the weibull dataset to '
                             'compare_weibulls. Please pass timeseries '
                             'sources.')
    else:
        weibull_sources = {s: weibull_ds(ds, num_sectors=num_sectors,
                                         method=method)
                           for s, ds in sources.items()}

    stats_list = []

    if info is None:
        info = {}

    for source, weibull in weibull_sources.items():
        if source == name_ref:
            continue

        for v in ['A', 'k']:
            ref_value = float(weibull_sources[name_ref][v])
            now_value = float(weibull[v])

            infoh = {**info, 'source': source,
                     'var': v, 'mean': now_value,
                     'bias': now_value - ref_value,
                     'factor': ref_value / now_value
                     }

            stats_list.append(infoh)

        if num_sectors is not None:
            for v in ['A_sector', 'k_sector']:
                ref_values = weibull_sources[name_ref][v]
                now_values = weibull[v]
                stats_s = compare_stats(ref_values, now_values, v,
                                        dim='sector', 
                                        weights=weibull['count_sector'])

                info_s = {**info, 'source': source,
                          'var': v, **stats_s}
                stats_list.append(info_s)

    df = pd.DataFrame(stats_list)

    return df



# Compare seasonal/daily

def compare_seasonal_daily_cycle(sources: Dict[str, xr.Dataset],
                                 stat='mean', min_days=20,
                                 name_ref: str = 'ref',
                                 info: Dict[str, Any] = None,
                                 vars_list: List[str] = None,
                                 normalized: Union[bool, str] = False,
                                 **kwargs) -> pd.DataFrame:

    if str(normalized) == 'all':
        df1 = compare_seasonal_daily_cycle(sources, name_ref=name_ref,
                                           stat=stat,
                                           min_days=min_days,
                                           info=info,
                                           vars_list=vars_list,
                                           normalized=False, **kwargs)
        df2 = compare_seasonal_daily_cycle(sources, name_ref=name_ref,
                                           stat=stat,
                                           min_days=min_days,
                                           info=info,
                                           vars_list=vars_list,
                                           normalized=True, **kwargs)

        df = pd.concat([df1, df2], ignore_index=True)
    else:
        cycles_sources = {s: seasonal_daily_cycle(ds, vars_list=vars_list,
                                                  normalized=normalized,
                                                  stat=stat,
                                                  min_days=min_days)
                          for s, ds in sources.items()}

        if vars_list is None:
            vars_list = [x for x in cycles_sources[name_ref].data_vars]

        stats_list = []

        if info is None:
            info = {}

        for source, cycles_source in cycles_sources.items():
            if source == name_ref:
                continue

            # Stack month and hour on a single dimension
            ref_values = cycles_sources[name_ref].stack(
                monthhour=("month", "hour")).dropna(dim='monthhour')
            now_values = cycles_source.stack(
                monthhour=("month", "hour")).dropna(dim='monthhour')

            # Compare values of bins (monthhour)
            for v in vars_list:
                if v == 'count':
                    continue

                var = str(v)
                if normalized:
                    var += '_norm'
                if stat != 'mean':
                    var += '_' + stat

                stats_s = compare_stats(ref_values[v],
                                        now_values[v], str(v),
                                        dim='monthhour')
                info_s = {**info, 'source': source,
                          'var': var + '_mthhr', **stats_s}
                stats_list.append(info_s)

                stats_s = compare_stats(cycles_sources[name_ref][v].mean(dim='hour'),
                                        cycles_source[v].mean(dim='hour'),
                                        str(v), dim='month')
                info_s = {**info, 'source': source,
                          'var': var + '_mth', **stats_s}
                stats_list.append(info_s)

                stats_s = compare_stats(cycles_sources[name_ref][v].mean(dim='month'),
                                        cycles_source[v].mean(dim='month'),
                                        str(v), dim='hour')
                info_s = {**info, 'source': source,
                          'var': var + '_hr', **stats_s}
                stats_list.append(info_s)

        df = pd.DataFrame(stats_list)

    return df


def distribution_da(vs: vSet,
                    num_sectors: int = 16,
                    max_bin: float = None) -> xr.DataArray:
    """
    Obtain the equivalent xr.DataArray to the distribution.nc files.

    Parameters
    ----------
    vs: vSet
        vSet from which we can find wind speed and direction
    num_sectors: int
        number of sectors of the distribution
    max_bin: float
        the highest wind speed bin. If None, the maximum speed bin is
        set as the maximum wind speed in this vSet.

    Returns
    -------
    dist: xr.DataArray
        named 'windbins', with dimensions ['lev', 'lat', 'lon',
        'direction', 'speed'], juts like distribution.nc files.

    Examples
    --------
    >>> print("Example (from a vSet that is a dask xr.object):")
    <xarray.DataArray 'windbins' (lev: 10, lat: 1, lon: 1,
                                    direction: 16, speed: 29)>
    dask.array<transpose, shape=(10, 1, 1, 16, 29), dtype=int32,
                chunksize=(10, 1, 1, 1, 1), chunktype=numpy.ndarray>
    Coordinates:
      * lat        (lat) float32 40.0
      * lev        (lev) float32 8.0 28.0 52.0 80.0 ... 220.0 263.0 330.0
      * lon        (lon) float32 -79.0
      * speed      (speed) float64 0.5 1.5 2.5 3.5 4.5 ...  26.5 27.5 28.5
      * direction  (direction) float64 0.0 22.5 45.0 67.5 ...  315.0 337.5

    """

    # Do we have the variables M and Dir? Find bins and sectors
    m = find_var('M', vs)
    m_bin = apply_general(np.trunc, m)
    # bins found truncating M. This is equal to width_bins=1 & at_zero='edge'

    direction = find_var('Dir', vs).to_dataset()
    dir_sector = find_var('sector', direction, num_sectors=num_sectors)

    # Wind speed bins coordinate
    if max_bin is None:
        max_bin = int(apply_general(np.trunc, m_bin.max()))
    center_bins_m = [x + 0.5 for x in range(max_bin)]
    speed = xr.DataArray(center_bins_m, dims=['speed'])

    # Direction Sectors coordinate
    width_sector = 360.0 / num_sectors
    center_sectors = [x * width_sector for x in range(num_sectors)]
    direction = xr.DataArray(center_sectors, dims=['direction'])

    # Unique variable 'windbins' to count bins/sectors
    # This is a trick that I think works well -> is it optimal? No way
    windbins = (m_bin + dir_sector / 1000).rename('windbins')
    # For example, an event with wind bin 15m/s and sector 3
    # will be encoded as 15.003. In that way we have a single time series
    # that includes both bin and sector information.

    arrays = []
    for s in range(num_sectors):
        bins = []
        for b in range(int(max_bin)):
            event_name = b + s / 1000
            # the events that can occur in the bin-sector range
            # are created and counted in the time series
            val = windbins.where(windbins == event_name,
                                 drop=False).count(dim='time')
            bins.append(val)
        vs = xr.concat(bins, dim='speed')
        arrays.append(vs)
    dist = xr.concat(arrays, dim='direction')

    # Dataset Properties to match Vortex's distribution.nc conventions
    dist['speed'] = speed
    dist['direction'] = direction
    dist = dist.astype('int32')

    dims_here = [d for d in ['lev', 'lat', 'lon', 'direction', 'speed']
                 if d in dist.dims]
    dist = dist.transpose(*dims_here)

    return dist


def seasonal_daily_cycle(vd: vData, vars_list: List[str] = None,
                         normalized: bool = False,
                         stat='mean', min_days: int = 20) -> xr.Dataset:
    """
    Seasonal-Daily Cycle (24x12)

    From a vData, obtain the mean values of several variables
    for each month/hour combination. If you request normalized=True,
    instead of the mean value, you obtain the percentage difference
    of that month/hour with respect to the full period mean.

    It can be used over a pandas if you pass a vars_list (using
    get_dataset).

    Parameters
    ----------
    vd: vData
        vData input object with time dimension (and possibly others)
    vars_list: List[str], optional
        List of variables for which to compute the cycle
    normalized: bool
        Whether to put the percentage difference with respect to full
        mean instead of the actual value.
    stat: str
        Keyword to specify which statistic we want to compute
        for all timestamps that are for the same hour/month.
    min_days: int
        Minimum number of days needed to show data in one month/hour

    Returns
    -------
    ds: xr.Dataset
        Seasonal-Daily dataset with dimensions month and hour instead of time

    Examples
    --------
    >>> ds
    <xarray.Dataset>
    Dimensions:  (lat: 1, lon: 1, lev: 1, time: 52405)
    Coordinates:
      * lat      (lat) float64 34.75
      * lon      (lon) float64 32.64
      * lev      (lev) float64 30.0
      * time     (time) datetime64[ns] 2005-06-19 ... 2006-06-19T23:30:00
    Data variables:
        M        (time, lev, lat, lon) float64 5.457 5.711 ... 5.162 5.108
        Dir      (time, lev, lat, lon) float64 315.2 313.1 ... 349.8 350.0
        SD       (time, lev, lat, lon) float64 0.413 0.4305 ... 0.4246 0.4227
    >>> seasonal_daily_cycle(ds['M'])
    <xarray.Dataset>
    Dimensions:  (month: 12, hour: 24, lev: 1, lat: 1, lon: 1)
    Coordinates:
      * month    (month) int32 1 2 3 4 5 6 7 8 9 10 11 12
      * hour     (hour) int32 0 1 2 3 4 5 6 7 8 9 ... 16 17 18 19 20 21 22 23
      * lev      (lev) float64 30.0
      * lat      (lat) float64 34.75
      * lon      (lon) float64 32.64
    Data variables:
        M        (month, hour, lev, lat, lon) float64 4.995 5.021 ... 4.138 4.125
    >>> seasonal_daily_cycle(ds, vars_list=['M', 'TI', 'SD', 'power'], normalized=True)
    <xarray.Dataset>
    Dimensions:  (month: 12, hour: 24, lat: 1, lon: 1, lev: 1)
    Coordinates:
      * month    (month) int32 1 2 3 4 5 6 7 8 9 10 11 12
      * hour     (hour) int32 0 1 2 3 4 5 6 7 8 9 ... 16 17 18 19 20 21 22 23
      * lat      (lat) float64 34.75
      * lon      (lon) float64 32.64
      * lev      (lev) float64 30.0
    Data variables:
        M        (month, hour, lat, lon, lev) float64 6.963 ... -11.37 -11.65
        TI       (month, hour, lat, lon, lev) float64 -6.168 ... -14.02 -18.81
        SD       (month, hour, lat, lon, lev) float64 -11.07 ... -26.18
        power    (month, hour, lat, lon, lev) float64 21.04 ... -13.83 -20.78
    """
    if vars_list is not None:
        vd = get_dataset(vd, vars_list=vars_list)
    df = vd.reset_coords(drop=True).to_dataframe()
    month = df.index.get_level_values('time').month.rename('month')
    hour = df.index.get_level_values('time').hour.rename('hour')
    other_dims = [x for x in vd.dims if x != 'time']

    count_vals = df.groupby([month, hour, *other_dims]).count()
    count = count_vals[count_vals.columns[0]].rename('count')

    freq = deduce_frequency(vd.coords['time'].values[:200])
    count_good = count > min_days * (60 // freq)

    if stat is None or stat == 'mean':
        stat = 'mean'
        dfg = df.groupby([month, hour, *other_dims]).mean()
    elif stat == 'median':
        dfg = df.groupby([month, hour, *other_dims]).median()
    elif stat == 'max':
        dfg = df.groupby([month, hour, *other_dims]).max()
    elif stat == 'min':
        dfg = df.groupby([month, hour, *other_dims]).min()
    elif stat == 'std':
        dfg = df.groupby([month, hour, *other_dims]).std()
    elif stat == 'count':
        dfg = df.groupby([month, hour, *other_dims]).count()
    elif 'q' in stat:
        quantile = float(stat.replace('q', ''))
        dfg = df.groupby([month, hour, *other_dims]).quantile(quantile)
    else:
        raise ValueError('Unknown stats to do seasonal/daily cycle')

    good_data = dfg.where(count_good)
    dsmh = xr.merge([good_data.to_xarray(), count.to_xarray()])

    # Take out circular variables if statistics are not mean/median
    drop_vars = [v for v in ['Dir'] if v in dsmh]
    if stat not in ['mean', 'median']:
        dsmh = dsmh.drop_vars(drop_vars)

    if normalized:
        # Take out variables that cannot be normalized by the mean
        # (because the 0 value is arbitrary)
        drop_vars_norm = [v for v in ['Dir', 'U', 'V'] if v in dsmh]
        mean_ds = dsmh.drop_vars(drop_vars_norm).mean(dim=['month', 'hour'])
        dsmh = 100 * (dsmh / mean_ds - 1)

    return dsmh

def ramps_ds(ds: xr.Dataset, vars_list: List[str] = None) -> xr.Dataset:

    if len(ds.coords['time']) == 0:
        raise ValueError('No times!')

    if vars_list is None:
        vars_list = [x for x in ds.data_vars]
        if 'U' in vars_list and 'V' in vars_list and 'Dir' not in vars_list:
            vars_list.append('Dir')

    t0 = ds.coords['time'].values[0]
    tL = ds.coords['time'].values[-1]
    freq = deduce_frequency(ds.coords['time'].values)
    if freq >= 3600:
        raise ValueError('Frequency larger than 1 day. Cannot compute ramps')

    trange = pd.period_range(start=t0, end=tL, freq=str(freq) + 'min')
    trange_ref = [np.datetime64(str(x)).astype('datetime64[ns]')
                  for x in trange]
    ds_rind = ds.reindex(time=trange_ref)
    diffs = ds_rind.diff(dim='time').dropna('time')

    name_map = {str(k): str(k) + '-ramp' for k in diffs.data_vars}
    diffs = diffs.rename(name_map)

    if 'Dir-ramp' in diffs:
        dirdiff = diffs['Dir-ramp']
        dirdiff = dirdiff.where(dirdiff <= 180, dirdiff - 360)
        dirdiff = dirdiff.where(dirdiff >= -180, dirdiff + 360)
        diffs['Dir-ramp'] = dirdiff

    return diffs


def noise_ds(ds: xr.Dataset, vars_list: List[str] = None) -> xr.Dataset:

    if len(ds.coords['time']) == 0:
        raise ValueError('No times!')

    if vars_list is None:
        vars_list = [x for x in ds.data_vars]
        if 'U' in vars_list and 'V' in vars_list and 'Dir' not in vars_list:
            vars_list.append('Dir')

    t0 = ds.coords['time'].values[0]
    tL = ds.coords['time'].values[-1]
    freq = deduce_frequency(ds.coords['time'].values)
    if freq >= 3600:
        raise ValueError('Frequency larger than 1 day. Cannot compute noise')

    trange = pd.period_range(start=t0, end=tL, freq=str(freq) + 'min')
    trange_ref = [np.datetime64(str(x)).astype('datetime64[ns]')
                  for x in trange]
    ds_rind = ds.reindex(time=trange_ref)

    noise = compute_noise(ds_rind, fillna=False)

    name_map = {str(k): str(k) + '-noise' for k in noise.data_vars}
    noise = noise.rename(name_map)

    return noise

def compute_noise(ds: Union[xr.DataArray, xr.Dataset],
                  fillna: Union[str, float] = None,
                  timesin: int = 3, timesout: int = 12,
                  ) -> Union[xr.DataArray, xr.Dataset]:
    """
    Compute the noise of a timeseries at highest frequency.

    Computed using Marta's arbitrary definition. We do not check the
    data is actually 10 minutal, so you could use it for noise at
    other scales.

    Parameters
    ----------
    ds: Union[xr.DataArray, xr.Dataset]
    fillna: str ('mean') or float
        Value to use to fill in Nans. By default, not filled
    timesin: int
        By default it is 3. The raw noise is the absolute difference
        between the timeseries and the mean of the rolled timeseries
        every `timesin` timestamps.
    timsesout: int
        By default it is 12. We do the rolling mean of raw_noise
        every `timesout` timestamps to have the smooth_noise value.

    Returns
    -------
    smooth_noise: Union[xr.DataArray, xr.Dataset]
        Same object as the input, with the "noise" values
    """
    raw_noise = np.abs(ds - ds.rolling(time=timesin,
                                       min_periods=timesin - 1,
                                       center=True).mean())
    smooth_noise = raw_noise.rolling(time=timesout,
                                     min_periods=timesout // 2,
                                     center=True).mean()

    if fillna is not None:
        if fillna == 'mean':
            fillna = smooth_noise.mean()
        smooth_noise = smooth_noise.fillna(fillna)

    return smooth_noise


def compare_ramps(sources: Dict[str, Union[xr.Dataset, xr.DataArray]],
                  name_ref: str = 'ref', do_cycle: bool = False,
                  info: Dict[str, Any] = None, view=False,
                  groups=None, **kwargs) -> Union[None, pd.DataFrame]:

    if 'M-ramp' in sources[name_ref] or 'SD-ramp' in sources[name_ref] \
            or 'Dir-ramp' in sources[name_ref]:
        sources_ramps = sources
        if groups is not None:
            raise ValueError('Cannot compare ramps by groups if you '
                             'pass directly the ramps dataset to '
                             'compare_ramps. Please pass timeseries '
                             'sources.')
    else:
        try:
            sources_ramps = {s: ramps_ds(ds) for s, ds in sources.items()}
        except ValueError:
            return None

    sources_ramps = time_match_sources(sources_ramps)

    if info is None:
        info = {}

    stats_list = []

    for var in sources_ramps[name_ref].data_vars:
        ref = sources_ramps[name_ref][var]

        if len(ref) < 10:
            print('Not enough timestamps to validate ' + var)
            continue

        if 'ramp' not in var:
            # We cannot validate a variable that is not a ramp!
            continue

        for source in sources_ramps:
            if source == name_ref:
                continue
            new = sources_ramps[source][var]
            base_var = str(var).replace('-ramp', '')

            st = compare_stats(ref, new, base_var)
            infoh = {**info, 'source': source, 'var': str(var), **st}
            stats_list.append(infoh)

    df = pd.DataFrame(stats_list)

    if do_cycle:
        df_seas = compare_seasonal_daily_cycle(sources_ramps, view=view,
                                               name_ref=name_ref,
                                               info=info, vars_list=None,
                                               normalized='all', **kwargs)
        df = pd.concat([df, df_seas])

    return df


def time_match_sources(sources):
    sources = {s: ds.dropna(dim='time', how='all')
               for s, ds in sources.items()}

    common_times = find_time_matching(sources.values())

    sources_tm = {s: ds.sel(time=common_times)
                  for s, ds in sources.items()}
    return sources_tm


def find_time_matching(list_xarrays, skip_duplicates=True):
    """
    Given a list of xarray DataArrays return the array of common timestamps.
    In case no duplicated entries are admitted (maybe because the input data
    have DAYLIGHT SAVING inconsistencies) one can set skip_duplicates=True.
    :param list_xarrays: list of xarray DataArrays or DataSets or Coordinates.
    They need to have a variable/coordinate called 'time' with the same
    kind of objects across arrays (default are numpy datetime64).
    :param skip_duplicates: bool. If True it first removes any duplicates
    (keeps first entries) of the input xarray dataarrays.
    :return: numpy array of common times (same object type as input)
    """
    times = []
    for x in list_xarrays:
        if skip_duplicates:
            _, index = np.unique(x['time'], return_index=True)
            x = x.isel(time=index)
        t = x['time']
        times.append(t.values)
    common_t = functools.reduce(np.intersect1d, times)
    return np.array(common_t)


def skip_duplicates(dataset, silent=True):
    initial_t = len(dataset.coords['time'])
    _, index = np.unique(dataset['time'], return_index=True)
    reduced = dataset.isel(time=index)
    final_t = len(reduced.coords['time'])
    skipped = initial_t - final_t
    if not silent:
        if skipped > 0:
            print('Skipped ' + str(skipped) + ' duplicated entries.')
    return reduced

def compare_noise(sources: Dict[str, Union[xr.Dataset, xr.DataArray]],
                  name_ref: str = 'ref', do_cycle: bool = False,
                  info: Dict[str, Any] = None, 
                  **kwargs) -> Union[None, pd.DataFrame]:

    if 'M-noise' in sources[name_ref] or 'SD-noise' in sources[name_ref] \
            or 'Dir-noise' in sources[name_ref]:
        sources_noise = sources
        if groups is not None:
            raise ValueError('Cannot compare noise by groups if you '
                             'pass directly the noise dataset to '
                             'compare_noise. Please pass timeseries '
                             'sources.')
    else:
        try:
            sources_noise = {s: noise_ds(ds) for s, ds in sources.items()}
        except ValueError:
            return None

    sources_noise = time_match_sources(sources_noise)

    if info is None:
        info = {}

    stats_list = []

    for var in sources_noise[name_ref].data_vars:
        ref = sources_noise[name_ref][var]

        if 'noise' not in var:
            # We cannot validate a variable that is not a noise!
            continue

        for source in sources_noise:
            if source == name_ref:
                continue

            new = sources_noise[source][var]
            base_var = str(var).replace('-noise', '')
            st = compare_stats(ref, new, base_var, **kwargs)
            infoh = {**info, 'source': source, 'var': str(var), **st}
            stats_list.append(infoh)

    df = pd.DataFrame(stats_list)

    if do_cycle:
        df_seas = compare_seasonal_daily_cycle(sources_noise, view=view,
                                               name_ref=name_ref,
                                               info=info, vars_list=None,
                                               normalized='all', **kwargs)
        df = pd.concat([df, df_seas])

    return df


def apply_general(func: Callable, va: vArray, *args,
                  **kwargs) -> vArray:
    """
    Apply a customized function to a vArray

    It doesn't matter if it is a xr.DataArray or a pd.Series, it will
    return the same object that was passed. It works for dask xarray
    objects too.

    It is not the recommended method! If numpy has a function to do it,
    better use it! For example, np.mean() or np.sqrt() will accept
    pandas and xarray objects and return them in the correct form too.

    Parameters
    ----------
    func: Callable
    va: vArray
        pd.Series or xr.DataArray to which we apply `func`
    args: list of objects
        Other positional arguments passed to `func`
    kwargs: dict of str: objects
        Other keyword arguments passed to `func`

    Returns
    -------
    vArray
        Result of applying the function
    """
    if isinstance(va, xr.DataArray):
        result_va = xr.apply_ufunc(func, va, *args, dask='allowed',
                                   kwargs=kwargs)
    elif isinstance(va, pd.Series):
        # we apply the function to the series' values
        # and then create a pd.Series with those values
        result_values = func(va.values, *args, **kwargs)
        result_va = pd.Series(result_values, index=va.index)
    else:
        raise ValueError('Not a vArray! Cannot apply func.')
    return result_va

def stats_in_resampling(ref: xr.DataArray,
                        new: xr.DataArray,
                        var: str, freq_name: str,
                        num_ts_in_one_hour=None) -> Dict[str, float]:
    """
    Resample a time series (not very carefully) and perform basic
    statistics.

    This is called from `compare_stats` and used if we pass the booleans
    for hourly, daily and monthly. These aggregated time series
    validations can be done much more extensively using the method
    `coarser_frequency_validation` from `vortexpy.compare_datasets`.

    Parameters
    ----------
    ref: xr.DataArray
    new: xr.DataArray
    var: str
        We need to know the var to know if it is aggregated by doing
        the mean (wind speed, SD,...) or the sum (energy)
    freq_name: str
        One of ['hourly', 'daily', 'monthly', 'yearly]
    num_ts_in_one_hour: float or None
        Number of timestamps in one hour (6 if 10min, 1 if 1H)

    Returns
    -------
    Dict[str, float]
        Dictionary of results with ['length', 'r2', 'rmse', 'MAE', 'EMD']
    """
    # Convert form freq_name to the pandas keyword
    keywords = {
        'hourly': '1H',
        'daily': '1D',
        'monthly': 'MS',
        'yearly': 'AS',
    }
    keyword = keywords[freq_name]

    if num_ts_in_one_hour is None:
        min_num = None
    else:
        min_num = get_min_values_period(keyword,
                                        num_ts_in_one_hour)

    results = {}
    if var in vars_aggregated_using_sum:
        func = np.sum
    else:
        func = np.mean

    # stupid function apply_for_valid_obs is needed because the great
    # min_num parameter which is available for np.sum is not
    # implemented for mean... which is absurd. Maybe new versions
    # of pandas/xarray will have this :)
    mref = ref.resample(time=keyword).map(apply_for_valid_obs,
                                          args=(func, min_num)).squeeze()
    mnew = new.resample(time=keyword).map(apply_for_valid_obs,
                                          args=(func, min_num)).squeeze()

    mref, mnew = mref.dropna(dim=mref.dims[0]), mnew.dropna(dim=mref.dims[0])
    results['length_' + freq_name] = len(mnew)
    if len(mref) > 2:
        mr2 = mnew.to_series().corr(mref.to_series()) ** 2
        results['r2_' + freq_name] = float(mr2)
    mbias = mnew - mref
    mrmse = np.sqrt((mbias ** 2).mean())
    results['rmse_' + freq_name] = float(mrmse)
    merror = np.abs(mbias)
    results['MAE_' + freq_name] = float(merror.mean())
    if var in vars_aggregated_using_sum:
        results['SAE_' + freq_name] = float(merror.sum())

    if len(mref) > 5:
        emd = wasserstein_distance(mref, mnew)
        results['EMD_' + freq_name] = float(emd)
    return results

def apply_for_valid_obs(x, func, min_num):
    valid_obs = x.notnull().count()
    value_da = func(x)
    if valid_obs < min_num:
        value_da.values = np.nan
    return value_da

def deduce_frequency(times: np.ndarray) -> int:
    """
    Deduce the frequency of a passed array of times (in minute).

    It takes the most frequent time interval in minutes between
    consecutive timestamps.

    Parameters
    ----------
    times: np.ndarray
        Array of time objects

    Returns
    -------
    int
        Frequency in minutes
    """
    intervals = np.array([j - i for i, j in zip(times[: -1], times[1:])])
    intervals_minutes = [np.timedelta64(x, 'm') for x in intervals]
    freq = mode(intervals_minutes)
    return int(freq / np.timedelta64(1, 'm'))

def get_min_values_period(keyword, num_ts_in_one_hour):
    min_nums = {
        '1H': int(0.75 * num_ts_in_one_hour),
        '6H': int(0.75 * 6 * num_ts_in_one_hour),
        '1D': int(0.90 * 24 * num_ts_in_one_hour),
        'W': int(0.90 * 7 * 24 * num_ts_in_one_hour),
        'MS': int(0.90 * 29 * 24 * num_ts_in_one_hour),
        'AS': int(0.90 * 365 * 24 * num_ts_in_one_hour),
    }
    min_num = min_nums.get(keyword, None)
    return min_num

def add_all_deduced_stats(stats_df: pd.DataFrame) -> pd.DataFrame:

    all_stats = stats_df.copy()
    mean = all_stats['mean'] - all_stats['bias']

    for v in ['bias', 'rmse', 'rmse_hourly', 'rmse_daily', 
              'rmse_monthly', 'weighted_rmse', 'weighted_MAE',
              'MAE', 'MAE_hourly', 'bias_sum', 'MAE_daily', 
              'MAE_monthly', 'bias_at_15', 'bias_q0.02', 
              'bias_q0.25', 'bias_q0.5', 'bias_q0.75', 'bias_q0.98']:
        if v in all_stats:
            all_stats[v + ' (%)'] = 100 * all_stats[v].astype('float64').div(mean)

    for v in ['bias_sum', 'SAE', 'SAE_hourly', 'SAE_daily', 'SAE_monthly']:
        if v in all_stats:
            sum = all_stats['sum']
            all_stats[v + ' (%)'] = 100 * all_stats[v].div(sum)

    if 'std' in all_stats:
        std_ref = all_stats['std'] - all_stats['std_bias']
        all_stats['cv'] = all_stats['std'].astype('float64').div(mean)
        cv_ref = std_ref.astype('float64').div(mean)
        all_stats['cv_bias'] = all_stats['cv'] - cv_ref

    if 'q0.75' in all_stats and 'q0.25' in all_stats:
        all_stats['iqr'] = all_stats['q0.75'] - all_stats['q0.25']
    if 'bias_q0.75' in all_stats and 'bias_q0.25' in all_stats:
        all_stats['bias_iqr'] = all_stats['bias_q0.75'] - \
                                all_stats['bias_q0.25']

    if 'q0.98' in all_stats and 'q0.02' in all_stats:
        all_stats['range'] = all_stats['q0.98'] - all_stats['q0.02']
    if 'bias_q0.98' in all_stats and 'bias_q0.02' in all_stats:
        all_stats['bias_range'] = all_stats['bias_q0.98'] - \
                                  all_stats['bias_q0.02']

    all_stats['rmse_norm_mean'] = all_stats['rmse'].div(mean)
    if 'range' in all_stats:
        range = all_stats['range']
        all_stats['rmse_norm_range'] = all_stats['rmse'].div(range)
    if 'iqr' in all_stats:
        iqr = all_stats['iqr']
        all_stats['rmse_norm_iqr'] = all_stats['rmse'].div(iqr)

    for v in ['bias', 'bias (%)', 'bias_max', 'bias_at_15 (%)', 'std_bias',
              'cv_bias', 'bias_sum', 'bias_sum (%)', 'bias_iqr', 'bias_range',
              'bias_q0.02', 'bias_q0.25', 'bias_q0.5', 'bias_q0.75', 'bias_q0.98',
              'bias_q0.02 (%)', 'bias_q0.25 (%)', 'bias_q0.5 (%)', 
              'bias_q0.75 (%)', 'bias_q0.98 (%)']:
        if v in all_stats:
            all_stats['absolute ' + v] = (all_stats[v]).abs()

    all_stats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return all_stats

compare_extra_functions = {
    'weibull': compare_weibulls,
    'tabfile': compare_tabfiles,
    'season_day': compare_seasonal_daily_cycle,
    'ramps': compare_ramps,
    'noise': compare_noise,
}

# Some variables must be aggregated using the mean
vars_aggregated_using_mean = ['M', 'variance', 'T', 'P', 'D',
                              'inflow', 'solar_elev', 'stability',
                              'SST', 'RH', 'PBLH', 'HFX']

# Some variables must be aggregated using the sum
vars_aggregated_using_sum = ['energy', 'total_veer']


def find_zonal_wind(vs: vSet) -> vArray:
    """
    Calculate the zonal wind component (U).

    Given a vSet we return the vArray
    of zonal wind, which may be already on the ``vSet``
    or it may need to be obtained from the wind speed and
    direction. It is computed lazily if the inputs are Dask arrays.

    Parameters
    ----------
    vs: vSet
        vSet with zonal wind speed (U) or wind speed and
         direction (M & Dir)

    Returns
    -------
    vArray
        Zonal wind speed data (named U)
    """
    if 'U' in vs:
        u = vs['U']
    else:
        try:
            m = vs['M']
            d = vs['Dir']
        except KeyError:
            raise ValueError('Cannot obtain U (no Dir or M)')

        u = -m * np.sin(d * math.pi / 180.)
        u = u.rename('U')

        u.attrs = attributes_vars['U']
    return u


def find_meridional_wind(vs: vSet) -> vArray:
    """
    Calculate the meridional wind component (V).

    Given a vSet we return the vArray
    of meridional wind, which may be already on the ``vSet``
    or it may need to be obtained from the wind speed and
    direction. It is computed lazily if the inputs are Dask arrays.

    Parameters
    ----------
    vs: vSet
        vSet with meridional wind speed (V) or wind speed and
         direction (M & Dir)

    Returns
    -------
    vArray
        Meridional wind speed data (named V)
    """
    if 'V' in vs:
        v = vs['V']
    else:
        try:
            m = vs['M']
            d = vs['Dir']
        except KeyError:
            raise ValueError('Cannot obtain V (no Dir or M)')

        v = -m * np.cos(d * math.pi / 180.)
        v = v.rename('V')

        v.attrs = attributes_vars['V']
    return v


def find_wind_speed(vs: vSet) -> vArray:
    """
    Calculate the wind speed.

    Given a vSet we return the vArray
    of wind speed, which may be already on the ``vSet``
    or it may need to be obtained from the wind components.
    It is computed lazily if the inputs are Dask arrays.

    Parameters
    ----------
    vs: vSet
        vSet with wind speed (M) or wind components (U & V)

    Returns
    -------
    vArray
        Wind speed data (named M)
    """
    if 'M' in vs:
        m = vs['M']
    else:
        try:
            u = vs['U']
            v = vs['V']
        except KeyError:
            raise ValueError('Cannot obtain M (no U or V)')

        m = np.sqrt(u ** 2 + v ** 2).rename('M')

        m.attrs = attributes_vars['M']
    return m


def find_direction(vs: vSet) -> vArray:
    """
    Calculate the wind direction.

    Given a vSet we return the vArray
    of wind direction, which may be already on the ``vSet``
    or it may need to be obtained from the wind components.
    It is computed lazily if the inputs are Dask arrays.

    Parameters
    ----------
    vs: vSet
        vSet with wind direction (Dir) or wind components (U & V)

    Returns
    -------
    vArray:
        Wind direction data (named Dir)

    """
    if 'Dir' in vs:
        d = vs['Dir']
    else:
        try:
            u = vs['U']
            v = vs['V']
        except KeyError:
            raise ValueError('Cannot obtain Dir (no U or V)')

        radians = np.arctan2(u, v)
        d = (radians * 180 / math.pi + 180).rename('Dir')

        d.attrs = attributes_vars['Dir']
    return d

def power_curve_function(wind_ref: Union[str, np.array] = 'vestas',
                         power_ref: Union[str, np.array] = 'vestas',
                         kind: str = 'cubic',
                         fill_value: float = 0.0,
                         bounds_error: bool = False) -> Callable:
    """
    Returns a function that converts from wind to power.

    It interpolates a given discrete ideal wind/power curve.

    Parameters
    ----------
    wind_ref: Union[str, np.array]
        Wind speeds to use as reference, or keyword.
    power_ref: Union[str, np.array]
        Wind power to use as reference, or keyword.
    kind: str
        Type of scipy interpolation. By default, 'cubic'
    fill_value: float
        By default is zero to take into account that turbines produce
        power zero for very low and very high winds.
    bounds_error: bool
        Argument passed to scipy interpolate.

    Returns
    -------
    Callable
    """

    if wind_ref is None or power_ref is None:
        wind_ref, power_ref = 'vestas', 'vestas'

    references = {
        'vestas': {
            'm': [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5,
                  10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15,
                  15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20,
                  20.5, 21, 21.5, 22, 22.5, 50],
            'p': [78, 172, 287, 426, 601, 814, 1068, 1367, 1717, 2110,
                  2546, 3002, 3428, 3773, 4012, 4131, 4186, 4198, 4200,
                  4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200,
                  4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200,
                  4200, 4200, 4200, 4200]
        }
    }

    if power_ref in references:
        power_ref = references[power_ref]['p']
    if wind_ref in references:
        wind_ref = references[wind_ref]['m']

    if len(wind_ref) != len(power_ref):
        raise ValueError('Mismatched variable vs power to '
                         'build the power curve.')

    power_f = interpolate.interp1d(wind_ref, power_ref, kind=kind,
                                   fill_value=fill_value,
                                   bounds_error=bounds_error)
    return power_f

def find_power(vs: vSet, wind_ref=None, power_ref=None,
               **kwargs) -> vArray:
    """Compute the power with respect to a power curve

    Find the power (kW) associated to each value of the wind,
    with respect to a given power curve.

    We need to pass a power curve in the form of two arrays or keywords
    (like vestas, for example): ``wind_ref`` are the wind values and
    ``power_ref`` the power for each of the reference wind values.

    Outside the range given by the reference values, power is assumed
    to be zero.

    Parameters
    ----------
    vs: vSet
        vSet with wind speed (M)
    wind_ref: List, np.array, dataarray, reference name (string)
        Reference values of wind
    power_ref: List, np.array, dataarray, reference name (string)
        Power corresponding to the ``wind_ref`` reference values. Must
        be of the same length as ``wind_ref``.
    kwargs: dict
        other arguments (passed to power_curve_function and then to
        scipy's function interpolate.interp1d)

    Returns
    -------
    xr.DataArray
    """

    if 'power' in vs:
        power = vs['power']
    else:
        m = find_wind_speed(vs)

        # Power Function
        power_f = power_curve_function(wind_ref=wind_ref,
                                       power_ref=power_ref,
                                       **kwargs)

        # Compute time series of power in kW
        power = apply_general(power_f, m).rename('power')
        power.attrs = attributes_vars['power']

    return power


def find_energy(vs: vSet, freq_min: int = None, **kwargs) -> vArray:
    """
    Compute the energy with respect to a power curve

    Find the energy (kW*hour) associated to each value of the wind,
    with respect to a given power curve, taking into account the
    frequency of the timeseries.

    We need to pass kwargs for the power curve (see find_power).

    Parameters
    ----------
    vs: vSet
        vSet with wind speed (M)
    freq_min: int
        Frequency in minutes of the timeseries (to convert from power
        to energy units)

    Returns
    -------
    xr.DataArray
    """

    if 'energy' in vs:
        energy = vs['energy']
    else:
        # Power in KiloWatts kW
        power = find_power(vs, **kwargs)

        if freq_min is None:
            try:
                times = get_times(vs)
                freq_min = deduce_frequency(times[:40])
            except Exception as e:
                raise ValueError('Cannot deduce frequency of data; '
                                 'the time variable is not clear'
                                 ).with_traceback(e.__traceback__)

        length_timestep_hours = freq_min / 60

        # Energy in kWh
        energy = (power * length_timestep_hours).rename('energy')
        energy.attrs = attributes_vars['energy']

    return energy

def find_wind_bins(wind_speed: vData) -> vArray:
    """
    Calculate the bin time series corresponding to a wind speed time
    series.

    Given a vData we return the vArray of the bin
    integer label, for the selected number of bins.
    It is computed lazily if the inputs are Dask arrays.

    Parameters
    ----------
    wind_speed: vData
        Wind speed vArray or vSet from which wind speed can be
        extracted or computed.

    Returns
    -------
    vArray
        Bin number (named bin)
    """
    if not hasattr(wind_speed, 'to_dataset') or \
            not hasattr(wind_speed, 'to_dataframe'):
        # If it could not be converted to a vSet, it means it is
        # a vSet. So, the wind_speed vArray has to be obtained:
        wind_speed = find_wind_speed(wind_speed)

    Mbin = apply_general(compute_bins_array, wind_speed,
                         width_bins=1.,
                         at_zero='bin_center'
                         ).rename('Mbin')

    Mbin.attrs = attributes_vars['Mbin']

    return Mbin


def find_sectors(direction: vData, num_sectors: int = 16) -> vArray:
    """
    Calculate the sector time series corresponding to
    a direction time series.

    Given a vData we return the vArray of the sector integer label,
    for the selected number of sectors. It is computed lazily if the
    inputs are Dask arrays.

    Parameters
    ----------
    direction: vData
        Direction vArray or vSet from which direction can be
        extracted or computed.
    num_sectors: int
        Number of sectors that we want (0 is the northernmost label,
        and they are labeled clockwise until num_sectors-1).

    Returns
    -------
    vArray
        Sector number (named sector)
    """

    if not hasattr(direction, 'to_dataset') or \
            not hasattr(direction, 'to_dataframe'):
        # If it could not be converted to a vSet, it means it is
        # a vSet. So, the direction vArray has to be obtained:
        direction = find_direction(direction)

    sectors = apply_general(compute_sectors_array, direction,
                            num_sectors=num_sectors,
                            ).astype('int8').rename('sector')
    sectors.attrs = attributes_vars['sector']

    return sectors


def expected_bin_edges(max_val: float, width_bins: float = 1.,
                       at_zero='bin_edge') -> np.array:
    """
    Given some key parameters for bin generation return the edge bins.

    The edge bins define the boundaries that separate one bin/sector
    from the next.

    Then the edges can be used by ``np.digitize`` to bin/sectorize an array.
    See compute_bins_array and compute_sectors_array.

    Parameters
    ----------
    max_val: float
        Highest value to be included in a bin
    width_bins: float
        Width of the bins
    at_zero: str, 'bin_edge' or 'bin_center'
        See inline comments

    Returns
    -------
    np.array
        Array of float values containing the edges of the bins

    Examples
    --------
    >>> expected_bin_edges(max_val=5.)
    array([0., 1., 2., 3., 4., 5., 6.])
    >>> expected_bin_edges(max_val=5., at_zero='bin_center')
    array([-0.5,  0.5,  1.5,  2.5,  3.5,  4.5,  5.5])
    >>> expected_bin_edges(max_val=8.43, width_bins=2., at_zero='bin_edge')
    array([ 0.,  2.,  4.,  6.,  8., 10.])

    """
    num_bins = int(max_val / width_bins) + 1
    # There are two options
    if at_zero == 'bin_edge':
        # The first bin starts for wind == 0 and covers until width_bins
        # ie, the center of the bin is width_bins / 2
        # This means that width_bins = 1 corresponds to bins
        # 0.5: (0, 1), 1.5: (1, 2), ...
        edge_bins = [width_bins * j for j in range(num_bins + 1)]
    elif at_zero == 'bin_center':
        # The first bin is centered at wind == 0 and covers until the
        # edge located at width_bins / 2
        # This means that width_bins = 1 corresponds to bins
        # 0: (-0.5, 0.5), 1: (0.5, 1.), ...
        # which looks like rounding
        edge_bins = [width_bins * j - width_bins / 2
                     for j in range(num_bins + 1)]
    else:
        raise ValueError('Unknown bin method at_zero=' + str(at_zero))

    return np.array(edge_bins)


def compute_bins_array(array: np.ndarray,
                       width_bins: float = 1.,
                       at_zero='bin_edge') -> np.ndarray:
    """
    Return the array with binned values.

    Given a time series of a certain continuous variable,
    define how we want to bin it (width of the bins and
    whether at zero we expect a bin edge or a bin center)
    and then obtain the same array that we passed but with the
    values being the bin center that corresponds to each value.

    The bin edges are found using `expected_bin_edges`.

    Parameters
    ----------
    array: np.array
    width_bins: float
    at_zero: str, 'bin_edge' or 'bin_center'
        See inline comments

    Returns
    -------
    np.array
        Same shape as input array, with each value replaced by the
        value that represents the center of the bin it belongs to.

    Examples
    --------
    >>> m = np.array([2.5, 2.3, 1.3, 1.8, 3.9, 0.4, 1.1])
    >>> compute_bins_array(m)
    array([2.5, 2.5, 1.5, 1.5, 3.5, 0.5, 1.5])
    >>> compute_bins_array(m, at_zero='bin_center')
    array([3., 2., 1., 2., 4., 0., 1.])

    """
    maxval = float(np.nanmax(array))
    edge_bins = expected_bin_edges(maxval,
                                   width_bins=width_bins,
                                   at_zero=at_zero)
    internal_edges = edge_bins[1:]

    # Time series of the INDEX of the bin each value belongs to
    ind_bins = np.digitize(array, internal_edges, right=False)

    # Convert from index of bin to "center value" of the bin
    center_bins = width_bins * ind_bins
    if at_zero == 'bin_edge':
        center_bins = center_bins + width_bins / 2

    return center_bins


def compute_sectors_array(direction: np.ndarray,
                          num_sectors: int = 16,
                          wrap_value: float = 360.) -> np.ndarray:
    """
    Return the array with the sector of each value.

    Given a time series of a certain continuous variable,
    define how we want to bin it (number of equally sized pieces wrapped
    circularly, in the direction sectors style) and then obtain the same
    array that we passed but with the values being the sector that
    corresponds to each value.

    The sector edges are found using `expected_bin_edges`.

    Parameters
    ----------
    direction: np.array
    num_sectors: int
    wrap_value: float

    Returns
    -------
    np.array
        Same shape as input array, with each value replaced by the
        value that represents the number of the sector it belongs to.

    Examples
    --------
    >>> _dir = [0., 5.0, 87.0, 56.0, 279.0, 182.0, 340., 355.]
    >>> compute_sectors_array(np.array(_dir))
    array([ 0,  0,  4,  2, 12,  8, 15,  0])
    >>> compute_sectors_array(np.array(_dir), num_sectors=12)
    array([ 0,  0,  3,  2,  9,  6, 11,  0])
    >>> compute_sectors_array(np.array(_dir), num_sectors=4)
    array([0, 0, 1, 1, 3, 2, 0, 0])

    """
    # Find sector edges
    width_sector = wrap_value / num_sectors
    edge_bins = expected_bin_edges(wrap_value,
                                   width_bins=width_sector,
                                   at_zero='bin_center')
    # take out last bin, because the largest values will become sector 0
    internal_edges = edge_bins[1:-1]

    # Time series of INDEX of bin (which is the sector)
    dbins = np.digitize(direction, internal_edges, right=False)

    # Correct last bin (==num_sectors) to bin 0th
    dbins = np.where(dbins < num_sectors, dbins, 0)
    return dbins

def weibull_ds(vs: vSet,
               num_sectors: int = None,
               for_wrg: bool = False,
               method: str = 'vtx_operative',
               with_stats: bool = False) -> xr.Dataset:
    """
    Obtain a weibull xr.Dataset from a vSet.

    There are 3 extra possibilities:

    * num_sectors:

    if we give a number of sectors, it will also compute the weibull
    parameters at each sector

    * method:

    specify the methodology for the weibull fit
        * scipy: native Pyhton; quite slow; relies on timeseries
        * params: very simple; very fast; accurate?; relies on mean and std
        * vtx_operative: translated from R; fast; relies on distribution

    If you want THE EXACT same dataset as the weibull.R would generate
    you can use the function vortex_weibull_ds.

    >>> print('For example, num_sectors=None:')
        <xarray.Dataset>
        Dimensions:   (lev: 1, lat: 1, lon: 1)
        Coordinates:
          * lev       (lev) int32 30
          * lat       (lat) float64 34.75
          * lon       (lon) float64 32.64
        Data variables:
            mean      (lev, lat, lon) float64 4.864
            variance  (lev, lat, lon) float64 6.186
            k         (lev, lat, lon) float64 2.051
            A         (lev, lat, lon) float64 5.492
            count     (lev, lat, lon) int64 117419


    >>> print('And if we want sectors (num_sectors=16):')
        <xarray.Dataset>
        Dimensions:          (lev: 1, lat: 1, lon: 1, sector: 16)
        Coordinates:
          * lev              (lev) int32 30
          * lat              (lat) float64 34.75
          * lon              (lon) float64 32.64
          * sector           (sector) float64 0.0 22.5 ...  315.0 337.5
        Data variables:
            mean             (lev, lat, lon) float64 4.864
            variance         (lev, lat, lon) float64 6.186
            k                (lev, lat, lon) float64 2.051
            A                (lev, lat, lon) float64 5.492
            count            (lev, lat, lon) int64 117419
            mean_sector      (sector, lev, lat, lon) float64 3.302  ...  4.219
            variance_sector  (sector, lev, lat, lon) float64 3.288 ... 3.711
            k_sector         (sector, lev, lat, lon) float64 1.926 ... 2.309
            A_sector         (sector, lev, lat, lon) float64 3.73 ... 4.761
            count_sector     (sector, lev, lat, lon) int64 6335 6302 ... 8303


    Parameters
    ----------
    vs: vSet
        vSet from which we can find the weibull parameters
    num_sectors: int
        number of sectors of the distribution
    method: str
        keyword that identifies the methodology used to do the Weibull fit
    with_stats: bool
        whether to add variables that inform about the adjustment of
        the histogram and the weibull fit

    Returns
    -------
    xr.Dataset
        Weibull dataset
    """
    vars_list = ['M']
    if num_sectors is not None:
        vars_list.append('Dir')
    if for_wrg:
        vars_list.append('HGT')

    ds = get_dataset(vs, vars_list=vars_list, strict=False)

    wb_ds = xrapply_weibull(ds, method=method, with_stats=with_stats)

    if num_sectors is not None:
        if 'Dir' not in ds:
            raise ValueError('No direction! Cannot do Weibull by sectors')
        all_ds = apply_by_sectors(ds, xrapply_weibull,
                                  num_sectors=num_sectors,
                                  method=method,
                                  with_stats=with_stats)
        all_ds = all_ds.rename_vars({v: v + '_sector'
                                     for v in all_ds.data_vars.keys()})
        wb_ds = xr.merge([wb_ds, all_ds])

    return wb_ds

def xrapply_weibull(wind_darray: Union[xr.DataArray, xr.Dataset],
                    core_dim: str = 'time',
                    method: str = 'vtx_operative',
                    with_stats: bool = False,
                    **kwargs) -> xr.Dataset:
    """
    Use weibull_fit_timeseries on a xarray Dataset of
    arbitrary dimensions.

    >>> print("The output weibull dataset looks like:")
        <xarray.Dataset>
        Dimensions:   (lev: 1, lat: 1, lon: 1)
        Coordinates:
          * lev       (lev) int32 30
          * lat       (lat) float64 34.75
          * lon       (lon) float64 32.64
        Data variables:
            mean      (lev, lat, lon) float64 4.864
            variance  (lev, lat, lon) float64 6.186
            k         (lev, lat, lon) float64 2.051
            A         (lev, lat, lon) float64 5.492
            count     (lev, lat, lon) int64 117419

    Parameters
    ----------
    wind_darray: Union[xr.DataArray, xr.Dataset]
        Wind data array or dataset where we can find the wind variable (M)
    core_dim: str
        Dimension over which to collapse the series to compute
        the Weibull fit. By default, time.
    method: str
        Method used to compute Weibull
    with_stats: bool
        whether to add variables that inform about the adjustment of
        the histogram and the weibull fit
    kwargs: Dict
        Arguments passed onto weibull_fit_timeseries

    Returns
    -------
    weibull_ds: xr.Dataset
        Dataset with mean, variance, k, A and count variables
    """
    # in case we pass the full dataset, we keep the variable M (wind)
    if isinstance(wind_darray, xr.Dataset):
        wind_darray = find_var('M', wind_darray)

    # we apply the weibull_fit_timeseries with the correct xarray options
    x = xr.apply_ufunc(weibull_fit_timeseries, wind_darray,
                       input_core_dims=[[core_dim]],
                       output_core_dims=[[], [], [], []],
                       vectorize=True,
                       kwargs={'output_as_dict': False,
                               'method': method,
                               **kwargs})

    # merge the output 4 standard weibull variables in a dataset
    wb_ds = xr.merge([x[0].rename('mean'),
                      x[1].rename('variance'),
                      x[2].rename('k'),
                      x[3].rename('A'),
                      ])

    # add the count variable
    wb_ds['count'] = wind_darray.count(dim=core_dim)

    if with_stats:
        weibstats = xr.apply_ufunc(weibull_fit_check_quality, wind_darray,
                                   wb_ds['A'], wb_ds['k'],
                                   input_core_dims=[[core_dim], [], []],
                                   output_core_dims=[[], [], [], [], []],
                                   vectorize=True,
                                   kwargs={'output_as_dict': False,
                                           **kwargs})

        # merge the statistics variables to the weibull dataset
        names = ['fit_mbias', 'fit_rmse', 'fit_MAE', 'fit_emd', 'fit_r2']
        for i, n in enumerate(names):
            wb_ds[n] = weibstats[i]

    return wb_ds


def scipy_weibull_fit(wind_array):
    """
    Fit a Weibull to a "wind timeseries" using python default method.

    NOT an exact reproduction of R's weibull_param function, basically
    because we compute it from the TIME SERIES instead of from the
    wind histogram.

    Information:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponweib.html

    exponweib is a python function with parameters:
        * a (exponentiation parameter),
        * c (shape of the non-exp Weibull law),
        * loc (shift parameter),
        * scale (scale parameter)

    To obtain the wind weibull: a=1 and loc=0 always
    The other parameters are usually called k (shape) and A (scale)

    Here we do a fit fixing a (f0) and loc (floc), and setting
    the initial value of shape k to 1.5 and of scale A to the mean.

    It the fit fails, we return NANs.

    Parameters
    ----------
    wind_array: np.ndarray
        A vector that stores the wind speed values for several
        timestamps (ie, the values of a data array with time dimension).

    Returns
    -------
    params: Tuple(float, float)
        Parameters shape (k) and scale (A)
    """

    wind_array = wind_array[~np.isnan(wind_array)]
    mean = float(np.mean(wind_array))
    try:
        f0, k, floc, A = stats.exponweib.fit(wind_array, 1, 1.5,
                                             floc=0, f0=1, scale=mean)
    except Exception:
        k, A = np.nan, np.nan

    return k, A

def weibull_fit_timeseries(wind_array: np.ndarray,
                           output_as_dict: bool = False,
                           method: str = 'scipy') -> \
        Union[Tuple[float, float, float, float], Dict]:
    """
    Fit a Weibull to a "wind timeseries" using python default method.

    NOT an exact reproduction of R's weibull_param function, basically
    because we compute it from the TIME SERIES instead of from the
    wind histogram.

    Parameters
    ----------
    wind_array: np.ndarray
        A vector that stores the wind speed values for several
        timestamps (ie, the values of a data array with time dimension).

    output_as_dict: bool, optional, default False
        Whether to return the parameters in a dictionary with
        the name of the metric instead of a list.

    method: str
        Method we want to use to fit the Weibull parameters.
        * 'scipy' -> use scipy.stats.exponweib.fit on the timeseries
        * 'params' -> naif estimation of A and k parameters
        * 'vtx_operative' -> pass to histogram and then do R's method

    Returns
    -------
    params: Tuple(float, float, float, float) or Dict
        Just like in R, the function returns 4 float values:
        mean, variance, shape, scale which are
        * the mean of the timeseries
        * the variance of the timeseries
        * the best-fit Weibull shape parameter (alpha, k)
        * the best-fit Weibull scale parameter (mu, A)
        If output_as_dict=True, it is returned as a dictionary with
        keys mean, variance, shape, scale.

    """

    wind_array = wind_array[~np.isnan(wind_array)]

    if len(wind_array) < 10:
        return np.nan, np.nan, np.nan, np.nan

    mean = float(np.mean(wind_array))
    variance = float(np.var(wind_array))

    if method == 'scipy':
        k_estimate, A_estimate = scipy_weibull_fit(wind_array)
    elif method == 'params':
        std = np.sqrt(variance)
        k_estimate = float((std / mean) ** -1.086)
        A_estimate = float(mean / gamma(1 + 1 / k_estimate))
    elif method == 'vtx_operative':
        max_wind = float(np.max(wind_array))
        bin_edges = expected_bin_edges(max_val=max_wind,
                                       width_bins=1.,
                                       at_zero='bin_edge')
        wind_events, edges = np.histogram(wind_array, bins=bin_edges)
        bin_centers = np.array([(edges[i] + edges[i+1])/2
                                for i in range(len(edges) - 1)])
        windbins_estimation = reproduce_r_weibull_param(wind_events,
                                                        bin_centers,
                                                        output_as_dict=False)
        # We OVERWRITE MEAN AND VARIANCE to obtain them from
        # the distribution, like the R vortex method would do
        mean, variance, k_estimate, A_estimate = windbins_estimation
    else:
        raise ValueError('Unidentified method!')

    if output_as_dict:
        names = ['mean', 'variance', 'shape', 'scale']
        vals = [mean, variance, k_estimate, A_estimate]
        return {names[i]: vals[i] for i in range(4)}
    else:
        return mean, variance, k_estimate, A_estimate


def reproduce_r_weibull_param(wind_events: np.ndarray,
                              bins: np.ndarray = None,
                              output_as_dict: bool = False) -> \
        Union[Tuple[float, float, float, float], Dict]:
    """
    Fit a Weibull to a "wind histogram" using weibull.R's method.

    Exact reproduction of R's weibull_param function.

    Parameters
    ----------
    wind_events: np.ndarray
        A vector that stores the number of events for certain
        wind speed bins. Considering the output of distribution_da,
        wind_events is the values of the data array when only
        the dimension 'speed' remains.

    bins: np.ndarray, optional
        The wind speed at the center of the bin for each bin population
        given in wind_events. By default, it is assumed that the wind bins
        have width 1 and have an edge at 0, ie, bins = [0.5, 1.5, ...]

    output_as_dict: bool, optional, default False
        Whether to return the parameters in a dictionary with
        the name of the metric instead of a list.

    Returns
    -------
    params: Tuple(float, float, float, float) or Dict
        Just like in R, the function returns 4 float values:
        mean, variance, shape, scale
        * in R, variables: ybar, s2,  k_estimate, A_estimate
        * in R, names: c("mean", "variance","alpha.hat","mu.hat")
        which are
        * the approx. mean (using the number of events at each bin)
        * the approx. variance of the variable
        * the best-fit Weibull shape parameter (alpha, k)
        * the best-fit Weibull scale parameter (mu, A)
        If output_as_dict=True, it is returned as a dictionary with
        keys mean, variance, shape, scale.

    """

    # default bins (width 1 and edge at zero)
    if bins is None:
        bins = np.arange(0.5, len(wind_events))

    # return Nans if there are no events
    sum_events = np.sum(wind_events)
    if sum_events > 0:
        # Approximate wind speed mean and variance from the distribution data
        ybar = np.average(bins, weights=wind_events)
        s2 = np.average((bins - ybar) ** 2, weights=wind_events)

        # Determine a good first estimation of the shape (alpha, k) parameter
        # Done like in R, no bibliography sorry
        temp = ybar ** 2 / (s2 + ybar ** 2)

        def alpha_trans_fn(al):
            return gamma(1 + 1 / al) * gamma(1 + 1 / al) / gamma(1 + 2 / al)

        def alpha_weibull_fn(y):
            tol = 0.001
            al_start = 0.0001
            al_end = 50
            al_mid = 0.5 * (al_start + al_end)
            y_tmp = alpha_trans_fn(al_mid)
            while abs(y_tmp - y) > tol:
                if y_tmp - y > 0:
                    al_end = al_mid
                else:
                    al_start = al_mid
                al_mid = 0.5 * (al_start + al_end)
                y_tmp = alpha_trans_fn(al_mid)
            return al_mid

        alpha_hat = alpha_weibull_fn(temp)

        # Determine a good first estimation of the scale (mu, A) parameter
        # using the shape estimation and the approximated wind speed mean
        # This is a standard formula to obtain the scale value
        mu_hat = ybar / gamma(1 + 1 / alpha_hat)
        first_guess = np.array([mu_hat, alpha_hat])

        # Starting at the first guess, find the weibull parameters
        # that minimize a certain cost related to the difference between
        # the real histogram (windbins) and the weibull fit
        def fcn(p):
            # exponweib is a python function with parameters
            # a (exponentiation parameter),
            # c (shape of the non-exp Weibull law),
            # loc (shift parameter),
            # scale (scale parameter)

            # To obtain the wind weibull: a=1 and loc=0 always
            # The other two parameters are usually called k (shape, also
            # called alpha in the R script) and A (scale, also mu in R)

            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponweib.html

            # Here we do the logpdf and a certain metric.
            # I just reproduced what the R script does.
            weib = stats.exponweib.logpdf(bins, 1, p[1], loc=0, scale=p[0])
            return -np.sum(wind_events * weib)

        estimate = optimize.minimize(fcn, first_guess,
                                     options={'maxiter': 10})
        A_estimate, k_estimate = estimate['x']
    else:
        ybar, s2, k_estimate, A_estimate = np.nan, np.nan, np.nan, np.nan

    if output_as_dict:
        names = ['mean', 'variance', 'shape', 'scale']
        vals = [ybar, s2, k_estimate, A_estimate]
        return {names[i]: vals[i] for i in range(4)}
    else:
        return ybar, s2, k_estimate, A_estimate



def weibull_fit_check_quality(wind_array: np.ndarray,
                              A: float,
                              k: float,
                              output_as_dict: bool = False,
                              ) -> \
        Union[Tuple[float, float, float, float, float], Dict]:
    """
    Find the stats that show if a Weibull accurately describes a
    "wind timeseries".

    Parameters
    ----------
    wind_array: np.ndarray
        A vector that stores the wind speed values for several
        timestamps (ie, the values of a data array with time dimension).

    A: float
        Weibull A parameter (scale) that corresponds to wind_array

    k: float
        Weibull k parameter (shape) that corresponds to wind_array

    output_as_dict: bool, optional, default False
        Whether to return the parameters in a dictionary with
        the name of the metric instead of a list.


    Returns
    -------
    params: Tuple(float, float, float, float, float) or Dict
        Metrics than compare the histogram of the wind timeseries
        using 0.5m/s bins and the corresponding Weibull fit
        * fit_mbias: wind speed bias (weibull_fit - timeseries)
        * fit_rmse: rmse for bin frequency between weibull_fit and histogram
        * fit_MAE: MAE for bin frequency between weibull_fit and histogram
        * fit_emd: distribution difference between weibull_fit and histogram
        * fit_r2: correlation difference for bin frequency between
        weibull_fit and histogram
        If output_as_dict=True, it is returned as a dictionary with
        keys fit_mbias, fit_rmse, fit_MAE, fit_emd, fit_r2.

    """

    wind_array = wind_array[~np.isnan(wind_array)]
    mean = float(np.mean(wind_array))

    maxval = float(np.nanmax(wind_array))
    bin_edges = expected_bin_edges(maxval,
                                   width_bins=0.2,
                                   at_zero='bin_edge')

    wind_density, bin_edges = np.histogram(wind_array,
                                           bins=bin_edges, density=True)

    bins_wind = [(bin_edges[i] + bin_edges[i + 1]) / 2
                 for i in range(len(bin_edges) - 1)]
    weibull_density = stats.exponweib.pdf(bins_wind, 1, k, scale=A, loc=0)
    weights_histogram_weibull = weibull_density * len(wind_array)
    mean_weibull = np.average(bins_wind, weights=weights_histogram_weibull)

    fit_mbias = mean_weibull - mean

    error = weibull_density - wind_density
    fit_rmse = np.sqrt(np.mean(error**2))
    fit_MAE = np.sqrt(np.mean(np.abs(error)))
    fit_r2 = pd.Series(wind_density, index=bins_wind).corr(
        pd.Series(weibull_density, index=bins_wind)) ** 2
    fit_emd = wasserstein_distance(weibull_density, wind_density)

    if output_as_dict:
        names = ['fit_mbias', 'fit_rmse', 'fit_MAE', 'fit_emd', 'fit_r2']
        vals = [fit_mbias, fit_rmse, fit_MAE, fit_emd, fit_r2]
        return {names[i]: vals[i] for i in range(4)}
    else:
        return fit_mbias, fit_rmse, fit_MAE, fit_emd, fit_r2

    
def find_var(var: str, ds: vSet, **kwargs) -> vArray:
    """
    Return the requested variable from the vSet if possible. Given a vSet
    we return the vArray of the variable `var` using the functions
    defined in this module or simply selecting it from the vSet.
    It is computed lazily if the inputs are Dask arrays.

    Parameters
    ----------
    var: str
        Name of the variable. Either existing in the vSet or
        one that is standard Vortex and can be computed:

        .. code-block:: python

            find_new_vars = {
                    # Some variables need to be computed
                    'U': find_zonal_wind,
                    'V': find_meridional_wind,
                    'M': find_wind_speed,
                    'Dir': find_direction,
                    'energy': find_energy,
                    'power': find_power,
                    'sector': find_sectors,
                    'Mbin': find_wind_bins,
                }

    ds: vSet

    Returns
    -------
    v: vArray
        Array called `var` and, in case it is a xarray object, with attributes.

    """

    if var in find_new_vars:
        v = find_new_vars[var](ds, **kwargs)
    elif var in ds:
        v = ds[var]
        if var in attributes_vars:
            v.attrs = attributes_vars[var]
    else:
        raise ValueError('Cannot obtain variable ' + var + ' from vSet.')

    return v


def get_dataset(vd: vData,
                vars_list: List[str] = None,
                strict: bool = True,
                no_zarr: bool = True) -> Union[xr.Dataset, None]:
    """
    Given a vData return the data in xr.Dataset format

    Sometimes it is useful to know what kind of objects are we dealing
    with instead of having the flexibility of vDatas.

    This function tries to smartly convert your vData to a xarray
    Dataset, and compute the requested variables.

    If the input is:
        - a xr.Datarray: simply convert to dataset
        - a pd.Series: convert to dataframe and then apply convert_to_xarray
        - a pd.DataFrame: add to the index lat, lon, lev, time if they were in the columns, and apply convert_to_xarray

    Try to find the variables of vars_list, and raise an error if
    none is found. If strict, fail if ANY variable is not found.

    If the vData passed was an vArray without name, and we request a
    single var, the code will assume the only var we passed is the one
    we want, and rename it to what we passed in vars_list.

    Parameters
    ----------
    vd: vData
    vars_list: list of variables
        Must be understood by find_var
    strict: bool
        If strict=True the function will fail if any variable
        is missing. If strict=False only fails if all variables fail.
    no_zarr: bool
        Compute the dask arrays if any, so that the result is not a
        dask object.

    Returns
    -------
    xr.Dataset
        The vData in xarray.Dataset format.
    """
    # Make sure we won't return a dask array (zarr)
    if no_zarr:
        if hasattr(vd, 'compute'):
            vd = vd.compute()

    # If we have a vArray, we just convert it to xr.Dataset
    if isinstance(vd, xr.DataArray):
        if vd.name == '' and len(vars_list) == 1:
            vd = vd.rename(vars_list[0])
        vd = vd.to_dataset()
    elif isinstance(vd, pd.Series):
        if vd.name == '' and len(vars_list) == 1:
            vd = vd.rename(vars_list[0])
        vd = convert_to_xarray(vd.to_dataframe())
    elif isinstance(vd, pd.DataFrame):
        newdims = [c for c in vd.columns
                   if c in ['lat', 'lon', 'lev', 'time']]
        coords = {c: np.unique([vd[c].values]) for c in vd.columns
                  if c in ['lat', 'lon', 'lev', 'time']}
        if 0 < len(newdims) < 4:
            vd = vd.set_index(newdims, append=True)
        elif len(newdims) == 4:
            vd = vd.set_index(newdims)

        vd = convert_to_xarray(vd, coords=coords)

    # If we get here, vd should be a xr.Dataset
    variables = []
    for v in vars_list:
        try:
            thisv = find_var(v, vd)
        except ValueError as e:
            if strict:
                print('One of the variables cannot be obtained: ' + v)
                raise e
        else:
            variables.append(thisv)

    if len(variables) == 0:
        return None

    full = xr.merge(variables, combine_attrs="drop")
    full = add_attrs_vars(full)
    full = add_attrs_coords(full)
    return full

def apply_by_sectors(ds, func, num_sectors=16, **kwargs):
    """
    Apply a function for each sector of a dataset

    Compute the sectors of a dataset and apply a certain func for
    the timestamps belonging for each sector. Afterwards concatenate the
    results so that the returned dataset has a 'sector' dimension.

    The function func must be a xarray function (given a dataset,
    returns a dataset).

    >>> print('For example:')
    all_ds = apply_by_sectors(ds, xrapply_weibull,
                              num_sectors=16,
                              method='vtx_operative')

    computes the weibull basic parameters on 16 sectors.

    Parameters
    ----------
    ds: xr.Dataset
    func: Callable
        Function that given one xr.Dataset returns one xr.Dataset
    num_sectors: int
        Number of sectors
    kwargs: Dict
        Passed as kwargs to func.

    Returns
    -------
    xr.Dataset
        With a new dimension 'sector' and coordinates the sector centers.
    """
    # timeseries of sectors
    sectors = find_var('sector', ds, num_sectors=num_sectors)

    # coordinates of the sectors: look for the sectors' centers
    width_sector = 360. / num_sectors
    edge_bins = expected_bin_edges(360.,
                                   width_bins=width_sector,
                                   at_zero='bin_center')
    center_sector = [(edge_bins[i] + edge_bins[i + 1]) / 2
                     for i in range(num_sectors)]

    # loop for each sector
    f_sectors = []
    for s in range(num_sectors):
        # part of the dataset that corresponds to this sector
        dss = ds.where(sectors == s)
        # apply the 'func' to this dataset (that has a lot of NaN in time)
        f_sector = func(dss, **kwargs)

        # assign the sector coordinate that corresponds
        if 'sector' not in f_sector.dims:
            f_sector = f_sector.expand_dims('sector')
        f_sector = f_sector.assign_coords({'sector': [center_sector[s]]})
        f_sectors.append(f_sector)
    # join the results of each sector in a single xr.Dataset
    all_sectors_ds = xr.concat(f_sectors, dim='sector', fill_value=np.nan)

    return all_sectors_ds


def add_attrs_vars(ds: xr.Dataset,
                   remove_existing_attrs: bool = False) -> xr.Dataset:
    """
    Add attributes information to variables from a dataset.

    If no `attributes_vars` dictionary is passed, the default
    attributes from the vars module are used.

    In xarray, a variable can have attributes :

    .. code-block:: python

        data['U'].attrs = {'description': 'Zonal Wind Speed',
                           'long_name'  : 'U wind speed',
                           'units'      : 'm/s'}}

    Parameters
    ----------
    ds : xarray.Dataset

    remove_existing_attrs : bool, False
        True will put only the attributes of `attributes_vars` and
        remove existing attributes, **including ENCODING details**.

    Returns
    -------
    xarray.Dataset
        Data with the new attributes
    """
    for var in ds.data_vars:
        if remove_existing_attrs:
            attributes = {}
        else:
            attributes = ds[var].attrs

        if var in attributes_vars:
            # noinspection PyTypeChecker
            for key, info in attributes_vars[var].items():
                attributes[key] = info

        ds[var].attrs = attributes

    return ds


def add_attrs_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Add attributes information to coordinates from a dataset.

    Used for lat, lon and lev.

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    xarray.Dataset
        Data with the new attributes for the coordinates
    """
    if 'lat' in ds:
        ds['lat'].attrs = {'units': 'degrees', 'long_name': 'Latitude'}
    if 'lon' in ds:
        ds['lon'].attrs = {'units': 'degrees', 'long_name': 'Longitude'}
    if 'lev' in ds:
        ds['lev'].attrs = {'units': 'metres', 'long_name': 'Level'}

    return ds


def fill_vertical_vars_at_lev(ds: xr.Dataset, lev: float,
                              vars_list: List[str],
                              step: float = 10) -> xr.Dataset:
    """
    Request a lev from a dataset and fill with vertical vars
    (RI, shear...) if possible (if there are enough levels above and below)

    You can set the step used for the bulk calculation of vertical vars.
    Using 10 replicates the operative behaviour.

    Parameters
    ----------
    ds: xr.Dataset
    lev: float
        Height already available in the dataset
    vars_list: List of str
        Vertical variables to compute: shear, RI, shear_sd, veer
    step: float, default 10
        Meters above and below of desired lev to compute bulk vars.

    Returns
    -------
    Dataset with single lev and vertical vars added (if possible) to the
    native vars passed in ds.
    """
    slice_lev = ds.sel(lev=lev)
    levs = ds.coords['lev'].values
    if levs[0] < lev - step and lev + step < levs[-1]:
        wind_vars_ds = get_dataset(ds, vars_list=['U', 'V',
                                                  'variance', 'T'],
                                   strict=False)
        mul_levs = wind_vars_ds.interp(lev=[lev - step, lev + step])
        new_vars = get_dataset(mul_levs, vars_list=vars_list)
        slice_lev = xr.merge([slice_lev, new_vars], compat='override')
    return slice_lev


find_new_vars = {
    # Some variables need to be computed
    'U': find_zonal_wind,
    'V': find_meridional_wind,
    'M': find_wind_speed,
    'Dir': find_direction,
    'energy': find_energy,
    'power': find_power,
    'sector': find_sectors,
    'Mbin': find_wind_bins,
}

attributes_vars = {
    'U': {'description': 'Zonal Wind Speed',
          'long_name': 'U wind speed',
          'units': 'm/s'},
    'V': {'description': 'Meridional Wind Speed Component',
          'long_name': 'V wind speed',
          'units': 'm/s'},
    'W': {'description': 'Vertical Wind Speed Component',
          'long_name': 'W wind speed',
          'units': 'm/s'},
    'M': {'description': 'Wind Speed (module velocity)',
          'long_name': 'Wind speed',
          'units': 'm/s'},
    'TI': {'long_name': 'Turbulence Intensity',
           'description': 'Turbulence Intensity',
           'units': '%'},
    'Dir': {'description': 'Wind Direction',
            'long_name': 'Wind direction',
            'units': 'degrees'},
    'SD': {'description': 'Wind Speed Standard Deviation',
           'long_name': 'Wind Speed Standard Deviation',
           'units': 'm/s'},
    'variance': {'description': 'Wind Speed Variance',
                 'long_name': 'Wind Speed Variance',
                 'units': 'm^2/s^2'},
    'T': {'description': 'Air Temperature',
          'long_name': 'Air Temperature',
          'units': 'Deg.Celsius'},
    'P': {'description': 'Pressure',
          'long_name': 'Pressure',
          'units': 'hPa'},
    'D': {'long_name': 'Density',
          'description': 'Air Density',
          'units': 'kg/m^(-3)'},
    'RMOL': {'description': 'Inverse Monin Obukhov Length',
             'long_name': 'Inverse Monin Obukhov Length',
             'units': 'm^-1'},
    'L': {'description': 'Monin Obukhov Length',
          'long_name': 'Monin Obukhov Length',
          'units': 'm'},
    'stability': {'description': 'Atmospheric Stability Index (RMOL)',
                  'long_name': 'Atmospheric Stability (idx)',
                  'units': ''},
    'stabilityClass': {'description': 'Atmospheric Stability Class (RMOL)',
                       'long_name': 'Atmospheric Stability (class)',
                       'units': ''},
    'HGT': {'description': 'Terrain Height (above sea level)',
            'long_name': 'Terrain Height',
            'units': 'm'},
    'inflow': {'long_name': 'Inflow angle',
               'description': 'Inflow angle',
               'units': 'degrees'},
    'RI': {'long_name': 'Richardson Number',
           'description': 'Richardson Number',
           'units': ''},
    'shear': {'long_name': 'Wind Shear Exponent',
              'description': 'Wind Shear Exponent',
              'units': ''},
    'shear_sd': {'long_name': 'Wind SD Shear',
                 'description': 'Wind SD Shear',
                 'units': ''},
    'veer': {'long_name': 'Wind Directional Bulk Veer',
             'description': 'Wind Directional Bulk Veer',
             'units': 'degrees m^-1'},
    'total_veer': {'long_name': 'Wind Directional TotalVeer',
                   'description': 'Wind Directional Total Veer',
                   'units': 'degrees m^-1'},
    'sector': {'long_name': 'Wind Direction Sector',
               'description': 'Wind Direction Sector',
               'units': ''},
    'Mbin': {'long_name': 'Wind Speed Bin',
             'description': 'Wind Speed Bin (round to nearest int)',
             'units': ''},
    'daynight': {'long_name': 'Day or Night',
                 'description': 'Day or Night',
                 'units': ''},
    'solar_elev': {'long_name': 'Solar Elevation',
                   'description': 'Solar Elevation Angle',
                   'units': 'degrees'},
    'power': {'long_name': 'Power',
              'description': 'Approximation to the power expected at '
                             'this instant (energy/time)',
              'units': 'kW'},
    'energy': {'long_name': 'Energy Production',
               'description': 'Approximation to the energy expected from '
                              'the power and time frequency of the series',
               'units': 'kWh'},
    'SST': {'long_name': 'Sea Surface Temperature',
            'description': 'Sea Surface Temperature',
            'units': 'K'},
    'HFX': {'long_name': 'Heat Flux Surface',
            'description': 'Upward heat flux at the surface',
            'units': 'W m-2'},
    'PBLH': {'long_name': 'Boundary Layer Height',
             'description': 'Boundary Layer Height',
             'units': 'm'},
    'RH': {'long_name': 'Relative Humidity',
           'description': 'Relative Humidity',
           'units': '%'},
    'TP': {'long_name': 'Potential Temperature',
           'description': 'Potential Temperature',
           'units': 'K'},
    'T2': {'long_name': 'Air Temperature at 2m',
           'description': 'Air Temperature at 2m',
           'units': 'K'},
    'TKE_PBL': {'long_name': 'Turbulent Kinetic Energy',
                'description': 'Turbulent Kinetic Energy',
                'units': 'm^2/s^2'}
}

def all_sources_stats(sources: Dict[str, xr.Dataset],
                      name_ref: str = 'ref',
                      vars_list: List[str] = None,
                      extra_validations: Dict[str, Dict[str, Any]] = None,
                      info: Dict[str, Any] = None,
                      silent=True,
                      **kwargs) -> pd.DataFrame:
    """
    Given a "sources" at only one height, validate the requested variables
    and extra validations (weibull, TI curve, spectra...).

    Parameters
    ----------
    sources: Dict[str, xr.Dataset]
        It must be a valid sources dictionary (datasets with same levs,
        coordinates and frequency)
    name_ref: str
        Name of the source to use as reference
    vars_list: List of str or None
        List of variables to validate. If not passed, we validate the
        variables found on the reference dataset.
    extra_validations: Dict
        Dictionary with keys that specify the extra validation to do
        and with values that indicate the kwargs we want to pass
        to the function that computes this extra validation.
        For example, `extra_validations={'weibull': {'num_sectors': 16}}`
    silent: bool
        Whether to supress all prints
    info: Dict
        Other columns to add in the statistics table (name of the site,
        identifier of the test, height...)
    kwargs
        Passed to `compare_stats`. See the options in the docs of that
        function.

    Returns
    -------
    df: pd.DataFrame
        Results of statistics
    """
    if info is None:
        info = {}

    others = [s for s in sources if s != name_ref]

    sources_tm = time_match_sources(sources)

    ref_ds = sources_tm[name_ref]
    if vars_list is not None:
        ref_ds = get_dataset(ref_ds, vars_list=vars_list, strict=False)
    vars_list = list(ref_ds.data_vars.keys())

    stats_list = []
    for var in vars_list:
        ref = find_var(var, ref_ds)

        if not silent:
            print('\t - Compare stats for ' + var)

        for source in others:
            new = find_var(var, sources_tm[source])
            if new is None:
                print('Cannot find variable ' + var + ' for ' + source)
                continue

            st = compare_stats(ref, new, var, **kwargs)
            info_here = {**info, 'source': source, 'var': var, **st}
            stats_list.append(info_here)

    df = pd.DataFrame(stats_list)

    if extra_validations is not None:
        for extra, extra_kwargs in extra_validations.items():
            if extra == 'windprofile':
                # Some extra can't be done for just one lev
                continue

            if not silent:
                print('\t>Validate Extra: ' + extra)
                if len(extra_kwargs) > 0:
                    print('\t with: ' + repr(extra_kwargs))

            compare_func = compare_extra_functions.get(extra, None)
            if compare_func is None:
                continue

            d_extra = compare_func(sources_tm, info=info,
                                   name_ref=name_ref,
                                   silent=silent,
                                   **extra_kwargs)

            if d_extra is not None:
                df = pd.concat([df, d_extra], ignore_index=True)

    return df