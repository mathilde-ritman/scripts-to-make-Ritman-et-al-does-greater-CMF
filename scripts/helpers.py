import pandas as pd
import xarray as xr
import numpy as np
import global3d_track as g3d

import datetime
import scipy


'''

Various functions used to aid analyses.

'''

grid_spacings = 11000  # m
time_spacings = 900    # s

def dccs_by_intensity_quantile(ds, data, key):
    ''' Bin clouds by their intensity quantile for the metric 'key'. '''
    
    pi = data.quantile((0, 0.25, 0.5, 0.75, 1)).rename({"quantile": "pi"})
    labels = ["0-25", "25-50", "50-75", "75-100"]
    bins = pi.values
    ds2 = ds.assign_coords({key: ("system", data.data)})
    grouped = ds2.groupby_bins(key, bins=bins, labels=labels, include_lowest=True)
    return pd.DataFrame(
        {
            "metric": key,
            "percentile_range": [str(bucket) for bucket, _ in grouped],
            "systems": [group.system.values.tolist() for _, group in grouped],
        }
    ).set_index(["metric", "percentile_range"])

def dccs_by_intensity_metrics(ds, idata):
    ''' Bin clouds by their intensity quantile for the all metrics in 'idata.data_vars'. '''
    
    dfs = []
    for key in idata.data_vars:
        dfs.append(dccs_by_intensity_quantile(ds[["system"]], idata[key], key))
    return pd.concat(dfs).sort_index()

def group_dataset_by_intensity(ds, df, apply=lambda x: x):
    ''' Add a new dimension to the dataset that groups clouds by their intensity quartile for the metric in 'df'. '''
    
    out = {}
    available = np.asarray(ds.system.values)
    for metric in df.index.get_level_values(0).unique():
        pieces = []
        for prange, row in df.loc[metric].iterrows():
            ids = np.asarray(row["systems"])
            ids = ids[np.isin(ids, available)]
            part = ds.sel(system=ids).expand_dims(bins=[prange]).apply(apply)
            pieces.append(part)
        out[metric] = xr.concat(pieces, dim="bins")
    return out

def evaluate_predictions(true, pred, params={}):
    # mask values
    x_ = true.values.ravel()
    y_ = pred.values.ravel()
    x = x_[np.isfinite(x_) & np.isfinite(y_)]
    y = y_[np.isfinite(x_) & np.isfinite(y_)]
    
    # caculate linear regression coefficient
    result = scipy.stats.linregress(x, y, **params)
    xline = np.linspace(x.min(), x.max(), 100)
    yline = result.slope * xline + result.intercept
    return x, y, xline, yline, result

