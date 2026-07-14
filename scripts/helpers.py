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


def temporal_aggregate_results(ds):
    ''' Aggregate key study statistics in time. '''
    
    results = xr.Dataset({
        "lifetime": ds.lifetime,
        "ocean_fraction": ds.ocean_fraction,
        "category": ds.category,
        "total_thick_area": ds.thick_area.sum("time").assign_attrs(long_name="Total thick area", units="m^2"),
        "total_cloud_area": ds.cloud_area.sum("time").assign_attrs(long_name="Total area", units="m^2"),
        "total_CMT": (ds.core_cmt_ABH.sum(("time","core")) * time_spacings).assign_attrs(long_name="Total mass inflow", units="kg", equation=r"$\int M_c dt\,$"),
        "max_CMT": ds.core_cmt_ABH.max(("time","core")).assign_attrs(long_name="Maximum CMT", units="kg s-1", equation=r"$M_{c,max}$"),
        "max_CMF": ds.core_cmf_max.max(("time","core")).assign_attrs(long_name="Core maximum CMF", units="kg s-1 m-2", equation=r"$(\rho w)_{max}$"),
        "max_CMF_ABH": ds.core_cmf_max_ABH.max(("time","core")).assign_attrs(long_name="Core maximum CMF", units="kg s-1 m-2", equation=r"$(\rho w)_{max}$"),
        "max_w": ds.core_max_w.max(("time","core")).assign_attrs(long_name="Maximum updraft", units="m s-1", equation="$w$"),
        "max_w_ABH": ds.core_mean_w_ABH.max(("time","core")).assign_attrs(long_name="Maximum updraft", units="m s-1", equation="$w$"),
        "max_Ac": (ds.core_area_ABH.max(("time","core"))).assign_attrs(long_name="Maximum core area", units="m s-1", equation="$A_c$"),
        "conv_lifetime": ds.conv_lifetime.assign_attrs(long_name="Duration of convection", units="hrs", equation="$t_c$"),
    })

    for var in ds.data_vars:
        if "hist" in var:
            results[var] = ds[var].sum("time", keep_attrs=True)
    
    return results
    
def normalise_by_overall_lifetime(obj_exists, ds, bins=np.arange(0, 1.05, 0.05)):
    ''' Normalise dataset of clouds to a shared lifetime range [0,1] by the period in which 'obj_exists', i.e., the cloud exists. '''

    def normalise_vectorised(data, exists):
        data = data[exists]
        ntimes = data.size
        if ntimes < 2:
            return np.full((len(bins),), np.nan, dtype=float)
        time_percentage = np.linspace(0, 1, ntimes)
        return np.interp(bins, time_percentage, data)

    def apply(da, cloud_exists):
        result = xr.apply_ufunc(
            normalise_vectorised,
            da,
            cloud_exists,
            input_core_dims=[['time'], ['time']],
            output_core_dims=[['interp_time']],
            vectorize=True,
            dask='parallelized',
            dask_gufunc_kwargs={"output_sizes": {"interp_time": len(bins)}},
            output_dtypes=[float],
        )
        return result.assign_coords(interp_time=bins)
    
    result = xr.Dataset(attrs=ds.attrs)

    for name, da in ds.data_vars.items():
        if 'time' in da.dims:
            result[name] = apply(da, obj_exists)
        else:
            result[name] = da

    for name, coord in ds.coords.items():
        if 'time' not in coord.dims:
            result = result.assign_coords({name: coord})

    return result

def cloud_lifecycle_results(ds):
    ''' Normalise dataset of clouds to a shared evolution, comprising of developing, mature and decay phases. '''
    
    # aggreagte individiual cores that belong to the same cloud
    def reduce_core_dimension(da, func, dim='core'):
        return getattr(da, func)(dim, keep_attrs=True) if dim in da.dims else da

    ds['core_cmt_ABH'] = reduce_core_dimension(ds.core_cmt_ABH, 'sum')
    ds['core_rho_ABH'] = reduce_core_dimension(ds.core_rho_ABH, 'mean')
    ds['core_fwc_ABH'] = reduce_core_dimension(ds.core_fwc_ABH, 'mean')
    ds['core_area_ABH'] = reduce_core_dimension(ds.core_area_ABH, 'sum')
    
    # choose variables to normalise in time
    model_data = ['core_cmt_ABH','core_rho_ABH','core_fwc_ABH','core_area_ABH',]
    model_predictions = [x for x in ds.data_vars if 'Apred' in x]
    general_results = ['anvil_area', 'cloud_area', 'cloud_depth', 'thick_area', 'anvil_tw', 'anvil_div_max', 'anvil_depth_mu', 'anvil_rho', ]
    ds_life = ds[model_data+model_predictions+general_results]

    # define new time axis
    tstep = 0.025
    new_taxis = np.arange(0, 1+tstep, tstep)

    # first interpolate the time series
    dsi = normalise_by_overall_lifetime(ds.cloud_area > 0, ds_life, bins=new_taxis).rename({"interp_time":"itime"})
    
    # identify phase transition times
    t_max_depth = dsi.cloud_depth.idxmax('itime')
    t_max_area = dsi.cloud_area.idxmax('itime')
    
    # split data into lifecycle phases
    developing = dsi.where(dsi.itime <= t_max_depth)
    mature = dsi.where(np.logical_and(dsi.itime >= t_max_depth, dsi.itime <= t_max_area))
    growing = dsi.where(dsi.itime <= t_max_area)
    decay = dsi.where(dsi.itime >= t_max_area)

    # assign new standard duration of each phase (calculated as mean phase duration)
    tref_depth = 0.25 
    tref_area = 0.65
    N = new_taxis.size
    n_dev = int(N*tref_depth)
    n_mat = int(N*tref_area)
    
    # apply normalisation to each phase
    
    def from_one_trange_to_the_next(tin, tout):
        return (tin - tin.min()) / (tin.max() - tin.min()) * (tout.max() - tout.min()) + tout.min()
        
    def interpolate_time_vectorized(da, t, new_t, valid_t):
        new_t = np.asarray(new_t, dtype=float)
        y = da[valid_t]
        t = t[valid_t]
        mask = np.isfinite(y)
        
        if t[mask].size == 0:
            return np.full(new_t.shape, np.nan, dtype=float)
        if t[mask].size == 1:
            return np.full(new_t.shape, y[mask].squeeze(), dtype=float)
    
        func = scipy.interpolate.PchipInterpolator(t[mask], y[mask], extrapolate=False)
        imaginary_vals = from_one_trange_to_the_next(new_t, t)
    
        return func(imaginary_vals)
    
    def apply(data, new_t, valid_t):
        result = xr.apply_ufunc(
            interpolate_time_vectorized,
            data,
            input_core_dims=[["itime"],],
            output_core_dims=[["interp_time"]],
            kwargs={"t": data.itime.values, "new_t": new_t, "valid_t":valid_t},
            vectorize=True,
            dask='parallelized',
            dask_gufunc_kwargs={"output_sizes": {"interp_time": new_taxis.size}},
            output_dtypes=[float],
        )
        return result.assign_coords(interp_time=new_t)
    
    def interpolate_time(data, new_t):
        sidx = data.coords["system"].item()
        data = data.isel(system=0)
        valid_t = data.dropna("itime", how="all").itime.values
        valid_t = data.itime.isin(valid_t).values
        if isinstance(data, xr.Dataset):
            result = xr.Dataset(coords={"interp_time": new_t})
            for name, da in data.data_vars.items():
                result[name] = apply(da, new_t, valid_t)
            return result.expand_dims(system=[sidx])
        return apply(data, new_t, valid_t).expand_dims(system=[sidx]).to_dataset(name=data.name)
    
    dev_intp = developing.groupby('system').map(interpolate_time, args=(new_taxis[0:n_dev+1],))
    mat_intp = mature.groupby('system').map(interpolate_time, args=(new_taxis[n_dev:n_mat+1],))
    grow_intp = growing.groupby('system').map(interpolate_time, args=(new_taxis[0:n_mat+1],))
    dec_intp = decay.groupby('system').map(interpolate_time, args=(new_taxis[n_mat:],))
    
    results = xr.concat([
        dev_intp.expand_dims(phase=['developing']),
        mat_intp.expand_dims(phase=['mature']),
        grow_intp.expand_dims(phase=['growing']),
        dec_intp.expand_dims(phase=['decay']),
        dsi.expand_dims(phase=['overall']).rename({"itime":"interp_time"})
        ], dim='phase')

    # keep units
    for v in results.data_vars:
        results[v].attrs = ds[v].attrs.copy()

    # keep intersection data
    results = results.merge(xr.Dataset({'t_max_depth': t_max_depth.assign_attrs(long_name="time of maximum cloud depth (normalised time)", units="unitless"), 
                                        't_max_area': t_max_area.assign_attrs(long_name="time of maximum cloud area (normalised time)", units="unitless"),}))

    # flag phase validity
    presence = (results.cloud_area > 0).any("interp_time")
    valid_decay = presence.sel(phase="decay")
    valid_dev_and_mat = presence.sel(phase=["developing", "mature"]).all("phase")
    valid_all_three = valid_decay & valid_dev_and_mat
    results["lifestages_valid"] = valid_decay & valid_dev_and_mat
    results["lifestages_valid"].attrs = dict(description="If True, the cloud lifecyle was sucessfully divided into all three lifestages with at least two data points in each.")

    return results

def results_for_analyses(ds):
    results = xr.merge(
        [
            temporal_aggregate_results(ds),
            cloud_lifecycle_results(ds),
        ],
        compat="override",
    )
    return results

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

def evaluate_predictions(true, pred):
    # mask values
    x_ = true.values.ravel()
    y_ = pred.values.ravel()
    x = x_[np.isfinite(x_) & np.isfinite(y_)]
    y = y_[np.isfinite(x_) & np.isfinite(y_)]
    
    # caculate linear regression coefficient
    res = scipy.stats.linregress(x, y)
    xline = np.linspace(x.min(), x.max(), 100)
    yline = res.slope * xline + res.intercept
    slope = res.slope # slope of the regression line
    r = res.rvalue # Pearson correlation coefficient

    return x, y, xline, yline, slope, r

