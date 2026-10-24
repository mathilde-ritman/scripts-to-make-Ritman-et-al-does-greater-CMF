import pathlib
import re
import logging
import datetime

import xarray as xr
import numpy as np
import pandas as pd
import scipy
import intake
import global3d_track as g3d

import analytical

# quantities

grid_spacing = 11000  # m
time_spacing = 900    # s

def _integral(da):
    result = da.sum("time") * time_spacing
    units = da.attrs.get("units", "").split(" s-1")[0]
    return result.assign_attrs(units=units)

def _threshold_area(da, dims, thresholds):
    threshold_da = xr.DataArray(thresholds, dims="threshold", coords={"threshold": thresholds})
    return ((da > threshold_da) * grid_spacing**2).sum(dims)

def _core_net_precip(da, dims, *params):
    level_dims = tuple(d for d in dims if d != "core")
    reduced = da.max(level_dims) if level_dims else da
    return reduced.sum("core")

def _histogram(da, dims, bins):
    def hist_1d(vals):
        vals = vals[np.isfinite(vals)]
        return np.histogram(vals, bins=bins)[0]
    result = xr.apply_ufunc(
        hist_1d,
        da,
        input_core_dims=[list(dims)],
        output_core_dims=[["lbins"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"lbins": len(bins) - 1}},
        output_dtypes=[int],
    )
    lbins = xr.DataArray(bins[:-1], dims="lbins", attrs=dict(long_name="ice water path histogram bins", units="kg m-2"))
    return result.assign_coords(lbins=lbins)

_SPATIAL_DIMS = ("level_full", "lat", "lon")
IWP_LOG_BINS = np.logspace(np.log10(1e-3), np.log10(1e2), 100)

iwp_pi = pd.read_csv('/home/b/b382635/s/my_notebooks/model_paper/grl_resubmission/scripts/iwp_percentiles.csv', index_col='percentile')
IWP_THRESHOLDS = iwp_pi.loc[np.arange(.1,1,.1).round(1),:].values.flatten()
# IWP_THRESHOLDS = np.array((1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-1, 2.5e-1, 5e-1, 7.5e-1, 1, 2.5, 5, 7.5, 10), dtype=float)

def _space_label(space_method):
    if space_method == "identity":
        return None
    if isinstance(space_method, tuple):
        kind, target = space_method
        if kind == "level_select":
            return f"spatial level-select ({target})"
        return f"spatial {kind}"
    names = {"threshold_area": "threshold area", "core_net_precip": "core net precip", "log_histogram": "log histogram"}
    return f"spatial {names.get(space_method, space_method)}"

def _time_label(time_method):
    if time_method in ("identity", None):
        return None
    if time_method == "lifecycle":
        return "lifecycle-normalised"
    names = {"integral": "integral"}
    return f"time {names.get(time_method, time_method)}"

def _add_aggregation_note(da, note):
    if not note:
        return da
    existing = da.attrs.get("aggregation")
    return da.assign_attrs(aggregation=f"{existing}, {note}" if existing else note)

def _attrs_with_override(da, **overrides):
    attrs = da.attrs.copy()
    attrs.update({k: v for k, v in overrides.items() if v is not None})
    return attrs

def apply_space_method(ds, source, space_method):
    da = ds[source]
    units_override = None
    long_name_override = None

    if isinstance(space_method, tuple):
        kind, target = space_method
        if kind != "level_select":
            raise ValueError(f"unknown space_method: {space_method}")
        target = ds[target] if isinstance(target, str) and target in ds.variables else target
        result = da.sel(level_full=target, method="nearest")

    elif space_method == "identity":
        result = da

    elif space_method == "threshold_area":
        dims = tuple(d for d in ("lat", "lon") if d in da.dims)
        result = _threshold_area(da, dims, IWP_THRESHOLDS)
        units_override = "m2"

    elif space_method == "core_net_precip":
        dims = tuple(d for d in ("level_full", "core") if d in da.dims)
        result = _core_net_precip(da, dims)

    elif space_method == "log_histogram":
        dims = tuple(d for d in ("lat", "lon") if d in da.dims)
        result = _histogram(da, dims, IWP_LOG_BINS)
        units_override = "count"
        long_name_override = f"{da.attrs.get('long_name', source)} histogram"

    elif space_method in ("max", "min", "mean", "sum"):
        dims = tuple(d for d in _SPATIAL_DIMS if d in da.dims)
        result = getattr(da, space_method)(dims, keep_attrs=True) if dims else da

    else:
        raise ValueError(f"unknown space_method: {space_method}")

    result = result.assign_attrs(_attrs_with_override(da, units=units_override, long_name=long_name_override))
    return _add_aggregation_note(result, _space_label(space_method))

def apply_time_method(da, time_method):
    dims = tuple(d for d in ("time","core") if d in da.dims)
    units_override = None

    if time_method in ("max", "min", "mean", "sum"):
        result = getattr(da, time_method)(dims, keep_attrs=True) if dims else da
    elif time_method == "integral":
        result = _integral(da).sum("core", keep_attrs=True) if "core" in da.dims else _integral(da)
        units_override = result.attrs.get("units")
    else:
        result = da

    result = result.assign_attrs(_attrs_with_override(da, units=units_override))
    return _add_aggregation_note(result, _time_label(time_method))

VARIABLE_OPERATIONS = {

    # identity
    "ABH": ("ABH", {"time_method": "identity", "space_method": "identity"}),
    "lifetime": ("lifetime", {"time_method": "identity", "space_method": "identity"}),
    "ncores": ("ncores", {"time_method": "identity", "space_method": "identity"}),
    "ocean_fraction": ("ocean_fraction", {"time_method": "identity", "space_method": "identity"}),
    "TOI": ("TOI", {"time_method": "identity", "space_method": "identity"}),
    "conv_lifetime": ("conv_lifetime", {"time_method": "identity", "space_method": "identity"}),
    "category": ("category", {"time_method": "identity", "space_method": "identity"}),
    
    # # ice water path histograms
    # "cloud_fwp_lhist": ("cloud_fwp", {"time_method": "sum", "space_method": "log_histogram"}),
    # "anvil_fwp_lhist": ("anvil_fwp", {"time_method": "sum", "space_method": "log_histogram"}),
    # "anvil_l400_fwp_lhist": ("anvil_l400_fwp", {"time_method": "sum", "space_method": "log_histogram"}),

    # anvil scalar results
    "max_cloud_area": ("cloud_area", {"time_method": "max", "space_method": "max"}),
    "max_anvil_area": ("anvil_area", {"time_method": "max", "space_method": "max"}),
    "max_thick_cloud": ("cloud_fwp", {"time_method": "max", "space_method": "threshold_area"}),
    "max_thick_area": ("anvil_fwp", {"time_method": "max", "space_method": "threshold_area"}),
    "total_anvil_area": ("anvil_area", {"time_method": "sum", "space_method": "max"}),
    "total_cloud_area": ("cloud_area", {"time_method": "sum", "space_method": "max"}),
    "total_thick_cloud": ("cloud_fwp", {"time_method": "sum", "space_method": "threshold_area"}),
    "total_thick_area": ("anvil_fwp", {"time_method": "sum", "space_method": "threshold_area"}),

    # convection metrics
    "total_CMT": ("core_cmt", {"time_method": "integral", "space_method": ("level_select", "ABH")}),
    "mean_CMT": ("core_cmt", {"time_method": "mean", "space_method": ("level_select", "ABH")}),
    "max_CMT": ("core_cmt", {"time_method": "max", "space_method": ("level_select", "ABH")}),
    "mean_CMF": ("core_cmf_mu", {"time_method": "mean", "space_method": ("level_select", "ABH")}),
    "max_CMF": ("core_cmf_max", {"time_method": "max", "space_method": ("level_select", "ABH")}),
    "max_Ac": ("core_area", {"time_method": "max", "space_method": ("level_select", "ABH")}),
    "max_w": ("core_amax_w", {"time_method": "max", "space_method": ("level_select", "ABH")}),
    "col_max_CMF": ("core_cmf_max", {"time_method": "max", "space_method": "max"}),
    "col_max_w": ("core_max_w", {"time_method": "max", "space_method": "max"}),
    "core_net_pr": ("core_pr", {"time_method": "integral", "space_method": "core_net_precip"}),
    "core_max_pr": ("core_pr", {"time_method": "max", "space_method": "max"}),
    "cloud_min_tt": ("cloud_tt", {"time_method": "min", "space_method": "min"}),

    # lifecycle results
    "core_cmt_ABH": ("core_cmt", {"time_method": "lifecycle", "space_method": ("level_select", "ABH")}),
    "core_area_ABH": ("core_area", {"time_method": "lifecycle", "space_method": ("level_select", "ABH")}),
    "core_rho_ABH": ("core_rho_mu", {"time_method": "lifecycle", "space_method": ("level_select", "ABH")}),
    "core_fwc_ABH": ("core_fwc", {"time_method": "lifecycle", "space_method": ("level_select", "ABH")}),
    "anvil_rho_mu": ("anvil_rho_mu", {"time_method": "lifecycle", "space_method": "mean"}),
    "anvil_fwc": ("anvil_fwc", {"time_method": "lifecycle", "space_method": "mean"}),
    "anvil_depth_mu": ("anvil_depth", {"time_method": "lifecycle", "space_method": "mean"}),
    "anvil_div_mu": ("anvil_div_mu", {"time_method": "lifecycle", "space_method": "mean"}),
    "anvil_div_max": ("anvil_div_max", {"time_method": "lifecycle", "space_method": "max"}),
    "anvil_area": ("anvil_area", {"time_method": "lifecycle", "space_method": "max"}),
    "cloud_area": ("cloud_area", {"time_method": "lifecycle", "space_method": "max"}),
    "cloud_depth": ("cloud_depth", {"time_method": "lifecycle", "space_method": "max"}),
    "thick_anvil_area": ("anvil_fwp", {"time_method": "lifecycle", "space_method": "threshold_area"}),
    "thick_cloud_area": ("cloud_fwp", {"time_method": "lifecycle", "space_method": "threshold_area"}),
    "anvil_tw": ("anvil_tw", {"time_method": "lifecycle", "space_method": "sum"}),

    # # investigate ABH(z)
    # "env_p": ("env_p", {"time_method": "identity", "space_method": "identity"}),
    # "env_zg": ("env_zg", {"time_method": "identity", "space_method": "identity"}),
    # "env_T": ("env_T", {"time_method": "identity", "space_method": "identity"}),
    # "mean_CMT_z": ("core_cmt", {"time_method": "mean", "space_method": "identity"}),
    # "max_CMT_z": ("core_cmt", {"time_method": "max", "space_method": "identity"}),
    # "mean_w_z": ("core_amax_w", {"time_method": "mean", "space_method": "identity"}),
    # "cloud_fwc": ("cloud_fwc", {"time_method": "mean", "space_method": "identity"}),
    # "cloud_fw": ("cloud_fw", {"time_method": "sum", "space_method": "identity"}),
    # "cloud_area_z": ("cloud_area", {"time_method": "sum", "space_method": "identity"}),

    # test ABH sensitvity
    # "cloud_cbh_mu": ("cloud_cbh_mu", {"time_method": "mean", "space_method": "identity"}),
    # "anvil_l400_ABH": ("anvil_l400_ABH", {"time_method": "identity", "space_method": "identity"}),
    # "total_CMT_l400": ("core_cmt", {"time_method": "integral", "space_method": ("level_select", "anvil_l400_ABH")}),
    # "total_anvil_area_l400": ("anvil_l400_area", {"time_method": "sum", "space_method": "max"}),
    # "total_thick_area_l400": ("anvil_l400_fwp", {"time_method": "sum", "space_method": "threshold_area"}),
    # "mean_CMT_l400": ("core_cmt", {"time_method": "mean", "space_method": ("level_select", "anvil_l400_ABH")}),
    # "max_CMT_l400": ("core_cmt", {"time_method": "max", "space_method": ("level_select", "anvil_l400_ABH")}),
    # "mean_CMF_l400": ("core_cmf_mu", {"time_method": "mean", "space_method": ("level_select", "anvil_l400_ABH")}),
    # "max_CMF_l400": ("core_cmf_max", {"time_method": "max", "space_method": ("level_select", "anvil_l400_ABH")}),
    # "max_Ac_l400": ("core_area", {"time_method": "max", "space_method": ("level_select", "anvil_l400_ABH")}),
    # "max_w_l400": ("core_amax_w", {"time_method": "max", "space_method": ("level_select", "anvil_l400_ABH")}),
}

df = pd.read_csv('/work/bb1153/b382635/data/final_tracks/dcc_frozen_1e4/tropics/track_stats/dcc_statistics.csv', parse_dates=['initial_time'], index_col='dcc')

nextgems_catalog = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
_data = nextgems_catalog.ICON.ngc4008(time="PT3H", zoom=9).to_dask().isel(time=0)
background = g3d.utils.regrid.Regrid('tropics').perform(_data[['ocean_fraction_surface']], zoom=9, resolution=0.1).drop_vars('time')

def general_results(ds):

    results = xr.Dataset()

    lifetime = (ds.time.isel(time=-1) - ds.time.isel(time=0)) / np.timedelta64(1, "h") + 0.25
    results["lifetime"] = lifetime.assign_attrs(long_name="Lifetime", units="hrs")

    ncores = 0
    if "core" in ds.coords or "core" in ds.dims:
        for core_id in ds.core.values:
            if ds.sel(core=core_id).core_volume.notnull().any():
                ncores += 1
    results["ncores"] = ncores
    results["ncores"] = results.ncores.assign_attrs(long_name="number of convective cores", units="unitless")

    if "core_volume" in ds:
        conv_times = ds.time.where((ds.core_volume > 0).max("core"), drop=True)
    else:
        conv_times = ds.time.isel(time=[])
    if conv_times.size > 0:
        results["conv_lifetime"] = (conv_times.isel(time=-1) - conv_times.isel(time=0)) / np.timedelta64(1, "h") + 0.25
    else:
        results["conv_lifetime"] = np.nan
    results["conv_lifetime"] = results.conv_lifetime.assign_attrs(long_name="convective lifetime", units="hrs")

    system_id = ds.system.item()
    initial_lat = df.loc[system_id, "initial_lat"]
    initial_lon = df.loc[system_id, "initial_lon"]
    results["initial_lat"] = initial_lat
    results["initial_lat"] = results.initial_lat.assign_attrs(long_name="Initial latitude", units="degrees North")
    results["initial_lon"] = initial_lon
    results["initial_lon"] = results.initial_lon.assign_attrs(long_name="Initial longitude", units="degrees East")

    results["ocean_fraction"] = background.ocean_fraction_surface.sel(
        lat=float(initial_lat), lon=float(initial_lon), method="nearest",
    ).assign_attrs(long_name="ocean fraction", units="unitless")

    lon_hours = float(initial_lon) / 15.0
    results["TOI"] = ds.time.isel(time=0) + np.timedelta64(int(lon_hours * 3600), "s")
    results["TOI"] = results.TOI.assign_attrs(long_name="time of initiaion", description="local solar time")

    organized = bool(df.loc[system_id, "valid_but_complex_convection"])
    isolated = bool(df.loc[system_id, "valid"])
    results["category"] = "isolated" if isolated else "organised" if organized else np.nan
    results["category"] = results.category.assign_attrs(
        long_name="category", units="unitless",
        description="If 'isolated' then there was no or minimal gap between the first and last convective cores "
                     "(i.e., the period of active convection was relatively constrained within the cloud lifecycle)",
    )

    return results

def calculate(ds):
    if "system" in ds.dims:
        ds = ds.squeeze("system")

    results = general_results(ds)

    for name, (source, methods) in VARIABLE_OPERATIONS.items():
        if source not in ds:
            continue
        results[name] = apply_space_method(ds, source, methods["space_method"])

    if "cloud_area" in results.data_vars:
        results = results.merge(analytical.make_area_predictions(results))
    return results

def temporally_aggregate_results(ds):

    results = xr.Dataset()
    for name, (source, methods) in VARIABLE_OPERATIONS.items():
        if methods["time_method"] == "lifecycle" or name not in ds:
            continue
        results[name] = apply_time_method(ds[name], methods["time_method"])

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

    if "core_cmt_ABH" in ds.data_vars:
        ds['core_cmt_ABH'] = reduce_core_dimension(ds.core_cmt_ABH, 'sum')
        ds['core_rho_ABH'] = reduce_core_dimension(ds.core_rho_ABH, 'mean')
        ds['core_fwc_ABH'] = reduce_core_dimension(ds.core_fwc_ABH, 'mean')
        ds['core_area_ABH'] = reduce_core_dimension(ds.core_area_ABH, 'sum')

    # choose variables
    lifecycle_vars = [name for name, (_, methods) in VARIABLE_OPERATIONS.items() if methods["time_method"] == "lifecycle"] + [x for x in ds.data_vars if 'Apred' in x]
    if len(lifecycle_vars) == 0:
        return xr.Dataset()
    ds_life = ds[lifecycle_vars]

    # define new time axis
    tstep = 0.025
    new_taxis = np.arange(0, 1 + tstep, tstep)

    # first interpolate the time series
    dsi = normalise_by_overall_lifetime(ds.cloud_area > 0, ds_life, bins=new_taxis).rename({"interp_time": "itime"})

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
    n_dev = int(N * tref_depth)
    n_mat = int(N * tref_area)

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
            input_core_dims=[["itime"]],
            output_core_dims=[["interp_time"]],
            kwargs={"t": data.itime.values, "new_t": new_t, "valid_t": valid_t},
            vectorize=True,
            dask='parallelized',
            dask_gufunc_kwargs={"output_sizes": {"interp_time": new_taxis.size}},
            output_dtypes=[float],
        )
        return result.assign_coords(interp_time=new_t)

    def interpolate_time(data, new_t):
        sidx = data.coords["system"].item()
        if "system" in data.dims:
            data = data.isel(system=0)
        valid_t = data.dropna("itime", how="all").itime.values
        valid_t = data.itime.isin(valid_t).values
        if isinstance(data, xr.Dataset):
            result = xr.Dataset(coords={"interp_time": new_t})
            for name, da in data.data_vars.items():
                result[name] = apply(da, new_t, valid_t)
            return result.expand_dims(system=[sidx])
        return apply(data, new_t, valid_t).expand_dims(system=[sidx]).to_dataset(name=data.name)

    def map_over_systems(ds, new_t):
        if "system" in ds.dims and ds.sizes["system"] > 1:
            return ds.groupby('system').map(interpolate_time, args=(new_t,))
        return interpolate_time(ds, new_t)

    dev_intp = map_over_systems(developing, new_taxis[0:n_dev + 1])
    mat_intp = map_over_systems(mature, new_taxis[n_dev:n_mat + 1])
    grow_intp = map_over_systems(growing, new_taxis[0:n_mat + 1])
    dec_intp = map_over_systems(decay, new_taxis[n_mat:])

    def ensure_system_dim(ds):
        if "system" in ds.dims:
            return ds
        return ds.expand_dims(system=[ds.coords["system"].item()])

    results = xr.concat([
        dev_intp.expand_dims(phase=['developing']),
        mat_intp.expand_dims(phase=['mature']),
        grow_intp.expand_dims(phase=['growing']),
        dec_intp.expand_dims(phase=['decay']),
        ensure_system_dim(dsi).expand_dims(phase=['overall']).rename({"itime": "interp_time"})
    ], dim='phase')

    # keep units, and flag these as lifecycle-normalised (spatial-method note
    # was already added to each var's attrs back in calculate())
    lifecycle_note = _time_label("lifecycle")
    for v in results.data_vars:
        if v in ds:
            results[v].attrs = _add_aggregation_note(ds[v], lifecycle_note).attrs

    # keep intersection data
    results = results.merge(xr.Dataset({'t_max_depth': t_max_depth.assign_attrs(long_name="time of maximum cloud depth (normalised time)", units="unitless"),
                                        't_max_area': t_max_area.assign_attrs(long_name="time of maximum cloud area (normalised time)", units="unitless")}))

    # flag phase validity
    presence = (results.cloud_area > 0).any("interp_time")
    valid_decay = presence.sel(phase="decay")
    valid_dev_and_mat = presence.sel(phase=["developing", "mature"]).all("phase")
    results["lifestages_valid"] = valid_decay & valid_dev_and_mat
    results["lifestages_valid"].attrs = dict(description="If True, the cloud lifecyle was sucessfully divided into all three lifestages with at least two data points in each.")

    results = results.rename({"thick_anvil_area": "thick_area"})
    return results

def results_for_analyses(ds):
    ''' Combine the temporal and lifecycle aggregations into the final analysis-ready dataset. '''
    return xr.merge(
        [
            temporally_aggregate_results(ds),
            cloud_lifecycle_results(ds),
        ],
        compat="override",
    )

def calculate_all(ds):
    return results_for_analyses(calculate(ds))
