import xarray as xr
import numpy as np
import pandas as pd
import intake
import global3d_track as g3d
import analytical

'''

Derive study statistics and model predictions from the tracked cloud dataset.

'''

# quantities

grid_spacing = 11000 # m
time_spacing = 900 # s

# load metadata and stats derived by initial cloud data filtering

df = pd.read_csv('/work/bb1153/b382635/data/final_tracks/dcc_frozen_1e4/tropics/track_stats/dcc_statistics.csv', parse_dates=['initial_time'], index_col='dcc')

# load background model data
nextgems_catalog = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
data = nextgems_catalog.ICON.ngc4008(time="PT3H", zoom=9).to_dask().isel(time=0)
background = g3d.utils.regrid.Regrid('tropics').perform(data[['ocean_fraction_surface']], zoom=9, resolution=0.1).drop_vars('time')

def general_results(ds):
    ''' Collect key cloud metadata. '''
    
    results = xr.Dataset()

    lifetime = (ds.time.isel(time=-1) - ds.time.isel(time=0)) / np.timedelta64(1, "h") + 0.25
    results["lifetime"] = lifetime.assign_attrs(long_name="Lifetime", units="hrs")

    ncores = 0
    if "core" in ds.coords or "core" in ds.dims:
        for core_id in ds.core.values:
            core_slice = ds.sel(core=core_id)
            if core_slice.core_volume.notnull().any():
                ncores += 1
    results["ncores"] = ncores
    results["ncores"] = results.ncores.assign_attrs(long_name="Number of convective cores", units="unitless")

    if "core_volume" in ds:
        conv_mask = (ds.core_volume > 0).max("core")
        conv_times = ds.time.where(conv_mask, drop=True)
        if conv_times.size > 0:
            results["conv_lifetime"] = (conv_times.isel(time=-1) - conv_times.isel(time=0)) / np.timedelta64(1, "h") + 0.25
        else:
            results["conv_lifetime"] = np.nan
    else:
        results["conv_lifetime"] = np.nan
    results["conv_lifetime"] = results.conv_lifetime.assign_attrs(long_name="Convective lifetime", units="hrs")

    initial_lat = ds["lat"].where(ds.cloud_depth.isel(time=0).max("lon") > 0, drop=True).mean()
    initial_lon = ds["lon"].where(ds.cloud_depth.isel(time=0).max("lat") > 0, drop=True).mean()
    results["initial_lat"] = initial_lat.assign_attrs(long_name="Initial latitude", units="degrees North")
    results["initial_lon"] = initial_lon.assign_attrs(long_name="Initial longitude", units="degrees East")

    results["ocean_fraction"] = background.ocean_fraction_surface.sel(
        lat=float(initial_lat),
        lon=float(initial_lon),
        method="nearest",
    ).assign_attrs(long_name="Ocean fraction", units="unitless")

    lon_hours = float(initial_lon) / 15.0
    results["TOI"] = ds.time.isel(time=0) + np.timedelta64(int(lon_hours * 3600), "s")
    results["TOI"] = results.TOI.assign_attrs(long_name="Time of initiaion", description="local solar time")

    def assign_region(lon, lat):
        if np.isnan(lon) or np.isnan(lat):
            return np.nan
        if -83 <= lon <= -43:
            return "amazon"
        if 1 <= lon <= 41:
            return "congo"
        if 90 <= lon <= 160:
            return "warm pool"
        if -180 <= lon <= 180:
            return "oceans"
        return np.nan

    results["region"] = assign_region(float(initial_lon), float(initial_lat))
    results["region"] = results.region.assign_attrs(long_name="Region", units="unitless")

    system_id = ds.system.item()
    organized = bool(df.loc[system_id, "valid_but_complex_convection"])
    isolated = bool(df.loc[system_id, "valid"])
    results["category"] = "isolated" if isolated else "organised" if organized else np.nan
    results["category"] = results.category.assign_attrs(long_name="Category", units="unitless")

    return results


def thick_area_results(ds):
    ''' Calculate the cloud area exceeding each IWP threshold. '''
    
    iwp_thresholds = np.array((1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-1, 2.5e-1, 5e-1, 7.5e-1, 1, 2.5, 5, 7.5, 10), dtype=float,)

    threshold_ds = xr.DataArray(
        iwp_thresholds,
        dims="threshold",
        coords={"threshold": iwp_thresholds},
        name="threshold",
    ).assign_attrs(units="kg m^-2", long_name="IWP threshold")

    area = ((ds["anvil_fwp"] > threshold_ds) * grid_spacing**2).sum(("lat", "lon"))
    area = area.assign_attrs(units="m^2", long_name="Thick area")
    return area.to_dataset(name="thick_area")


def anvil_base_height_results(ds):
    ''' Index the convective core data at the anvil base height. '''
    
    results = xr.Dataset()
    variables = ("core_cmt", "core_area", "core_fwc", "core_rho", "core_cmf_max", "core_mean_w")

    for var in variables:
        results[f"{var}_ABH"] = ds[var].sel(level_full=ds.ABH, method="nearest")

    results["ABH"] = ds.ABH
    return results


def spatial_aggregate_results(ds):
    ''' Aggregate (spatially) the cloud data. '''
    
    spatial_dims = ("level_full", "lat", "lon")
    results = xr.Dataset()

    for var in ("cth", "anvil_volume"):
        if var in ds:
            results[var] = ds[var]

    max_vars = ("cloud_area", "cloud_depth", "core_max_w", "core_cmf_max", "anvil_area", "core_pr")
    for var in max_vars:
        if var in ds:
            dims = tuple(dim for dim in spatial_dims if dim in ds[var].dims)
            da = ds[var].max(dims, keep_attrs=True)
            # da.attrs["long_name"] = var if not "long_name" in da.attrs else da.attrs.long_name
            results[var] = da.assign_attrs(long_name=da.attrs["long_name"] + "(spatial maximum)", units=da.attrs["units"])

    sum_vars = ("core_tw", "anvil_tw")
    for var in sum_vars:
        if var in ds:
            dims = tuple(dim for dim in spatial_dims if dim in ds[var].dims)
            da = ds[var].sum(dims, keep_attrs=True)
            results[var] = da.assign_attrs(long_name=da.attrs["long_name"] + "(spatial total)", units=da.attrs["units"])

    mean_vars = ("core_rho", "anvil_rho", "core_fwc", "anvil_fwc")
    for var in mean_vars:
        if var in ds:
            dims = tuple(dim for dim in spatial_dims if dim in ds[var].dims)
            da = ds[var].mean(dims, keep_attrs=True)
            results[var] = da.assign_attrs(long_name=da.attrs["long_name"] + "(spatial mean)", units=da.attrs["units"])

    da = ds.anvil_depth.mean(("lat", "lon"), keep_attrs=True)
    results["anvil_depth_mu"] = da.assign_attrs(long_name=da.attrs["long_name"] + "(spatial mean)", units=da.attrs["units"])

    da = ds.anvil_div.mean("level_full", keep_attrs=True)
    results["anvil_div_mu"] = da.assign_attrs(long_name=da.attrs["long_name"] + "(spatial mean)", units=da.attrs["units"])
    
    da = ds.anvil_div.max("level_full", keep_attrs=True)
    results["anvil_div_max"] = da.assign_attrs(long_name=da.attrs["long_name"] + "(spatial maximum)", units=da.attrs["units"])
    
    results["total_pr"] = ds.cloud_pr.sum(("lat", "lon")) * grid_spacing**2 * time_spacing
    results["total_pr"].attrs = dict(long_name="total precipitation", units="kg")

    return results

def water_path_histograms(ds):
    # bins
    lbins = np.logspace(np.log10(1e-3), np.log10(1e2), 100)
    # hist
    def hist_1d(vals, bins):
        vals = vals[~np.isnan(vals)]
        hist = np.histogram(vals, bins=bins)[0]
        return hist
    # data
    hdata = ('anvil_fwp',)
    # results
    results = xr.Dataset(coords={'time': ds.time, 'lbins': lbins[:-1]})
    for v in hdata:
        out = []
        l_out = []
        for t in ds.time:
            da_t = ds[v].sel(time=t)
            l_out.append(hist_1d(da_t.values, lbins))
        results[v + '_lhist'] = (('time', 'lbins'), np.stack(l_out))
    for v in hdata:
        results[v + '_lhist'].attrs = ds[v].attrs
    return results


def calculate(ds):
    ''' Calculate study statistics from cloud dataset, then make area predictions. '''
    
    ds = ds.copy()

    results = general_results(ds)
    results = results.merge(spatial_aggregate_results(ds))
    results = results.merge(anvil_base_height_results(ds))
    results = results.merge(thick_area_results(ds))
    results = results.merge(water_path_histograms(ds))
    results = results.merge(analytical.make_area_predictions(results))

    return results
