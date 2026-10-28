
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr
import scipy
import datetime
import global3d_track as g3d

''' Functions to aid plotting '''

def get_evolution(d):
    dev = d.sel(phase='developing').dropna('interp_time', how='all')
    mat = d.sel(phase='mature').dropna('interp_time')
    dec = d.sel(phase='decay').dropna('interp_time')
    evolution = xr.concat((dev, mat, dec), dim='interp_time')
    return evolution

def show_phases(ax, y=0, xvals=(.1,.45,.85), fs=12, show_decay=True):
    phase_ends = {'developing': 0.25, 'growing': 0.65}
    buff = 0.0125
    tparams = dict(c='grey', weight='bold', fontsize=fs-2, va='top')
    sparams = dict(fc='grey', zorder=2, alpha=.1)
    ax.axvspan(phase_ends['developing']-buff, phase_ends['developing']+buff, **sparams)
    ax.axvspan(phase_ends['growing']-buff, phase_ends['growing']+buff, **sparams)
    ax.text(xvals[0], y, "DEVELOPING", ha='center', **tparams)
    ax.text(xvals[1], y, "MATURE", ha='center', **tparams)
    if show_decay:
        ax.text(xvals[2], y, "DECAY", ha='center', **tparams)
    
def annotate_arrow(ax, x0, y_lo, y_hi, x_lo=None, x_hi=None, fs=12):
    # percentage increase
    pct = (y_hi - y_lo) / y_lo * 100

    # dashed connectors
    if x_lo is not None:
        ax.hlines(y_lo, x_lo, x0, ls="--", lw=1, color="k")
    if x_hi is not None:
        ax.hlines(y_hi, x_hi, x0, ls="--", lw=1, color="k")

    # end bars
    bar = 0.025 * np.diff(ax.get_ylim())[0]
    if x_lo is not None:
        ax.vlines(x_lo, y_lo - bar, y_lo + bar, lw=1, color="k")
    if x_hi is not None:
        ax.vlines(x_hi, y_hi - bar, y_hi + bar, lw=1, color="k")

    # double-headed arrow
    ax.annotate(
        "",
        xy=(x0, y_hi),
        xytext=(x0, y_lo),
        arrowprops=dict(arrowstyle="<->", lw=1, color="k"),
    )

    # label
    ax.text(x0 + 0.02, 0.9 * y_hi, f"{pct:.0f}%", va="center", fontsize=fs - 2)

def panel_label(ax, letter, text='', fs=12):
    label = f"$\\mathbf{{({letter})}}$" + (f" {text}" if text else '')
    ax.text(0.015, 0.95, label, transform=ax.transAxes, fontsize=fs, va='top')

def plot_relationship(ax, x, y, scale='linear', low=0, high=100, scatter_kw=None, line_kw=None, cmap='viridis', density=True, scatter=True, line=True):
   
    scatter_kw = scatter_kw or {}
    line_kw = line_kw or {}

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if scale == 'log':
        mask &= (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    lo, hi = np.percentile(x, [low, high])
    mask_pct = (x >= lo) & (x <= hi)
    x, y = x[mask_pct], y[mask_pct]

    # regression
    if scale == 'log':
        xr, yr = np.log10(x), np.log10(y)
    else:
        xr, yr = x, y

    m, b, r, p, se = result = scipy.stats.linregress(xr, yr)
    
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')

    sc = None
    if scatter:
        if density:
            density = scipy.stats.gaussian_kde(np.vstack([xr, yr]), bw_method=0.3)(np.vstack([xr, yr]))
            density = (density - density.min()) / (density.max() - density.min())
            order = density.argsort()
            c = density[order]
            x, y = x[order], y[order]
        else:
            c = cmap
    
        _scatter_kw = dict(s=5, alpha=0.6, rasterized=True, linewidths=0)
        _scatter_kw.update(scatter_kw)
        sc = ax.scatter(x, y, c=c, cmap=cmap, **_scatter_kw)

    ln = None
    if line:
        x_fit = np.linspace(xr.min(), xr.max(), 200)
        y_fit = m * x_fit + b
        if scale == 'log':
            x_fit, y_fit = 10**x_fit, 10**y_fit
    
        _line_kw = dict(c='k', lw=1.5)
        _line_kw.update(line_kw)
    
        ax.plot(x_fit, y_fit, c='w', lw=_line_kw.get('lw', 1.5) + 0.5)  # white outline
        ln = ax.plot(x_fit, y_fit, **_line_kw)[0]

    return result, sc, ln

''' The below functions were used to make Figure 1 '''
    
def plot_masks(ax, d, t, cmap='viridis', vmin=None, vmax=None, params={}):
    d.sel(time=t).sel(level_full=55).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True, **params)
    gl = ax.gridlines(draw_labels=True, alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    return ax

def plot_surface(ax):
    img = mpimg.imread("/work/bb1153/b382635/data/world.200407.3x5400x2700.jpg")
    ax.imshow(img, transform=ccrs.PlateCarree())
    return ax

def plot_surface3d(ax, data, z, params={}):
    # Load background image
    img = mpimg.imread("/work/bb1153/b382635/data/world.200407.3x5400x2700.jpg")

    lat = data.lat.values[::-1]
    lon = data.lon.values
    X, Y = np.meshgrid(lon, lat)
    z_surface = np.full_like(X, z)

    # Image coordinates
    img_lat_res = 180 / img.shape[0]   # degrees per pixel in latitude
    img_lon_res = 360 / img.shape[1]   # degrees per pixel in longitude

    # Compute cropping indices
    lat_start = int((90 - lat.max()) / img_lat_res)
    lat_end   = int((90 - lat.min()) / img_lat_res)

    lon_start = int((lon.min() + 180) / img_lon_res)
    lon_end   = int((lon.max() + 180) / img_lon_res)

    # Crop the image
    img_cropped = img[lat_start:lat_end, lon_start:lon_end]
    img_cropped = img_cropped / 255
    img_cropped = np.clip(img_cropped * 1.2, 0, 1)

    # Normalize if needed
    if img_cropped.ndim == 3:
        img_rgb = img_cropped[..., :3]  # Remove alpha if present
    else:
        img_rgb = np.stack([img_cropped]*3, axis=-1)

    # Plot the image as a 3D surface
    di=dict(rstride=1, cstride=1, alpha=0.5, edgecolor='none', linewidth=0, shade=False)
    di.update(params)
    ax.plot_surface(
        X, Y, z_surface,
        facecolors=img_rgb,
        zorder=0,
        **di
    )
    return ax
    
def plot_scatter(ax, ds, hue, params={}, cmap='Blues', f=0):
    di = dict(add_colorbar=False, zorder=3)
    di.update(params)

    # Assume original coordinates are 'lat' and 'lon'
    if f:
        lat_fine = np.linspace(ds.lat.min(), ds.lat.max(), num=ds.dims['lat'] * f)
        lon_fine = np.linspace(ds.lon.min(), ds.lon.max(), num=ds.dims['lon'] * f)
        level_fine = np.linspace(ds.level_full.min(), ds.level_full.max(), num=ds.dims['level_full'] * f)
        ds = ds.interp(lat=lat_fine, lon=lon_fine, level_full=level_fine, method='nearest')

    # cmap
    cmap = plt.get_cmap(cmap)(np.linspace(0.5, 1.0, 256))

    # alpha
    cmap[:,-1] = np.linspace(0,1,256) # gets less transparent as color gets darker
    cmap = ListedColormap(cmap)

    ds.plot.scatter(ax=ax, x='lon', y='lat', z='level_full', hue=hue, edgecolors='none', cmap=cmap, **di,)
    return ax

def modify_axis(ax, d, fsize=12):
    ax.set_xlim(d.lon.min(), d.lon.max())
    ax.set_ylim(d.lat.min(), d.lat.max())
    ax.set_zlim(d.level_full.max(), d.level_full.min())
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_zlabel("")
    ax.tick_params(labelsize=fsize-2)
    ax.set_zticklabels([])
    return ax

def make_map(ax, pdata, utc, cmap, vmax, fs):
    proj = ccrs.Robinson()
    
    # plot tropics
    ax = plot_masks(ax, pdata, utc, cmap=cmap, vmin=0, vmax=vmax, params=dict(add_colorbar=0))
    ax.set_extent((0,360,-30,30), crs=ccrs.PlateCarree())
    ax.plot([0,360],[15,15],ls='--',c='y',transform=ccrs.PlateCarree())
    ax.plot([0,360],[-15,-15],ls='--',c='y',transform=ccrs.PlateCarree())
    ax.coastlines()
    plot_surface(ax)
    ax.set(title=None)
    ax.set_xlabel('Longitude', fontsize=fs)
    
    # add solar time
    utc_dt = datetime.datetime.strptime(utc, '%Y%m%d%H%M')
    utc_hour = utc_dt.hour + utc_dt.minute / 60
    pc = ccrs.PlateCarree()
    clat = 0
    
    def solar_time(xproj):
        x = np.atleast_1d(np.asarray(xproj, dtype=float))
        lons = pc.transform_points(proj, x, np.full_like(x, clat))[:, 0]
        return np.array([(utc_dt + datetime.timedelta(hours=lon / 15.0)).hour for lon in lons])
        
    top_ax = ax.secondary_xaxis('top', functions=(lambda x:x, lambda x:x))
    top_ax.set_xlabel('Local solar time (LST) [hour]', fontsize=fs)
    xticks = np.linspace(*ax.get_xlim(), 9)
    xlables = np.nan_to_num(solar_time(xticks).round(0).astype(int), nan=0)
    top_ax.set_xticks(xticks, xlables)
    return ax

def prep_example_data(mask):
    # load
    variables = ['cli', 'clw', 'pfull', 'zg', 'zghalf', 'wa_phy']
    data = g3d.utils.data_tools.load_corresponding_data(mask, None, variables=variables)
    data = g3d.utils.data_tools.preprocess_for_tobac(data)
    data['lat'] = data.lat.round(2)
    data['lon'] = data.lon.round(2)
    mask['lat'] = mask.lat.round(2)
    mask['lon'] = mask.lon.round(2)
    data['condensate'] = data.cli + data.clw
    # prep
    z_pfull = data.pfull.mean(('lat','lon','time'))
    z_zg = data.zg.mean(('lat','lon')) / 1000
    dw = data.where(data > 1).where(mask > 0)
    dc = data.where(data > 1e-5).where(mask > 0)
    dw = dw.assign_coords({'zg':z_zg}).compute()
    dc = dc.assign_coords({'zg':z_zg}).compute()
    mask = mask.assign_coords({'zg':z_zg})
    return mask, dc, dw
    
def make_proj(ax, data, coord='level_full', levels=[0.5], params=dict(offset=-1, cmap='viridis')):
    if coord=='level_full':
        m = data.sel(time=t).max('level_full')
        X, Y = np.meshgrid(m.lon, m.lat)
        ax.contour(X, Y, (m.fillna(0)).astype(int), levels, zdir='z', **params)
    if coord=='lon':
        m = data.sel(time=t).max('lon')
        Y, Z = np.meshgrid(m.lat, m.zg)
        ax.contour((m.fillna(0)).astype(int), Y, Z, levels, zdir='x', **params)
    if coord=='lat':
        m = data.sel(time=t).max('lat')
        X, Z = np.meshgrid(m.lon, m.zg)
        ax.contour(X, (m.fillna(0)).astype(int), Z, levels, zdir='y', **params)
    return ax

def make_example(ax, mask, utc, show_dc=1, show_dw=0, zaxis=1, haxis=1, name="", fsize=12, dpi=100):
    mask, dc, dw = prep_example_data(mask)

    # local time
    t = pd.to_datetime(utc)
    def solar_time(lon):
        return (t + datetime.timedelta(hours=lon / 15.0))
    t_local = solar_time(mask.lon.mean().item())
    
    # plot condensate
    # norm = plt.Normalize(vmin=1e-5, vmax=1e-3)
    norm = mpl.colors.LogNorm()
    cmap = plt.get_cmap('Blues')
    ax = plot_surface3d(ax, dc, z=90, params=dict(alpha=.8))
    
    # scatter size
    p0 = ax.transData.transform((float(dc.lon.min()), float(dc.lat.min())))
    p1 = ax.transData.transform((float(dc.lon.min() + 0.1), float(dc.lat.min() + 0.1)))
    dx_pix = abs(p1[0] - p0[0])
    size = (dx_pix * 72.0 / dpi)
    
    # scatter
    if show_dc:
        ax = plot_scatter(ax, dc.sel(time=t), hue='condensate', params=dict(s=size, norm=norm, marker='s'), cmap='Blues', f=0)
    if show_dw:
        ax = plot_scatter(ax, dw.sel(time=t), hue='wa_phy', params=dict(s=size, norm=None, marker='s'), cmap='Reds', f=0)
    date_str = mask.time.dt.strftime('%b-%d').values
    ax.view_init(20, None)
    ax = modify_axis(ax, dc, fsize=fsize)
    
    # label z axis
    if zaxis:
        zg = dc.zg.values
        ax.set_zticks(dc.level_full[::10])
        ax.set_zticklabels(zg[::10].astype(int))  # Equivalent to ax.zaxis.set_ticks(zg)
        ax.set_zlabel("Height [km]", fontsize=12)
    else:
        ax.get_zaxis().set_visible(False)

    # label horizontal
    if haxis:
        ax.set_ylabel("Latitude [°] ", fontsize=fsize)
        ax.set_xlabel("Longitude [°]", fontsize=fsize)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylabel(None)
        ax.set_xlabel(None)
    
    # stretch height
    xrange = dc.lon.max() - dc.lon.min()
    yrange = dc.lat.max() - dc.lat.min()
    zrange = dc.zg.max() - dc.zg.min()
    ax.set_box_aspect((xrange, yrange, zrange*0.2))
    
    # horizontal ticks
    xticks = np.linspace(float(dc.lon.min()), float(dc.lon.max()), 4).round(1)
    yticks = np.linspace(float(dc.lat.min()), float(dc.lat.max()), 3).round(1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    
    # title
    tstr = f"{name}\n{t_local.strftime('%H:%M')} LST"
    ax.set_title(None)
    ax.text2D(0.95, 1.125, tstr, ha='center', va='top', transform=ax.transAxes, fontsize=fsize,)
    return ax