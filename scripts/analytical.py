import xarray as xr
import numpy as np

'''
Analytical predictions of anvil area expansion and thickness.

'''

time_spacings = 900 # s

def predict_t_all(ds):
    # time varying all inputs
    D = ds.anvil_depth_mu # m
    rho = ds.anvil_rho_mu # kg m-3
    CMT = ds.core_cmt_ABH.sum('core').fillna(0) # kg
    A = (1 / (D*rho) * CMT).cumsum('time') * time_spacings # m2
    t0 = ds.cloud_area.notnull().idxmax('time') # first time the anvil actually exists
    A0 = ds.cloud_area.sel(time=t0)
    return A + (A0 - A.sel(time=t0))

def predict(ds):
    # using constant inputs
    D = ds.anvil_depth_mu.max('time') # m
    rho = ds.anvil_rho_mu.mean('time') # kg m-3
    CMT = ds.core_cmt_ABH.max(('time','core')) # kg
    A = 1 / (D*rho) * CMT.broadcast_like(ds.time).cumsum('time') * time_spacings # m2
    t0 = ds.cloud_area.notnull().idxmax('time') # first time the anvil actually exists
    A0 = ds.cloud_area.sel(time=t0)
    return A + (A0 - A.sel(time=t0))

def make_area_predictions(ds):
    valid = ds.cloud_area > 0
    convecting = ds.core_area_ABH.max("core") > 0
    Apred_v1 = predict(ds)
    Apred_v2 = predict_t_all(ds)
    result = xr.Dataset({"Apred_v1": Apred_v1.where(convecting).assign_attrs(dict(equation=r"$\frac{1}{D\bar{\rho}}\int M_{c,max}\,dt$", period="$t\in(0, t_c)$")),
                         'Apred_v1_cont': Apred_v1.where(valid).assign_attrs(dict(equation=r"$\frac{1}{D\bar{\rho}}\int M_{c,max}\,dt$", period="$t\in(0, t_{max})$")),
                         'Apred_v2': Apred_v2.where(convecting).assign_attrs(dict(equation=r"$\int \frac{M_c}{D\bar{\rho}}\,dt$", period="$t\in(0, t_c)$")),
                         'Apred_v2_cont': Apred_v2.where(valid).assign_attrs(dict(equation=r"$\int \frac{M_c}{D\bar{\rho}}\,dt$, where $t\in(0, t_c)$, else $\int \frac{M_c}{D\bar{\rho}}\,dt|_{t=t_c}$ where $t\in(t_c,t_{max})$", period="$t\in(0, t_{max})$")),
                         })
    for v in result.data_vars:
        result[v].attrs['long_name'] = 'predicted anvil area'
        result[v].attrs['description'] = r'prediction using anvil mean quantities, $D$ and $\bar{\rho}$, and core convective mass flux at the anvil base, $M_c$'
        result[v].attrs['units'] = 'm^2'
    return result

def initial_value(da, dim='interp_time'):
    return da.sel({dim: da.notnull().idxmax(dim)})

class Thickness:

    def __init__(self, ds, pred='Apred_v1', n=0, xs=[0, 0.025, 0.25, 2.5], phase='growing', tau=1):

        # data
        ds = ds.sel(phase=phase)
        A = ds[pred] # predicted anvil area (t)
        rmax_t = np.sqrt(A / np.pi) # predicted anvil radius (t)
        Ac = ds.core_area_ABH # core area (t)
        rc = np.sqrt(Ac / np.pi) # core radius (t)
        rc_0 = initial_value(rc) # initial core radius
        rc_max = rc.max('interp_time') # max core radius
        D = ds.anvil_depth_mu # anvil depth (t)
        D_max = ds.anvil_depth_mu.max('interp_time') # maximum anvil depth
        iwc = ds.core_fwc_ABH # FWC at anvil base (t)
        iwc_init = initial_value(iwc) # initial FWC
        iwc_max = iwc.mean('interp_time') # maximum FWC
        t = ds.interp_time # time
        valid = (ds.cloud_area > 0) # valid period
        valid_c = (Ac > 0) # valid convective period
        t_conv = (Ac > 0).idxmin('interp_time') - 0.05 # time convection ends
        
        # create radius dimension
        Rmax = rmax_t.max('interp_time') # max r reached over time
        r_axis = np.linspace(0, float(Rmax.max()), 100) # all r values in time
        r_axis = xr.DataArray(r_axis, dims="radius", coords={"radius": r_axis})
        r = r_axis.where(r_axis <= rmax_t) # r(t)
                
        # rate of decay of fwc in time
        t_decay = 1 / (1-t_conv) - t / (1-t_conv) * ds.lifetime
        t_decay = xr.where(iwc>0, iwc, t_decay * iwc.ffill('interp_time'))
        
        # radial decay
        rc_ext = xr.where(valid_c, rc, rc.where(valid_c).ffill('interp_time'))
        rc_ext = xr.where(valid, rc_0, 0)
        if n != 0:
            r_decay = 1 - ((r-rc_ext) / (rmax_t-rc_ext))**n # decays to zero
        else:
            r_decay = (rc_ext / r)** 2 # Fabian Senf, decays from core radius independent of time
        r_decay = xr.where(rc_ext > r, 1, r_decay) # no decay at the core
        
        # combine
        iwp = D * r_decay * t_decay
        
        # thick area
        thick_A = []
        for x in xs:
            mask = iwp > x
            thick_r = xr.where(
                mask.all("radius"),
                iwp.radius.isel(radius=-1),
                iwp.radius.sel(radius=mask.idxmin("radius")),
            )
            res = (np.pi * (thick_r**2)).expand_dims(threshold=[x])
            res.name = 'Thick area'
            thick_A.append(res)

        thick_A = xr.concat(thick_A, dim='threshold').assign_attrs(dict(long_name='Thick area', units='m^2'))
    
        # store
        self.A = A
        self.rc = rc
        self.iwc = iwc
        self.D = D
        self.t_decay = t_decay.where(valid)
        self.r_decay = r_decay.where(valid)
        self.iwp = iwp.where(valid)
        self.thick_A = thick_A.where(valid)
