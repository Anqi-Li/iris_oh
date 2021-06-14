#%%
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar

# %%
def error_prop(a, b, sigma_a, sigma_b, cov_ab):
    return np.sqrt(a**2 * sigma_b**2 + b**2 * sigma_a**2 + 2*a*b*cov_ab)
    
#%%

def mod_zenith(year):
    path_org = '~/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/years/false_zenith/'
    filename = 'iri_ch1_ver_{}.nc'

# year = 2001
    with xr.open_dataset(path_org+filename.format(year)) as ds:
        zi_a = ds.zenith_intensity.attrs
        zie_a = ds.zenith_intensity_error.attrs

        ds['zenith_intensity'] = (
            np.sqrt(2*np.pi) * ds.peak_intensity*ds.peak_sigma*1e2
            ).assign_attrs(
                zi_a
            )
        ds['zenith_intensity_error'] = (
            2*np.pi * error_prop(
                ds.peak_intensity, ds.peak_sigma*1e2, #m->cm
                ds.peak_intensity_error, ds.peak_sigma_error*1e2, #m->cm
                ds.cov_peak_intensity_peak_sigma*1e2) #m->cm
            ).assign_attrs(
                zie_a
            )
    return ds.compute()
# %% save ds
# year = 2001
for year in range(2012,2018):
    print(year)
    ds = mod_zenith(year)
    path_new = '~/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/years/'
    filename = 'iri_ch1_ver_{}.nc'

    delayed_obj = ds.to_netcdf(path_new+filename.format(year), compute=False)
    with ProgressBar():
        delayed_obj.compute()
    
# %%
# import matplotlib.pyplot as plt
# year=2001
# orbit = 4660
# path_new = '~/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/years/'
# filename = 'iri_ch1_ver_{}.nc'
# with xr.open_dataset(path_new+filename.format(year)) as ds:
#     print(ds.zenith_intensity.where(ds.sza>96).sel(
#         time='2001-11').mean('time'))
    
#%%    
    # ds_orbit = ds_new.where(ds_new.orbit==orbit, drop=True)

    # fig, ax = plt.subplots(2,1, sharex=True)
    # line_args = dict(ax=ax[0], x='time', ylim=(0,200e9))
    # ds_orbit.zenith_intensity.plot(**line_args, c='k', ls='-')
    # (ds_orbit.zenith_intensity+ds_orbit.zenith_intensity_error).plot(**line_args, c='b', ls=':')
    # (ds_orbit.zenith_intensity-ds_orbit.zenith_intensity_error).plot(**line_args, c='b', ls=':')

    # contour_args = dict(ax=ax[1], x='time', y='z', add_colorbar=False, cmap='viridis')
    # ds_orbit.ver.where(ds_orbit.A_diag>0.8).plot(**contour_args)
# %%
# ds.where(ds.zenith_intensity_error==np.inf, drop=True).where(ds.peak_sigma_error!=np.inf, drop=True).isel(time=0)

# %%
# test = ds.sel(time='2001-10-28T16:11:35.968332800')
# error = 2*np.pi * error_prop(
#         test.peak_intensity, test.peak_sigma*1e2, #m->cm
#         test.peak_intensity_error, test.peak_sigma_error*1e2, #m->cm
#         test.cov_peak_intensity_peak_sigma*1e2) #m->cm
