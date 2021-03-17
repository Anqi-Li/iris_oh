#%%
import numpy as np
import xarray as xr
import glob
from os import listdir
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d

#%% limb orbit plot example
ch = 1
path_limb = '/home/anqil/Documents/sshfs/oso_extra_storage/StrayLightCorrected/Channel{}/'.format(ch)
orbit = 62293
filename_limb = 'ir_slc_{}_ch{}.nc'.format(str(orbit).zfill(6), ch)
with xr.open_dataset(path_limb+filename_limb) as ds:
    ds.pixel.attrs['long_name'] = 'IRI Ch{} pixel number'.format(ch)
    ds.data.attrs['long_name'] = 'IRI CH{} radiance'.format(ch)
    fig, ax = plt.subplots(3,1, sharex=True, sharey=False, figsize=(15,15))
    ds.data.plot(x='time', ax=ax[0], 
                norm=LogNorm(), vmin=1e8, vmax=1e15, extend='both',
                cmap='viridis', robust=True)

    alts = np.arange(10000., 90000., 1000.)
    ir_altitude = []
    error_altitude = []
    for (data, error, alt) in zip(ds.data.T, ds.error.T, ds.altitude):
        f = interp1d(alt, data, bounds_error=False)
        ir_altitude.append(f(alts))
        f = interp1d(alt, error, bounds_error=False)
        error_altitude.append(f(alts))
    ir_altitude = xr.DataArray(ir_altitude, coords=[ds.time, alts], dims=['time', 'Altitude'], attrs=ds.data.attrs)
    error_altitude = xr.DataArray(error_altitude, coords=[ds.time, alts], dims=['time', 'Altitude'], 
                    attrs=dict(long_name='IRI CH1 error', units=ds.data.attrs['units']))

    ir_altitude.plot(x='time', y='Altitude', ax=ax[1],
                    norm=LogNorm(), vmin=1e10, vmax=1e12)
    error_altitude.plot(x='time', y='Altitude', ax=ax[2],
                    norm=LogNorm())

#%%