#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from xhistogram.xarray import histogram
import glob

# %% open VER file
orbit = 4814 #3713
orbit_num = str(orbit).zfill(6)
path = '/home/anqil/Documents/osiris_database/iris_oh/'
filename = 'iri_oh_ver_{}.nc'
ds = xr.open_dataset(path+filename.format(orbit_num))
ds.close()
ds = ds.update({'tp' : (['time'],
        (ds.time - ds.time[0])/(ds.time[-1]-ds.time[0]))})

#%%
isel_args = dict(time=0)
# ds.isel(**isel_args).ver.plot(y='z')

#%% characterise the airglow layer by a gaussian fit
from characterise_agc_routine import characterise_layer, gauss

da_ver_profile = ds.ver.where(ds.A_diag>=0).isel(**isel_args)
popt = characterise_layer(da_ver_profile)

ds.ver.isel(**isel_args).plot(y='z', label='VER data')
ds.z.pipe(gauss, *popt).plot(y='z', label='Gaussian fit: \n a=%1.0f, x0=%1.0f, sigma=%1.0f' % tuple(popt))
ds.ver.isel(**isel_args).rolling(z=5, center=True).mean(
    ).plot(y='z', label='SMA 5km')
p0 = (5e3, 85e3, 5e3)
ds.z.pipe(gauss, *p0).plot(y='z', label='Initial guess: \n a=%1.0f, x0=%1.0f, sigma=%1.0f' % tuple(p0))
plt.legend()
# %%
from characterise_agc_routine import process_file
f = path+filename.format(orbit_num)
ds_agc = process_file(f, save_file=False)
ds = ds.update(ds_agc).swap_dims({'time': 'tp'})

#%%
fig, ax = plt.subplots(3,1, sharex=True)
plot_args = dict(x='tp')
ds.amplitude.plot(ax=ax[0], **plot_args)
ds.peak_height.plot(ax=ax[1], **plot_args)
ds.thickness.plot(ax=ax[2], **plot_args)

ax[0].set(title='Amplitude', xlabel='')
ax[1].set(title='Peak height', xlabel='')
ax[2].set(title='Thickness', xlabel='Ratio into the night')
# %%
