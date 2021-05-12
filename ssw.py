#%%
from re import X
import numpy as np
from numpy.core.fromnumeric import size
import xarray as xr
import glob
from os import listdir
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, NoNorm, Normalize
from scipy.interpolate import interp1d
import pandas as pd

#%% Daily VER
am_pm = 'All'
min_sza = 96
d_m = 'daily'
path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/ver/'
filename = '{2}/{0}/{1}_{0}_{2}_ver_clima_{3}.nc'.format(min_sza, am_pm, d_m, '{}')
with xr.open_mfdataset(path+filename.format('*')) as mds:

    mds['mean_ver'] *= np.pi*4/0.55 
    mds['std_ver'] *= np.pi*4/0.55
    mds = mds.reindex(latitude_bins=mds.latitude_bins[::-1])
    
# %%
year = 2010
mds.mean_ver.sel(
    time=slice('{}-10-20'.format(year), '{}-02'.format(year+1)),
    latitude_bins=80, 
    z=slice(70e3, 95e3)
    ).plot.contourf(
        x='time', y='z', 
        cmap='viridis', vmin=0, vmax=1.5e5,
        add_colorbar=False)
ax = plt.gca()
ax.text(0.05, 0.9, '{}-{}'.format(year, year+1), 
            backgroundcolor='w', fontweight='bold', transform=ax.transAxes)
# %%
mds.mean_ver.resample(time='1M').mean().sel(
    latitude_bins=80, z=slice(70e3, 95e3)
        ).plot(x='time', y='z', 
                cmap='viridis', vmin=0, vmax=1.5e5,
                add_colorbar=True, cbar_kwargs=dict(label='[photons cm-3 s-1]'),
                figsize=(8,2)
                )
plt.gca().set_title('70$^\circ$ - 90$^\circ$ N')
# %%
