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
path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/oh/'
filename = 'iri_oh_ver_{}.nc'
ds = xr.open_dataset(path+filename.format(orbit_num))
ds.close()
ds = ds.update({'tp' : (['time'],
        (ds.time - ds.time[0])/(ds.time[-1]-ds.time[0]))})

#%%
isel_args = dict(time=0)
# ds.isel(**isel_args).ver.plot(y='z')

#%% characterise the airglow layer by a gaussian fit
from characterise_agc_routine import gauss #characterise_layer, gauss
from scipy.optimize import curve_fit

da_ver_profile = ds.ver.where(ds.A_diag>=0).isel(**isel_args)

def characterise_layer(da_ver_profile, a0=5e3, mean0=85e3, sigma0=5e3):
    y = da_ver_profile.dropna(dim='z')
    x = y.z
    popt, pcov = curve_fit(gauss, x, y, 
                    p0=[a0, mean0, sigma0], 
                    # bounds=([0, 70e3, 0], [1e5, 100e3, 40e3]), #some reasonable ranges for the airglow characteristics
                    )
    
    amplitude, peak_height, thickness_FWHM = popt
    return amplitude, peak_height, thickness_FWHM, np.diag(pcov)

*popt, pcov = characterise_layer(da_ver_profile)

ds.ver.isel(**isel_args).plot(y='z', label='VER data')
ds.z.pipe(gauss, *popt).plot(y='z', color='r', 
    label='Gaussian fit: \n a=%1.0f, x0=%1.0f, sigma=%1.0f' % tuple(popt))
ds.ver.isel(**isel_args).rolling(z=5, center=True).mean(
    ).plot(y='z', color='g', label='moving average 5km')
# p0 = (5e3, 85e3, 5e3)
# ds.z.pipe(gauss, *p0).plot(y='z', label='Initial guess: \n a=%1.0f, x0=%1.0f, sigma=%1.0f' % tuple(p0))
plt.legend()
# %% gaussian fit the whole orbit
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
