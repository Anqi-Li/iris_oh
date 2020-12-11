#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from xhistogram.xarray import histogram

# %%
orbit = 9997
orbit_num = str(orbit).zfill(6)
path = '/home/anqil/Documents/osiris_database/iris_oh/archive/'
filename = 'iri_oh_ver_{}.nc'
ds = xr.open_dataset(path+filename.format(orbit_num))

ver_data = ds.ver.where(ds.mr>0.8)
moving_window_big = 200
ver_sma_big = ver_data.rolling(
    time=moving_window_big, center=True).mean().rename('ver_sma_big')
moving_window_small = 20
ver_sma_small = ver_data.rolling(
    time=moving_window_small, center=True).mean().rename('ver_sma_small')
#%%
fig, ax = plt.subplots(3,1, figsize=(10,15), sharex=True, sharey=True)
plot_args = dict(y='z', vmin=0, vmax=1e4, cmap='viridis', robust=True, ylim=[60e3,100e3])

ver_data.plot(ax=ax[0], **plot_args)
ver_sma_small.plot(ax=ax[1], **plot_args)
ver_sma_big.plot(ax=ax[2], **plot_args)

ax[0].set(title='Original',
        xlabel='')
ax[1].set(title='Moving Average ({})'.format(moving_window_small),
        xlabel='')
ax[2].set(title='Moving Average ({})'.format(moving_window_big),
        )

fig.savefig('/home/anqil/Documents/osiris_database/iris_oh/full_orbit_ver_{}.png'.format(ds.orbit.item()))
plt.show()

# %%
fig, ax = plt.subplots(3,1, figsize=(6,10), sharex=True, sharey=True)
plot_args = dict(y='z', vmin=-2000, vmax=2000, ylim=(60e3, 100e3))

ver_mean = ver_data.mean('time')
(ver_data - ver_mean).plot(ax=ax[0], **plot_args)
(ver_data - ver_sma_big).plot(ax=ax[1], **plot_args)
(ver_sma_small - ver_sma_big).plot(ax=ax[2], **plot_args)

ax[0].set(title='Original - Total mean of the orbit',
        xlabel='')
ax[1].set(title='Original - Moving Average ({})'.format(moving_window_small),
        xlabel='')    
ax[2].set(title='MA ({}) - MA ({})'. format(moving_window_small, moving_window_big),
        )

fig.savefig('/home/anqil/Documents/osiris_database/iris_oh/structure_{}.png'.format(ds.orbit.item()))
plt.show()

# %% Check mean, std, histogram
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
# Choose colormap
cmap = pl.cm.viridis
# Get the colormap colors
my_cmap = cmap(np.arange(cmap.N))
# Set alpha
my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
# Create new colormap
my_cmap = ListedColormap(my_cmap)

def plot_hist2d(data, ax):
    if data.name == None:
        data.rename('data')
    mean = data.mean('time')
    std = data.std('time')
    h_z = histogram(data, bins=[np.linspace(-1e3,8e3)], dim=['time'])
    h_z.plot(ax=ax, vmax=h_z.sel(z=85e3).max(), cmap=my_cmap)
    mean.plot(ax=ax, y='z', color='k', ls='-')
    std.pipe(lambda x: mean+x).plot(ax=ax, y='z', color='k', ls='--')
    std.pipe(lambda x: mean-x).plot(ax=ax, y='z', color='k', ls='--')

fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
plot_hist2d(ver_data, ax[0])
plot_hist2d(ver_sma_big, ax[1])

ax[0].set(title='Original',
        xlabel='')
ax[1].set(title='Running mean ({}-image-window)'.format(moving_window_big),
        ylim=(60e3,100e3))

ds.error2_retrieval.pipe(
    lambda x: np.sqrt(x) + ver_data).mean(
        'time').plot(ax=ax[0], y='z', color='b', ls=':')
ds.error2_retrieval.pipe(
    lambda x: -np.sqrt(x) + ver_data).mean(
        'time').plot(ax=ax[0], y='z', color='b', ls=':')

# %%
