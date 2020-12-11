#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
# %%
path='/home/anqil/Documents/osiris_database/ex_data/cmam/'
file = '{}_6hrChem_CMAM-Ext_CMAM30-SD_r1i1p1_2010010100-2010063018.nc'

def get_ds(species):
    return xr.open_dataset(path+file.format(species))

#%%
species = 'numden'
ds = get_ds(species)

numden_mean = ds[species].mean(dim=['lon', 'time', 'lat']
    ).assign_attrs(ds[species].attrs)

numden_mean.plot(y='plev', xscale='log', yscale='log')
plt.gca().invert_yaxis()

#%%
species = 'ntveo200'
ds = get_ds(species)

mean = ds[species].mean(dim=['lon', 'time', 'lat']
    ).assign_attrs(ds[species].attrs)
mean.plot(y='plev', xscale='log', yscale='log')
plt.gca().invert_yaxis()

# def p2z(plev):
#     p_sea = 1e5
#     return (10**(np.log10(plev/p_sea)/5.2558797) - 1) / -6.8755856e-6 
# mean = mean.assign_coords(plev=p2z(ds.plev)/3
#     ).rename(dict(plev='z'))
# mean.plot(y='z', xscale='log')


# %%
species = 'ntvegrnline'
ds = get_ds(species)

mean = ds[species].mean(dim='time lat lon'.split(), keep_attrs=True)
std = ds[species].std(dim='time lat lon'.split(), keep_attrs=True, skipna=True)

plot_args = dict(y='plev', xscale='linear', yscale='log', color='k')
mean.pipe(lambda x: x+std).plot(**plot_args, ls='--')
mean.pipe(lambda x: x-std).plot(**plot_args, ls='--')
mean.plot(**plot_args, ls='-')

plt.gca().invert_yaxis()
plt.show()

# %% heatmap plot
from xhistogram.xarray import histogram
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

h_plev = histogram(ds[species], bins=[np.linspace(-10, 5e2)], 
        dim='time lat lon'.split())
h_plev.plot(cmap=my_cmap, vmax=h_plev.sel(plev=1e-1).max())

plot_args = dict(y='plev', xscale='linear', yscale='log', color='k')
mean.plot(**plot_args, ls='-')
mean.pipe(lambda x: x+std).plot(**plot_args, ls='--')
mean.pipe(lambda x: x-std).plot(**plot_args, ls='--')

plt.gca().invert_yaxis()
plt.show()
# %%
test_arr= xr.DataArray(np.random.randn(10000, 100), name='test')
histogram(test_arr, bins=[np.linspace(-4,4,100)], dim=['dim_0']).plot()
test_arr.mean(dim='dim_0').plot(y='dim_1', color='k')
test_arr.std(dim='dim_0').plot(y='dim_1', color='k')