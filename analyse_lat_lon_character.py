#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from xhistogram.xarray import histogram
import cartopy.crs as ccrs
import glob 
import pandas as pd

#%% open statistics files -- time_lat
path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/'
filename = 'bounded/bounded_time_lat_{}.nc'
def set_idx(ds, year):
    idx = [np.datetime64('{}-{}'.format(year, str(month).zfill(2))) for month in ds.time_bins.values] 
    ds = ds.assign_coords(time_bins=idx)
    return ds

mds = []
for year in range(2001,2018):
    with xr.open_dataset(path+filename.format(year)) as ds:
        mds.append(set_idx(ds, year))
# for file in glob.glob(path+filename.format('????')):
#     year = file[-7:-3]
#     with xr.open_dataset(file) as ds:
#         mds.append(set_idx(ds, year))
mds = xr.merge(mds)

#% some adjustments on the longterm dataset
mds = mds.rename(dict(latitude_bins='Latitude', time_bins='Time'))
mds.Latitude.attrs['units'] = 'deg N'

data_vars_lst = 'amplitude peak_height thickness'.split()
data_vars_mul = [1, 1e-3, 1e-3]
data_vars_units = 'pho_cm-1_s-1 km km'.split()
for i in range(len(data_vars_lst)):
    for s in 'mean_{} std_{}'.split():
        mds[s.format(data_vars_lst[i])] *= data_vars_mul[i]
        mds[s.format(data_vars_lst[i])].attrs['units'] = data_vars_units[i] 

#%% Longterm contourf plot
fig, ax = plt.subplots(len(data_vars_lst)+1,1, figsize=(15,10), sharex=True, sharey=True)
contourf_args = dict(x='Time', robust=True)
for i, var in enumerate(['std_{}'.format(v) for v in data_vars_lst]):
    mds[var].plot.contourf(ax=ax[i], **contourf_args)
[ax[i].set(xlabel='') for i in range(len(data_vars_lst))]
mds.count_amplitude.plot.contourf(ax=ax[-1], vmax=8e3, **contourf_args)
ax[0].set(title='STD')
ax[-1].set(title='Sample Count')

#%% open statistics files -- time_lat_lon
filename = 'archive/time_lat_lon_{}.nc'
def set_midx(ds):
    midx = pd.MultiIndex.from_product(
        [[year], ds.time_bins.values], names=('year', 'season'))
    ds = ds.assign_coords(time_bins=midx)
    return ds

mds = []
for year in range(2001,2011):
    with xr.open_dataset(path+filename.format(year)) as ds:
        mds.append(set_midx(ds))
mds = xr.merge(mds)
#% final adjustment on the longterm dataset
mds = mds.rename(dict(latitude_bins='Latitude', longitude_bins='Longitude', time_bins='Time'))
mds.Latitude.attrs['units'] = 'deg N'
mds.Longitude.attrs['units'] = 'deg E'

data_vars_lst = 'max_pw_freq max_pw amplitude peak_height thickness'.split()
data_vars_mul = [1e3, 1, 1, 1e-3, 1e-3]
data_vars_units = 'km-1 ? pho_cm-1_s-1 km km'.split()
for i in range(len(data_vars_lst)):
    for s in 'mean_{} std_{}'.split():
        mds[s.format(data_vars_lst[i])] *= data_vars_mul[i]
        mds[s.format(data_vars_lst[i])].attrs['units'] = data_vars_units[i] 

# mds['max_pw_freq'] = mds.max_pw_freq*1e3
# mds.max_pw_freq.attrs['units'] = '  km-1'
# mds.max_pw.attrs['units'] = '?'
# mds.amplitude.attrs['units'] = 'photons cm-1 s-1'
# mds['peak_height'] = mds.peak_height*1e-3
# mds.peak_height.attrs['units'] = 'km'
# mds['thickness'] = mds.thickness*1e-3
# mds.thickness.attrs['units'] = 'km'

mds = mds.unstack('Time')

#%% cartography plot
contourf_args = dict(y='Latitude', x='Longitude', col='season', row='year', robust=True,
    subplot_kws=dict(projection=ccrs.PlateCarree(central_longitude=180)),
    cbar_kwargs=dict()
    )
time_seq = ['DJF', 'MAM', 'JJA', 'SON']
year = [2002]
data_vars_lst = 'max_pw_freq max_pw amplitude peak_height thickness'.split()
for var in ['mean_{}'.format(v) for v in data_vars_lst]:
    fig = plt.figure(figsize=(10,5))
    p = mds.sel(year=year).reindex(season=time_seq)[var].plot.contourf(**contourf_args)
    for ax in p.axes.flat:
        ax.coastlines()
        ax.set_global()
    plt.suptitle(var)
    plt.show()

#%%
var = 'sp_sample_count'#'count_amplitude'
fig = plt.figure(figsize=(10,5))
p = mds.sel(year=year).reindex(season=time_seq)[var].plot.contourf(**contourf_args)
for ax in p.axes.flat:
    ax.coastlines()
    ax.set_global()
plt.suptitle(var)
plt.show()

































#%%


# %% Histograms
amplitude_bins = np.arange(2e3, 6e3, .5e3)
peak_height_bins = np.arange(83e3, 85e3, .05e3)
thickness_bins = np.arange(32e2, 40e2, .5e2)
h = histogram(ds.amplitude, ds.peak_height, ds.thickness,
    bins=[amplitude_bins, peak_height_bins, thickness_bins], dim=['time'])
h.plot(x='peak_height_bin', y='thickness_bin', col='amplitude_bin', col_wrap=5)

# %%
zonal_groups = ds.groupy_bins(ds.latitude, bins=latitude_bins, labels=latitude_labels)
h_amp, h_peak_height, h_thickness, labels = [], [], [], []
for latitude_label, data in zonal_groups:
    print(latitude_label)
    labels.append(latitude_label)
    histogram_args = dict(dim=['time'])
    h_amp.append(histogram(data.amplitude, bins=[amplitude_bins], **histogram_args))
    h_peak_height.append(histogram(data.peak_height, bins=[peak_height_bins], **histogram_args))
    h_thickness.append(histogram(data.thickness, bins=[thickness_bins], **histogram_args))

h_amp = xr.concat(h_amp, dim='latitude').assign_coords(latitude=labels).sortby('latitude')
h_peak_height = xr.concat(h_peak_height, dim='latitude').assign_coords(latitude=labels).sortby('latitude')
h_thickness = xr.concat(h_thickness, dim='latitude').assign_coords(latitude=labels).sortby('latitude')

# %%
fig, ax = plt.subplots(3,1, figsize=(5,8), sharex=True)#, sharey=True)
plot_args = dict(x='latitude')#, add_legend=False)
h_amp.plot(ax=ax[0], **plot_args)
h_peak_height.plot(ax=ax[1], **plot_args)
h_thickness.plot(ax=ax[2], **plot_args)

ax[0].set(title='Amplitutde', xlabel='')
ax[1].set(title='Peak Height', xlabel='')
ax[2].set(title='Thickness')


