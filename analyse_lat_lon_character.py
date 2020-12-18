#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# from scipy.interpolate import interp1d
from xhistogram.xarray import histogram
import cartopy.crs as ccrs

import glob 

# %% open files
path = '/home/anqil/Documents/osiris_database/iris_oh/'
agc_file_lst = glob.glob(path + 'airglow_character/agc_*.nc')
ds = xr.open_mfdataset(agc_file_lst).set_coords(['longitude', 'latitude'])
ds = ds.where(ds.time.dt.year==2002, drop=True)

# %% groupby time, latitude
dlat = 20
latitude_bins = np.arange(-90,90+dlat,dlat)
latitude_labels = latitude_bins[1:]-dlat/2

time_bins, mean, count = [], [], []
for s, data in ds.groupby(ds.time.dt.month):
    time_bins.append(s)
    mean.append(data.groupby_bins(data.latitude, 
        bins=latitude_bins, labels=latitude_labels).median('time', keep_attrs=True))
    count.append(data.groupby_bins(data.latitude, 
        bins=latitude_bins, labels=latitude_labels).count('time'))
mean = xr.concat(mean, dim='time_bins').assign_coords(time_bins=time_bins).sortby('time_bins')
count = xr.concat(count, dim='time_bins').assign_coords(time_bins=time_bins).sortby('time_bins')

# mean = mean.assign({'season': mean.time_bins}).assign_coords(time_bins=np.arange(4))

fig, ax = plt.subplots(4,1, figsize=(6,8), sharex=True)
plot_args = dict(x='time_bins', y='latitude_bins')
parms = 'amplitude peak_height thickness'.split()
for i, par in enumerate(parms):
    mean[par].plot.contourf(ax=ax[i], **plot_args)
    ax[i].set(title=par, xlabel='')
    # ax[i].get_legend().set(bbox_to_anchor=(1 , 1)) 

count[par].rename('').plot.contourf(ax=ax[-1], cmap='viridis', **plot_args)
ax[-1].set(title='sample count', xlabel='Month')

plt.suptitle('year = 2002')

# %% groupby latitude, longtiude
dlat = 20
latitude_bins = np.arange(-90,90+dlat,dlat)
latitude_labels = latitude_bins[1:]-dlat/2

dlon = 20
longitude_bins = np.arange(0,360+dlon, dlon)
longitude_labels = longitude_bins[1:] - dlon/2
longitude_lab, mean, count = [], [], []
for s, data in ds.groupby_bins(ds.longitude, bins=longitude_bins, labels=longitude_labels):
    longitude_lab.append(s)
    mean.append(data.groupby_bins(data.latitude, 
        bins=latitude_bins, labels=latitude_labels).median('time', keep_attrs=True))
    count.append(data.groupby_bins(data.latitude, 
        bins=latitude_bins, labels=latitude_labels).count('time'))
mean = xr.concat(mean, dim='longitude_bins').assign_coords(longitude_bins=longitude_lab).sortby('longitude_bins')
count = xr.concat(count, dim='longitude_bins').assign_coords(longitude_bins=longitude_lab).sortby('longitude_bins')

#%% cartopy plot
if mean.longitude_bins[-1]<360:
    mean = xr.auto_combine([mean, 
        mean.isel(longitude_bins=0).assign_coords(
            longitude_bins=360+dlon/2)])
    count = xr.auto_combine([count, 
        count.isel(longitude_bins=0).assign_coords(
            longitude_bins=360+dlon/2)])

fig, ax = plt.subplots(4,1, figsize=(5,8), sharex=True,
        subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=180)) #ccrs.Orthographic(-80, 35)
                    )
plot_args = dict(x='longitude_bins', y='latitude_bins', 
    transform=ccrs.PlateCarree(),
    )
parms = 'amplitude peak_height thickness'.split()
for i, par in enumerate(parms):
    mean[par].plot.contourf(ax=ax[i], **plot_args)
    ax[i].set(title=par, xlabel='')
    ax[i].coastlines()
    ax[i].set_global()
# ax[-1].set(xlabel='longitude_bins')

count[par].rename('').plot.contourf(ax=ax[-1], cmap='viridis', **plot_args)
ax[-1].set(title='sample count', xlabel='Month')
ax[-1].coastlines()
ax[-1].set_global()
plt.suptitle('year = 2002')

# %% groupby time, lat and lon
dlat = 20
latitude_bins = np.arange(-90,90+dlat,dlat)
latitude_labels = latitude_bins[1:]-dlat/2

dlon = 20
longitude_bins = np.arange(0,360+dlon, dlon)
longitude_labels = longitude_bins[1:] - dlon/2

time_bins, mean, count = [], [], []
for t, t_data in ds.groupby(ds.time.dt.season):
    time_bins.append(t)
    longitude_lab, lat_lon_mean, lat_lon_count = [], [], []
    for lon, lon_t_data in t_data.groupby_bins(t_data.longitude, bins=longitude_bins, labels=longitude_labels):
        longitude_lab.append(lon)
        print(t, lon, len(lon_t_data.time))
        lat_lon_mean.append(lon_t_data.groupby_bins(lon_t_data.latitude, bins=latitude_bins, labels=latitude_labels).median('time', keep_attrs=True))
        # lat_lon_count.append(lon_t_data.groupby_bins(lon_t_data.latitude, bins=latitude_bins, labels=latitude_labels).count('time', keep_attrs=True))
        
    mean.append(xr.concat(lat_lon_mean, dim='longitude_bins').assign_coords(longitude_bins=longitude_lab).sortby('longitude_bins'))
#     count.append(xr.concat(lat_lon_count, dim='longitude_bins').assign_coords(longitude_bins=longitude_lab).sortby('longitude_bins'))
mean = xr.concat(mean, dim='time_bins').assign_coords(time_bins=time_bins).sortby('time_bins')
# count = xr.concat(count, dim='time_bins').assign_coords(time_bins=time_bins).sortby('time_bins')

#%% cartopy plot
# interpolation in the end array of longitude_bins
if mean.longitude_bins[-1]<360:
    mean = xr.auto_combine([mean, 
        mean.isel(longitude_bins=0).assign_coords(
            longitude_bins=360+dlon/2)])
    # count = xr.auto_combine([count, 
    #     count.isel(longitude_bins=0).assign_coords(
    #         longitude_bins=360+dlon/2)])

fig, ax = plt.subplots(3,4, figsize=(10,5), sharex=True,
        subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=180)) #ccrs.Orthographic(-80, 35)
                    )
plot_args = dict(x='longitude_bins', y='latitude_bins', 
                transform=ccrs.PlateCarree(),
                add_legend=False,
                add_colorbar=False
                )
parms = 'amplitude peak_height thickness'.split()
parm_lim = [(0,7e3), (81e3, 88e3), (0, 48e2)]
parm_lim = [dict(vmin=min, vmax=max) for min, max in parm_lim]
for i, par in enumerate(parms):
    for ti in range(len(mean.time_bins)):
        mean.isel(time_bins=ti)[par].plot.contourf(ax=ax[i, ti], **parm_lim[i], **plot_args)
        ax[i, ti].set(title=par+'\n'+mean.time_bins[ti].item(), xlabel='')
        ax[i, ti].coastlines()
        ax[i, ti].set_global()

#%%









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


