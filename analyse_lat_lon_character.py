#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from xhistogram.xarray import histogram
import cartopy.crs as ccrs
import glob 
import pandas as pd

#%% open OH daily climatology files - NP SP plots
am_pm = 'PM'
min_sza = 96
d_m = 'daily'
path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/'
filename = '/{2}/{0}/{1}_{0}_{2}_ver_clima_{3}.nc'.format(min_sza, am_pm, d_m, '{}') #'daily_ver_clima_{}.nc'
filename = '/daily_ver_clima_{}.nc'
with xr.open_mfdataset(path+filename.format('*')) as mds:
    mds = mds.assign_coords(z=mds.z*1e-3)
    mds.z.attrs['units'] = 'km'

    mds = mds.roll(time=0, roll_coords=False).assign_coords(
        dict(year=mds.time.dt.year, doy=mds.time.dt.dayofyear)).set_index(
            time=['year', 'doy']).unstack()
    # mds.mean_ver.sel(latitude_bins=80, z=80, year=slice(2001,2009)).plot(
    #     x='doy', hue='year')

    # mds.mean_ver.sel(latitude_bins=0).plot(
    #     x='doy', y='z', row='year', ylim=(75,90), vmin=0,
    #     robust=True, cmap='viridis', figsize=(10,15)
    # )

    mds.mean_ver.sel(latitude_bins=0, z=85).plot.line(
        x='doy', row='year', ylim=(3e3, 6e3), figsize=(10,15)
    )
#%% OH daily and resample monthly means
am_pm = 'PM'
min_sza = 96
d_m = 'daily'
path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/'
filename = '{2}/{0}/{1}_{0}_{2}_ver_clima_{3}.nc'.format(min_sza, am_pm, d_m, '{}')
with xr.open_mfdataset(path+filename.format('*')) as mds:
    mds = mds.assign_coords(z=mds.z*1e-3)
    mds.z.attrs['units'] = 'km'
    mds = mds.reindex(
        latitude_bins=mds.latitude_bins[::-1])
    
    # mds.where(mds.count_ver>100).resample(time="1M").mean(
    #     ).mean_ver.plot.contourf(
    #         x='time', y='z', row='latitude_bins', ylim=(75,95),
    #         vmin=0, vmax=4865, robust=True, cmap='viridis', figsize=(10,10)
    #         )
    # mds.where(mds.count_sza>100).resample(time="1M").mean(
    #     ).mean_sza.plot.line(
    #         ylim=(17.5, 22),
    #         x='time', row='latitude_bins', figsize=(10,10)
    #         )
    xr.Dataset(dict(
        upper=mds.mean_sza.where(mds.count_sza>100) + mds.std_sza, 
        lower=mds.mean_sza.where(mds.count_sza>100) - mds.std_sza)
        ).to_array(
            dim='bounds', name='mean_sza'
            ).plot.line(
                x='time', row='latitude_bins', hue='bounds', 
                figsize=(10,10))
    plt.suptitle('{}, SZA>{}'.format(am_pm, min_sza), x=0.9, y=1, ha='right')
    plt.show()
# %% solar radio flux F10.7 index
path = '/home/anqil/Documents/osiris_database/'
filename = 'f107_index.nc'
with xr.open_dataset(path+filename) as ds_f107:
    ds_f107.f107.rolling(time=40, center=True).mean().plot()
    # print(ds_f107)

filename = 'composite_lya_index.nc'
with xr.open_dataset(path+filename) as ds_y107:
    ds_y107.irradiance.rolling(time=40, center=True).mean().plot(ax=plt.gca().twinx(), color='r')
    # print(ds_y107)
#%% open OH climatology monthly files 
path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/monthly/'
filename = 'All_96_monthly_ver_clima_{}.nc'
with xr.open_mfdataset(path+filename.format('*')) as mds:
    mds = mds.assign_coords(z=mds.z*1e-3)
    mds.z.attrs['units'] = 'km'
    fc = mds.mean_ver.where(mds.count_ver>0).reindex(
        latitude_bins=mds.latitude_bins[::-1]).sel(
        latitude_bins=slice(80,-80)).plot(
        y='z', x='time', row='latitude_bins', ylim=(72, 95),
        vmin=0, cmap='viridis', robust=False, figsize=(10, 10), 
        add_colorbar=True,
        # cbar_kwargs=dict(label='[photon cm-3 s-1]', location='right')
#        xticks=pd.date_range(start='2001', end='2018', freq='Y'),
        )
    # plt.xticks(pd.date_range(start='2001', end='2018', freq='Y'), range(2002,2019),
    #     va='top', ha='left', rotation=0,
        # )
    # for ax in range(len(fc.axes)):
    #     fc.axes[ax,0].grid()

    # ds_f107.f107.sel(time=slice('2002', '2017')).rolling(time=30).mean().plot(
    #     ax=fc.axes[1,0].twinx(), color='r', alpha=0.6)

#%% open agc statistics files -- time_lat
path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/'
filename = 'gauss_clima_{}.nc'
def set_idx(ds, year):
    idx = [np.datetime64('{}-{}'.format(year, str(month).zfill(2))) for month in ds.time_bins.values] 
    ds = ds.assign_coords(time_bins=idx)
    return ds

# mds = []
# for year in range(2001,2018):
#     with xr.open_dataset(path+filename.format(year)) as ds:
#         mds.append(set_idx(ds, year))
# mds = xr.merge(mds)

with xr.open_mfdataset(path+filename.format('*')) as mds:
    # print(mds)
    #% some adjustments on the longterm dataset
    mds = mds.rename(dict(latitude_bins='latitude', time='time'))
    mds.latitude.attrs['units'] = 'deg N'

data_vars_lst = 'peak_intensity peak_height thickness'.split()
data_vars_mul = [1, 1e-3, 1e-3]
data_vars_units = 'pho_cm-1_s-1 km km'.split()
for i in range(len(data_vars_lst)):
    for s in 'mean_{} std_{}'.split():
        mds[s.format(data_vars_lst[i])] *= data_vars_mul[i]
        mds[s.format(data_vars_lst[i])].attrs['units'] = data_vars_units[i] 


#%% Longterm contourf plot
vmin_lst = [1e3, 80, 2.5]
vmax_lst = [7e3, 85, 5]
fig, ax = plt.subplots(len(data_vars_lst),1, figsize=(10,6), sharex=True, sharey=True)
contourf_args = dict(x='time', robust=True)
for i, var in enumerate(['mean_{}'.format(v) for v in data_vars_lst]):
    mds[var].rename(data_vars_lst[i]).sel(latitude=slice(-50,50)).plot.contourf(
        ax=ax[i], #vmin=vmin_lst[i], vmax=vmax_lst[i], 
        **contourf_args)
[ax[i].set(xlabel='', title=data_vars_lst[i]) for i in range(len(data_vars_lst))]
# mds.count_amplitude.rename('num. of sample').plot.contourf(ax=ax[-1], vmax=8e4, **contourf_args)
# ax[-1].set(title='Sample Count')
ax[0].set(title='peak_intensity')

ax_f107 = ax[0].twinx()
p = ds_f107.f107.sel(time=slice('2001','2018')).rolling(time=40, center=True).mean().plot(ax=ax_f107, color='r')

#%% Longterm line plot
fig, ax = plt.subplots(5, 1, figsize=(7,10), sharex=True, sharey=True)
var = data_vars_lst[0]
line_args = dict(hue='latitude', x='time', ylim=(0,8e3)) #ylim=(80,87))#, 
for lat_bin_idx in range(4):
    mds['mean_'+var].isel(latitude=[lat_bin_idx,-(lat_bin_idx+1)]).plot.line(
        ax=ax[lat_bin_idx], **line_args)

mds['mean_'+var].isel(latitude=4).plot.line(
    ax=ax[4], **line_args)
    
for i in range(5):    
    ax_f107 = ax[i].twinx()
    ds_f107.f107.sel(time=slice('2001','2018')).rolling(
        time=40, center=True).mean().plot(
            ax=ax_f107, color='k', alpha=0.5, ylim=(50,250))

#%% Seasonal line plot
fig, ax = plt.subplots(5, 1, figsize=(7,10), sharex=True, sharey=True)
var = data_vars_lst[0]
line_args = dict(x='time', alpha=0.5)
for year in range(2002,2018):
    ds_yearly = mds['mean_'+var].rename(var).sel(time=str(year))
    for lat_bin_idx in range(4):
        line_north, = ds_yearly.assign_coords(
            time=ds_yearly.time.dt.dayofyear).roll(time=180, roll_coords=False).isel(
                latitude=-(lat_bin_idx+1)).plot.line(
                    **line_args, label=year, color='r', ax=ax[lat_bin_idx])
        mds['mean_'+var].rename(var).roll(time=180, roll_coords=False).isel(
            latitude=-(lat_bin_idx+1)).groupby(
                mds.time.dt.dayofyear).mean(...).plot.line(
                    x='dayofyear', color='k', ls='--', ax=ax[lat_bin_idx])

        line_south, = ds_yearly.assign_coords(
            time=ds_yearly.time.dt.dayofyear).roll(time=0, roll_coords=False).isel(
                latitude=lat_bin_idx).plot.line(
                    **line_args, label=year, color='b', ax=ax[lat_bin_idx])
        mds['mean_'+var].rename(var).roll(time=0, roll_coords=False).isel(
            latitude=lat_bin_idx).groupby(
                mds.time.dt.dayofyear).mean(...).plot.line(
                    x='dayofyear', color='k', ls='--', ax=ax[lat_bin_idx])
    # equator band
    ds_yearly.assign_coords(time=ds_yearly.time.dt.dayofyear).isel(
        latitude=4).plot.line(
            **line_args, label=year, color='k', ax=ax[-1], 
            ylim=(.5e3, 7.5e3), #peak intensity
            # ylim=(78, 90), #peak height in km
            # ylim=(1, 8), #thickness in km
            )
    mds['mean_'+var].rename(var).isel(
            latitude=4).groupby(
                mds.time.dt.dayofyear).mean(...).plot.line(
                    x='dayofyear', color='k', ls='--', ax=ax[-1])

[ax[i].set(xlabel='') for i in range(4)]
# [ax[i].set(ylabel='Peak Intensity \n [photon cm-1 s-1]') for i in range(5)]
ax[-1].set(xlabel='dayofyear-eq to S. hemsphere',
            # xticks = [1, 4, 7, 10],
            )
[ax[i].legend([line_north, line_south], ['N', 'S'], loc='upper right') for i in range(4)]

#% plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')











#%% check outliers
filename = 'filterin_{}.nc'
par_lst = 'amplitude thickness peak_height residual all'.split()
lat_bins = np.arange(-90, 100, 20)
hist = hist_amplitude, hist_thickness, hist_peak_height, hist_residual, hist_all = [], [], [], [], []
year_month = [[], [], [], [], []]
for year in range(2001,2018):
    print(year)
    with xr.open_dataset(path+filename.format(year)) as cond:
        # h_amplitude, _ = cond.latitude.where(~cond.cond_amplitude).pipe(np.histogram, bins=lat_bins)
        # h_thickness, _ = cond.latitude.where(~cond.cond_thickness).pipe(np.histogram, bins=lat_bins)
        # h_peak_height, _ = cond.latitude.where(~cond.cond_peak_height).pipe(np.histogram, bins=lat_bins)
        # h_residual, _ = cond.latitude.where(~cond.cond_residual).pipe(np.histogram, bins=lat_bins) 
        # h_all, _ = cond.latitude.where(~(
        #     cond.cond_residual * cond.cond_peak_height * cond.cond_thickness * cond.cond_amplitude)).pipe(
        #         np.histogram, bins=lat_bins)
        
        # hist_amplitude.append(h_amplitude)
        # hist_thickness.append(h_thickness)
        # hist_peak_height.append(h_peak_height)
        # hist_residual.append(h_residual)
        # hist_all.append(h_all)

        cond = cond.update({'cond_all': cond.cond_residual * cond.cond_peak_height * cond.cond_thickness * cond.cond_amplitude})
        for i, par in enumerate(par_lst):
            for month, group in cond.latitude.where(~cond['cond_'+par]).groupby(cond.time.dt.month):
                # h, _ = group.pipe(np.histogram, bins=lat_bins)
                h = group.groupby_bins(group.latitude, lat_bins).count(...)
                hist[i].append(h)
                year_month[i].append(np.datetime64('{}-{}'.format(year, str(month).zfill(2))))

hist = xr.Dataset({'hist_'+par: (('time', 'latitude'), hist[i]) for i, par in enumerate(par_lst)}).assign_coords(time=year_month[0], latitude=lat_bins[:-1]+10)
#%% histogram outlier
fig, ax = plt.subplots(len(par_lst),1, figsize=(10,8), sharex=True, sharey=True)
for i, par in enumerate(par_lst):
    hist['hist_'+par].pipe(lambda x: x/mds.count_amplitude * 100).plot(x='time', ax=ax[i], robust=True, cbar_kwargs=dict(label='%'))
    ax[i].set(title=par, xlabel='')

#%% open statistics files -- time_lat_lon
filename = 'time_lat_lon_{}.nc'
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

data_vars_lst = 'amplitude peak_height thickness residual'.split()
data_vars_mul = [1, 1e-3, 1e-3, 1]
data_vars_units = 'pho_cm-1_s-1 km km pho_cm-1_s-1'.split()
for i in range(len(data_vars_lst)):
    for s in 'mean_{} std_{}'.split():
        mds[s.format(data_vars_lst[i])] *= data_vars_mul[i]
        mds[s.format(data_vars_lst[i])].attrs['units'] = data_vars_units[i] 

# data_vars_lst = 'max_pw_freq max_pw amplitude peak_height thickness'.split()
# data_vars_mul = [1e3, 1, 1, 1e-3, 1e-3]
# data_vars_units = 'km-1 ? pho_cm-1_s-1 km km'.split()
# for i in range(len(data_vars_lst)):
#     for s in 'mean_{} std_{}'.split():
#         mds[s.format(data_vars_lst[i])] *= data_vars_mul[i]
#         mds[s.format(data_vars_lst[i])].attrs['units'] = data_vars_units[i] 

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
year = [2002, 2003, 2004]
for var in ['mean_{}'.format(v) for v in data_vars_lst]:
    fig = plt.figure(figsize=(10,5))
    p = mds.sel(year=year).reindex(season=time_seq)[var].plot.contourf(**contourf_args)
    for ax in p.axes.flat:
        ax.coastlines()
        ax.set_global()
    plt.suptitle(var)
    plt.show()

#%% CMAM OH file
filename = '/home/anqil/Documents/Python/iris-oh/ntveoh62_monChem_CMAM-Ext_CMAM30-SD_r1i1p1_197901-201006.nc'
with xr.open_dataset(filename) as ds:
    # print(ds)
    ds.ntveoh62.sel(lat=slice(-90, 90), time=slice('2000', '2010')).mean(
        dim=['lon', 'time'], keep_attrs=True).plot(
        x='lat', y='plev', yscale='log')
    plt.gca().invert_yaxis()

































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


