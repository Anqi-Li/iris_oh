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

# %% Y107 index
filename = '/home/anqil/Documents/osiris_database/composite_lya_index.nc'
with xr.open_dataset(filename) as ds_y107:
    ds_y107.irradiance.rolling(time=40, center=True).mean().plot(ax=plt.gca().twinx(), color='r')

# %% Open daily average Gauss parameters
am_pm = 'PM'
path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/gauss/{}/'.format(am_pm)
filename = 'gauss_{}_D_96_{}.nc'.format(am_pm, '{}')
years = list(range(2001,2018))
with xr.open_mfdataset([path+filename.format(y) for y in years]) as mds:
    mds = mds.reindex(latitude_bins=mds.latitude_bins[::-1]).load()

    mds['mean_peak_intensity'] *= 4*np.pi
    mds['mean_peak_intensity'].attrs['units'] = 'photons cm-3 s-1'
    mds['mean_peak_height'].attrs['units'] = 'm'
    mds['mean_peak_sigma'].attrs['units'] = 'm'
    mds['mean_thickness'] = 2*mds['mean_peak_sigma']
    mds['mean_thickness'].attrs['units'] = 'm'
    mds['std_thickness'] = 2*mds['std_peak_sigma']
    
    # low sample size data
    mds = mds.where(mds.count_sza>10)#.rolling(time=30, min_periods=10, center=True).mean('time')

    #shift NH 6 months
    mds_nh = mds.sel(latitude_bins=slice(20,80)).roll(time=180, roll_coords=False)
    mds_sh = mds.sel(latitude_bins=slice(-80,0))

# %% look at each month at one latiude
var = 'mean_apparent_solar_time'
test = mds[var].resample(time='QS-Dec').mean('time').sel(latitude_bins=0)
mean = test.assign_coords(
    year=test.time.dt.year, season=test.time.dt.season).set_index(
        time=['year', 'season']).unstack().mean('year')
unstacked = test.assign_coords(
    year=test.time.dt.year, season=test.time.dt.season).set_index(
        time=['year', 'season']).unstack() - mean
unstacked.plot(x='year', hue='season')#, row='latitude_bins')


# %% scatter plot
var1 = 'mean_peak_intensity'
var2 = 'mean_apparent_solar_time'
data1 = mds[var1].rolling(time=365, min_periods=90, center=True).mean('time')
data2 = mds[var2].rolling(time=365, min_periods=90, center=True).mean('time')

# da = data_running_mean.assign_coords(y=data_y107).swap_dims(dict(time='y'))
# poly_coeff = da.polyfit(dim='y', deg=1).polyfit_coefficients

fig, ax = plt.subplots(5, 2, figsize=(8,10), sharex=True, sharey=True,
            gridspec_kw=dict(hspace=0.5))
scatterplot_args = dict(y='y', ls='', marker='.', markersize=0.1)
for i, lat in enumerate(range(80,0,-20)):
    da = data1.sel(latitude_bins=lat).assign_coords(y=data2.sel(latitude_bins=lat)).swap_dims(dict(time='y'))
    da = da.where(~da.y.isnull(), drop=True)
    poly_coeff = da.polyfit(dim='y', deg=1).polyfit_coefficients
    da.plot.line(**scatterplot_args, ax=ax[i,0])
    xr.polyval(coord=da.y, coeffs=poly_coeff).plot(y='y', ax=ax[i,0])
    ax[i,0].set_title('{} - {} $^\circ$ N'.format(lat-10, lat+10))

    da = data1.sel(latitude_bins=-lat).assign_coords(y=data2.sel(latitude_bins=lat)).swap_dims(dict(time='y'))
    da = da.where(~da.y.isnull(), drop=True)
    poly_coeff = da.polyfit(dim='y', deg=1).polyfit_coefficients
    da.plot.line(**scatterplot_args, ax=ax[i,1])
    xr.polyval(coord=da.y, coeffs=poly_coeff).plot(y='y', ax=ax[i,1])
    ax[i,1].set_title('{} - {} $^\circ$ S'.format(lat-10, lat+10))

da = data1.sel(latitude_bins=0).assign_coords(y=data2.sel(latitude_bins=lat)).swap_dims(dict(time='y'))
da = da.where(~da.y.isnull(), drop=True)
poly_coeff = da.polyfit(dim='y', deg=1).polyfit_coefficients
da.plot.line(**scatterplot_args, ax=ax[-1,0])
xr.polyval(coord=da.y, coeffs=poly_coeff).plot(y='y', ax=ax[-1,0])
ax[-1,0].set_title('{} $^\circ$ N - {} $^\circ$ S'.format(10, 10))    

ax[-1,1].set_axis_off()
ax[-2,1].tick_params(labelbottom=True)
ax[-1,0].set_xlabel(var1)
ax[-2,1].set_xlabel(var1)
[ax[i,0].set_ylabel(var2) for i in range(5)]
[ax[i,1].set_ylabel('') for i in range(5)]

fig.suptitle('{} measurements'.format(am_pm), fontsize=16)

# %% scatter plot
var = 'mean_thickness'
data_running_mean = mds[var].rolling(time=365, min_periods=90, center=True).mean('time')
data_y107 = ds_y107.irradiance.rolling(time=365, center=True).mean('time').interp_like(mds).assign_attrs({'long_name':'Y10.7 index'})
da = data_running_mean.assign_coords(y=data_y107).swap_dims(dict(time='y'))
poly_coeff = da.polyfit(dim='y', deg=1).polyfit_coefficients

fig, ax = plt.subplots(5, 2, figsize=(8,10), sharex=True, sharey=True,
            gridspec_kw=dict(hspace=0.5))
scatterplot_args = dict(y='y', ls='', marker='o', markersize=0.1)
for i, lat in enumerate(range(80,0,-20)):
    da.sel(latitude_bins=lat).plot.line(**scatterplot_args, ax=ax[i,0])
    da.sel(latitude_bins=-lat).plot.line(**scatterplot_args, ax=ax[i,1])
    xr.polyval(coord=da.y, coeffs=poly_coeff.sel(latitude_bins=lat)).plot(y='y', ax=ax[i,0])
    xr.polyval(coord=da.y, coeffs=poly_coeff.sel(latitude_bins=-lat)).plot(y='y', ax=ax[i,1])
    ax[i,0].set_title('{} - {} $^\circ$ N'.format(lat-10, lat+10))
    ax[i,1].set_title('{} - {} $^\circ$ S'.format(lat-10, lat+10))

da.sel(latitude_bins=0).plot.line(**scatterplot_args, ax=ax[-1,0])
xr.polyval(coord=da.y, coeffs=poly_coeff.sel(latitude_bins=0)).plot(y='y', ax=ax[-1,0])
ax[-1,0].set_title('{} $^\circ$ N - {} $^\circ$ S'.format(10, 10))    

ax[-1,1].set_axis_off()
ax[-2,1].tick_params(labelbottom=True)
ax[-1,0].set_xlabel(var)
ax[-2,1].set_xlabel(var)
[ax[i,0].set_ylabel('Y10.7 index') for i in range(5)]
[ax[i,1].set_ylabel('') for i in range(5)]

fig.suptitle('{} measurements'.format(am_pm), fontsize=16)

# %% line plot
var = 'mean_sza'
data_running_mean = mds[var].rolling(time=1, min_periods=1, center=True).mean('time')
data_y107 = ds_y107.irradiance.rolling(time=27, center=True).mean('time').interp_like(mds).assign_attrs({'long_name':'Y10.7 index'})

fig, ax = plt.subplots(5, 2, figsize=(8,10), sharex=True, sharey=True,
            gridspec_kw=dict(hspace=0.5))
dataplot_args = dict(x='time', color='C0')

def plot_y107(ax, ticks=True):
    y107plot_args = dict(x='time', color='r', alpha=0.3)
    a_t = ax.twinx()
    data_y107.plot(ax=a_t, **y107plot_args)
    if ticks:
        a_t.tick_params(colors=y107plot_args['color'])
    else:
        a_t.tick_params(labelright=False)
        a_t.set_ylabel('')

for i, lat in enumerate(range(80,0,-20)):
    data_running_mean.sel(latitude_bins=lat).plot(ax=ax[i,0], **dataplot_args)
    plot_y107(ax[i,0], ticks=False)
    ax[i,0].set(ylabel='', xlabel='', title='{} - {} $^\circ$ N'.format(lat-10, lat+10))
    data_running_mean.sel(latitude_bins=-lat).plot(ax=ax[i,1], **dataplot_args)
    plot_y107(ax[i,1])
    ax[i,1].set(ylabel='', xlabel='', title='{} - {} $^\circ$ S'.format(lat-10, lat+10)) 

data_running_mean.sel(latitude_bins=0).plot(ax=ax[-1,0], **dataplot_args)
plot_y107(ax[-1,0])
ax[-1,0].set(ylabel='', xlabel='', title='{} $^\circ$ S - {} $^\circ$ N'.format(10, 10))

ax[-1,1].set_axis_off()
ax[-2,1].tick_params(labelbottom=True, labelrotation=30)
ax[-1,0].set_xlabel('Time')
ax[-2,1].set_xlabel('Time')
# [ax[i,0].set_ylabel('Peak Intensity') for i in range(5)]
# [ax[i,0].set_ylim(17, 19.5) for i in range(1,5)]
[ax[i,0].set_ylabel(var) for i in range(5)]
[ax[i,1].set_ylabel('') for i in range(5)]

fig.suptitle('{} measurements'.format(am_pm), fontsize=16)

#%% Countourf plot
var = 'mean_peak_intensity'
mds[var].rolling(time=11, min_periods=1, center=True).mean('time').plot.contourf(
    x='time', y='latitude_bins', figsize=(15,3), robuts=True)
# %% Line plot
var = 'mean_peak_intensity'
fc = mds[var].rolling(time=360, min_periods=50, center=True).mean('time').plot.line(
    x='time', row='latitude_bins', figsize=(5,15))

ax_twins = []
for i in range(len(fc.axes)):
    ax = fc.axes[i][0].twinx()
    l_y107, = ds_y107.irradiance.rolling(time=40, center=True).mean().plot(ax=ax, color='r', alpha=0.3)
    ax.tick_params(colors=l_y107.get_color(), )
    ax_twins.append(ax)

# %% histogram plot
from xhistogram.xarray import histogram
var = 'mean_peak_intensity'
bins = [np.linspace(8e4, 13e4, 50), np.linspace(6e-3, 8e-3, 50)]
data_running_mean = mds[var].rolling(time=365, min_periods=100, center=True).mean('time')
data_y107 = ds_y107.irradiance.interp_like(mds).rolling(time=1, center=True).mean('time')

fig, ax = plt.subplots(5, 2, figsize=(8,10), sharex=True, sharey=True)
for i, lat in enumerate(range(80,0,-20)):
    histogram(data_running_mean.sel(latitude_bins=lat), data_y107,
        bins=bins).plot(ax=ax[i, 0])
    histogram(data_running_mean.sel(latitude_bins=-lat), data_y107,
        bins=bins).plot(ax=ax[i, 1])
histogram(data_running_mean.sel(latitude_bins=0), data_y107,
    bins=bins).plot(ax=ax[-1, 0])


# %% monthly mean differences
var = 'mean_peak_intensity'
test = mds[var].rolling(#.resample(
        time=700, min_periods=90, center=True
        # time='1M'
    ).mean(
        'time'
    ).assign_coords(
        dict(year=mds.time.dt.year, month=mds.time.dt.month)
    )
mean = test.groupby('month').mean('time')
diff = test.groupby('month') - mean
fc = diff.plot.line(x='time', row='latitude_bins', figsize=(8,15))
for i in range(len(mds.latitude_bins)):
    ds_y107.irradiance.rolling(time=40, center=True).mean().interp(
        time=mds.time
    ).plot(
        ax=fc.axes[i][0].twinx(), color='r', alpha=0.3)

# %%
data_y107 = ds_y107.irradiance.rolling(time=700, center=True).mean('time').interp_like(mds).assign_attrs({'long_name':'Y10.7 index'})
da = diff.assign_coords(y=data_y107).swap_dims(dict(time='y'))
poly_coeff = da.polyfit(dim='y', deg=1).polyfit_coefficients

fig, ax = plt.subplots(5, 2, figsize=(8,10), sharex=True, sharey=True,
            gridspec_kw=dict(hspace=0.5))
scatterplot_args = dict(y='y', ls='', marker='o', markersize=0.1)
for i, lat in enumerate(range(80,0,-20)):
    da.sel(latitude_bins=lat).plot.line(**scatterplot_args, ax=ax[i,0])
    da.sel(latitude_bins=-lat).plot.line(**scatterplot_args, ax=ax[i,1])
    xr.polyval(coord=da.y, coeffs=poly_coeff.sel(latitude_bins=lat)).plot(y='y', ax=ax[i,0])
    xr.polyval(coord=da.y, coeffs=poly_coeff.sel(latitude_bins=-lat)).plot(y='y', ax=ax[i,1])
    ax[i,0].set_title('{} - {} $^\circ$ N'.format(lat-10, lat+10))
    ax[i,1].set_title('{} - {} $^\circ$ S'.format(lat-10, lat+10))

da.sel(latitude_bins=0).plot.line(**scatterplot_args, ax=ax[-1,0])
xr.polyval(coord=da.y, coeffs=poly_coeff.sel(latitude_bins=0)).plot(y='y', ax=ax[-1,0])
ax[-1,0].set_title('{} $^\circ$ N - {} $^\circ$ S'.format(10, 10))    

ax[-1,1].set_axis_off()
ax[-2,1].tick_params(labelbottom=True)
# ax[-1,0].set_xlabel(var)
# ax[-2,1].set_xlabel(var)
[ax[i,0].set_ylabel('Y10.7 index') for i in range(5)]
[ax[i,1].set_ylabel('') for i in range(5)]

fig.suptitle('{} measurements'.format(am_pm), fontsize=16)



