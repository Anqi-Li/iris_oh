#%%
import numpy as np
import xarray as xr
import glob
from os import listdir
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, NoNorm, Normalize
from scipy.interpolate import interp1d
import pandas as pd

# %% Open daily average Gauss parameters
am_pm = 'PM'
# path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/gauss/{}/'.format(am_pm)
# filename = 'gauss_{}_D_96_{}.nc'.format(am_pm, '{}')
path = '~/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/averages/zenith/'
filename = '{}_daily_zonal_mean_{}.nc'.format(am_pm, '{}')
years = list(range(2001,2016))
with xr.open_mfdataset([path+filename.format(y) for y in years]) as mds:
    mds = mds.rename({'latitude_bin': 'latitude_bins'})
    mds = mds.reindex(latitude_bins=mds.latitude_bins[::-1]).load()
    mds = mds.assign_coords(latitude_bins=mds.latitude_bins.astype(int),
                            z = mds.z*1e-3, #m -> km
                            )
    
    # low sample size data
    mds = mds.where(mds.count_sza>50)

    # croasen the sample frequency
    # mds = mds.rolling(time=30, min_periods=10, center=True).mean('time')
    # mds = mds.resample(time='1Y').mean('time')

    #shift NH 6 months
    mds_nh = mds.sel(latitude_bins=slice(20,80)).roll(time=180, roll_coords=False)
    mds_sh = mds.sel(latitude_bins=slice(-80,0))
#%% observational pattern: sza, lst, etc.
# with plt.style.context('ggplot'):

gridspec_kw = dict(height_ratios=[.8,1,1,1, .1], hspace=.4)
fig, ax = plt.subplots(5, 1, sharex=False, figsize=(6,11), 
    gridspec_kw=gridspec_kw)

# for i in range(len(mds.latitude_bins)):
#     mds.count_sza.resample(time='1Y', loffset='-6M').sum('time').isel(latitude_bins=i).plot.step(
#         ax=ax[0], x='time', where='mid')
mds.count_sza.resample(time='1Y', loffset='-6M').sum('time').sum('latitude_bins').plot(
    ax=ax[0], drawstyle='steps-mid', fillstyle='bottom', c='k')
cbar_kwargs = dict(shrink=0.9, orientation='horizontal') #location='top')#, 
contourf_args = dict(cmap='viridis_r', robust=True, x='time', y='latitude_bins', add_colorbar=True, cbar_kwargs=cbar_kwargs, )
lines_args = dict(x='time', hue='latitude_bins')
lst = mds.mean_apparent_solar_time.resample(time='1Y', loffset='-6M').mean('time')
lst.sel(latitude_bins=slice(90,-90)).plot.line(
    ax=ax[1], add_legend=True, **lines_args)
# lst.sel(latitude_bins=[80, 60, -60, -80]).plot.line(
#     ax=ax[2], add_legend=True, **lines_args)
# lst.where(abs(lst.latitude_bins)>40).plot.line(
#     ax=ax[1], add_legend=True, **lines_args)
# lst.where(abs(lst.latitude_bins)<=40).plot.line(
#     ax=ax[2], add_legend=False, **lines_args)
mds.mean_sza.resample(time='1Y', loffset='-6M').mean('time').plot.line(
    ax=ax[2], add_legend=False, **lines_args)
mds.mean_sza.resample(time='1M', loffset='-15D').mean('time').plot.contourf(
    ax=ax[3], cbar_ax=ax[-1], **contourf_args)
# mds.mean_sza.resample(time='1Y').mean('time').plot.line(ax=ax[2], x='time', hue='latitude_bins', add_legend=False)

l = ax[1].get_legend()
l.set_bbox_to_anchor((1, 1.05))
l.set_title('Latitude bins \n($\pm$ 10 deg N)')
# l = ax[2].get_legend()
# l.set_bbox_to_anchor((1, 1.05))
# l.set_title('Latitude bins \n($\pm$ 10 deg N)')

ax[0].set(ylabel='', title='Num. of nightglow profile per year', xlabel='', xticklabels='')
ax[1].set(ylabel='[hour]', title='Annually ave. LST', xlabel='', xlim=ax[0].get_xlim(), xticklabels='')
# ax[2].set(ylabel='[hour]', title='LST', xlabel='', xlim=ax[0].get_xlim(), xticklabels='')
ax[2].set(ylabel='SZA [deg]', title='Annually ave. SZA', xlabel='', xlim=ax[0].get_xlim(), xticklabels='')
ax[3].set(ylabel='Latitude [deg N]', title='Monthly ave. SZA', xlabel='', ylim=(-85,85), xlim=ax[0].get_xlim())
ax[-1].set(xlabel='SZA [deg]')

for i, s in enumerate('a) b) c) d)'.split()):
    ax[i].text(.015, .87, s,
        # backgroundcolor='w', 
        fontweight='bold', 
        transform=ax[i].transAxes)

# %% load lya index
filename = '/home/anqil/Documents/osiris_database/composite_lya_index.nc'
with xr.open_dataset(filename) as ds_y107:
    # ds_y107.irradiance.rolling(time=40, center=True).mean().plot(ax=plt.gca().twinx(), color='r')
    ds_y107 = ds_y107.interp_like(mds)
ds_lya = ds_y107.copy()
ds_lya['irradiance'] *= 1e3
ds_lya['irradiance'] = ds_lya['irradiance'].assign_attrs(dict(units='$10^{-3} Wm^{-2}$'))

# %% time series
var = 'mean_ver'
var_label = 'Annually ave. VER'
# var_units = '$cm^{-3} s^{-1}$'
data_oh = mds[var].resample(time='1Y', loffset='-6M').mean('time', keep_attrs=True)
data_lya = ds_lya.irradiance.resample(time='1Y', loffset='-6M').mean('time', keep_attrs=True)

lat_seq = [-80, 80, -60, 60, -40, 40, -20, 20, 0]
fc = data_oh.reindex(
    latitude_bins=lat_seq).plot.contourf(
        figsize=(9,9),
        x='time', col='latitude_bins', col_wrap=2, 
        cmap='viridis', 
        # vmin=-50, vmax=50, #robust=True, 
        vmin=0, #levels=np.arange(0,12e4, 2e4),
        ylim=(75,95), 
        cbar_kwargs=dict(shrink=0.8, anchor=(0, 0.9)),
        )
fc.cbar.set_label('{} [{}]'.format(var_label, data_oh.units))
fc.set_axis_labels('Year', 'Altitude [km]')
pos = fc.axes[-1,1].get_position()
twin_ax = fc.axes[-1,1].twinx()
data_lya.plot(ax=twin_ax, color='r')
twin_ax.set_ylabel('[{}]'.format(data_lya.units))
fc.axes[-1,1].set_visible(True)
fc.axes[-1,1].set_position(pos)
fc.axes[-1,1].tick_params(labelbottom=True, labelrotation=30)
fc.axes[-1,1].set_xlabel(fc.axes[-1,0].get_xlabel())

for j, sn in enumerate('S N'.split()):
    for i, lat in enumerate(np.linspace(80,0,num=len(fc.axes)).astype(int)):
        fc.axes[i,j].set_title('{} - {} $^\circ$ {}'.format(lat-10, lat+10, sn))
fc.axes[-1,1].set_title('Composite Lyman-alpha')

#%% Anomaly correlation
def cal_anual_anomaly(da):
    mds_doy = da.assign_coords(
        dict(year=da.time.dt.year, doy=da.time.dt.dayofyear)).set_index(
        time=['year', 'doy']).unstack()
    anual_mean = mds_doy.pipe(lambda x: ((x-x.mean('year'))/x.mean('year')*100).mean('doy')
        ).assign_coords(year=np.array(mds_doy.year.astype('str'), dtype='datetime64')
        ).rename(dict(year='time')
        ).assign_attrs(dict(units='%')
        )
    anual_std = mds_doy.pipe(lambda x: ((x-x.mean('year'))/x.mean('year')*100).std('doy')
        ).assign_coords(year=np.array(mds_doy.year.astype('str'), dtype='datetime64')
        ).rename(dict(year='time')
        ).assign_attrs(dict(units='%')
        )
    return anual_mean, anual_std

#%% Pablo model
path_model = '/home/anqil/Documents/Python/Photochemistry/Pablo/'
path_model += 'results_nc/initial_OO3_OOM_zerodO2dt_ztop200_Jnight/'
with xr.open_dataset(path_model+'model_day.nc') as model_day:
    # print(model_day)
    # model_day['lat'] = model_day.lat.assign_attrs(units='deg N')
    model_day = model_day.assign(
        OH_v_zenith=model_day.OH_v.sel(z=slice(70,105)).integrate('z')*1e5, #cm-2
        lat=model_day.lat.assign_attrs(units='deg N')
    )
    fc = model_day.OH_v_zenith.rename(
        dict(month='Month', lat='Latitude')
    ).pipe(
        lambda x: x/x.mean(['t', 'Latitude'])
    ).plot.line(
        x='t', hue='Latitude', col='Month', col_wrap=4
        )
    fc.set_axis_labels('LST [hour]', 'Norm. OH*')

#%%
oh_model_odin = model_day.OH_v_zenith.pipe(
    lambda x: x/x.mean(['t', 'lat'])
).assign_coords(
    # month = np.linspace(1,365.25,12)
).interp(
    t=mds.mean_apparent_solar_time,
    lat=mds.latitude_bins, 
    month=mds.time.dt.month,#dayofyear,
    ).rename(
        'Modeled OH*'
    )

da = oh_model_odin.copy()
# da = mds.mean_zenith_intensity.copy()
da.assign_coords(
        dict(year=da.time.dt.year, doy=da.time.dt.dayofyear)).set_index(
        time=['year', 'doy']).unstack().pipe(
            # lambda x: ((x-x.mean('year'))/x.mean('year')*100)
            lambda x: x.mean('year')
            ).plot.contourf(
                x='doy', y='latitude_bins', #col='year', 
                robust=True, figsize=(10,4));

#%%  anomaly correlation with lya plot
# var = 'mean_peak_height' 
# var_label = 'Height anomaly' 
# var = 'mean_zenith_intensity'
# var_label = 'ZER anomaly'
# data_oh, error_oh = [x.rename(var_label) for x in cal_anual_anomaly(mds[var])]
data_oh, error_oh = cal_anual_anomaly(oh_model_odin)
data_lya, error_lya = [x.rename('Ly-a anomaly') for x in cal_anual_anomaly(ds_lya.irradiance)]
# data_oh, error_oh = (oh_model_odin.resample(time='1Y', loffset='1D').mean('time', keep_attrs=True).assign_attrs(dict(units='')),
#                     oh_model_odin.resample(time='1Y', loffset='1D').std('time', keep_attrs=True).assign_attrs(dict(units='')))
# data_lya, error_lya = (ds_lya.irradiance.resample(time='1Y', loffset='1D').mean('time', keep_attrs=True),
#                     ds_lya.irradiance.resample(time='1Y', loffset='1D').std('time', keep_attrs=True))

da = data_oh.assign_coords(lya=data_lya).swap_dims(dict(time='lya'))
pf = da.polyfit(dim='lya', deg=1, cov=True)
poly_coeff = pf.polyfit_coefficients
poly_deg1_error = pf.polyfit_covariance.sel(cov_i=0, cov_j=0).pipe(np.sqrt)
poly_deg1 = poly_coeff.sel(degree=1)
# scatter plots
lat_seq = [-80, 80, -60, 60, -40, 40, -20, 20, 0]
fc = da.reindex(
    latitude_bins=lat_seq).plot.line(
        figsize=(8,9),
        y='lya', col='latitude_bins', col_wrap=2, 
        ls='-', marker='o', markersize=5,
    )
for j in range(fc.axes.shape[1]):
    for i in range(fc.axes.shape[0]):
        if (i+1,j+1) == fc.axes.shape:
            pass 
        else:
            # error bars
            fc.axes[i,j].errorbar(
                da.sel(**fc.name_dicts[i,j]), da.lya, 
                xerr=error_oh.sel(**fc.name_dicts[i,j]),
                yerr=error_lya,
                ecolor='C0', alpha=0.5, ls='', capsize=3,
            )
            # slopes
            xr.polyval(
                coord=da.lya, coeffs=poly_coeff.sel(**fc.name_dicts[i,j])
                ).plot(
                    y='lya', ax=fc.axes[i,j], 
                    color='C3', ls='-', lw=1,
                    label='s = {}\n$\pm${} ({}%)'.format(
                        poly_deg1.sel(**fc.name_dicts[i,j]).values.round(2),
                        poly_deg1_error.sel(**fc.name_dicts[i,j]).values.round(3),
                        abs(poly_deg1_error/poly_deg1*100).sel(**fc.name_dicts[i,j]).values.astype(int),
                    ),
                )
            fc.axes[i,j].legend(handlelength=0.5, frameon=False)
            if j==1:
                fc.axes[i,j].set_ylabel('')

fc.set_axis_labels('{} [{}]'.format(data_oh.name, data_oh.units),
                    '{} [{}]'.format(data_lya.name, data_lya.units))
fc.axes[-2,1].tick_params(labelbottom=True, labelrotation=0)
fc.axes[-2,1].set_xlabel(fc.axes[-1,0].get_xlabel())
for j, sn in enumerate('S N'.split()):
    for i, lat in enumerate(np.linspace(80,0,num=len(fc.axes)).astype(int)):
        fc.axes[i,j].set_title('{} - {} $^\circ$ {}'.format(lat-10, lat+10, sn))
# fc.axes[-1,0].ticklabel_format(axis='x', style='sci', scilimits=(3,3))

#%% Adams request
plt.figure()
xr.polyval(
    coord=da.lya, coeffs=poly_coeff#.sel(**fc.name_dicts[i,j])
    ).plot.line(
        y='lya', hue='latitude_bins', #ax=fc.axes[i,j], 
        # color='C3', ls='-', lw=1,
    );
# plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(3,3))
plt.gca().set_xlabel('{} [{}]'.format(data_oh.name, data_oh.units))

#%%
def cal_poly_deg1(data_oh, data_lya):
    da = data_oh.assign_coords(lya=data_lya).swap_dims(dict(time='lya'))
    pf = da.polyfit(dim='lya', deg=1, cov=True)
    poly_coeff = pf.polyfit_coefficients
    poly_deg1_error = pf.polyfit_covariance.sel(cov_i=0, cov_j=0).pipe(np.sqrt)
    poly_deg1 = poly_coeff.sel(degree=1)
    return poly_deg1, poly_deg1_error

data_lya = cal_anual_anomaly(ds_lya.irradiance)[0]
poly_deg1_zer, poly_deg1_error_zer = cal_poly_deg1(cal_anual_anomaly(mds.mean_zenith_intensity)[0], data_lya)
poly_deg1_height, poly_deg1_error_height = cal_poly_deg1(cal_anual_anomaly(mds.mean_peak_height)[0], data_lya)
poly_deg1_model, poly_deg1_error_model = cal_poly_deg1(cal_anual_anomaly(oh_model_odin)[0], data_lya)

plt.figure()
plt.errorbar(poly_deg1_height.latitude_bins, poly_deg1_height*10, yerr=poly_deg1_error_height*10, label='Height IRI x10', c='C3')
plt.errorbar(poly_deg1_zer.latitude_bins, poly_deg1_zer, yerr=poly_deg1_error_zer, label='ZER IRI', c='b')
plt.errorbar(poly_deg1_zer.latitude_bins, poly_deg1_model, yerr=poly_deg1_error_model, label='ZER model', c='C0', ls='--')
plt.errorbar(
    poly_deg1_zer.latitude_bins, 
    poly_deg1_zer - poly_deg1_model, 
    yerr=np.sqrt(poly_deg1_error_zer**2 + poly_deg1_error_model**2),
    label='ZER (IRI-model)', c='grey', ls='--')

plt.axhline(y=0, ls=':', c='k')
plt.legend()
plt.gca().set(
    title='',
    xlabel='Latitude [deg N]',
    ylabel='Slope [%/%]',
    );
# %% correlation with lya plot
# var = 'mean_zenith_intensity'
# var_label = 'ZER'
# var_units = '$cm^{-2} s^{-1}$'
var = 'mean_peak_height'
var_label = 'Height'
var_units = 'm'
data1 = mds[var].resample(time='1Y', loffset='-6M').mean('time', keep_attrs=True)
error1 = mds[var].resample(time='1Y', loffset='-6M').std('time', keep_attrs=True)

data_lya = ds_lya.irradiance.resample(time='1Y', loffset='-6M').mean('time', keep_attrs=True)
error_lya = ds_lya.irradiance.resample(time='1Y', loffset='-6M').std('time', keep_attrs=True)

da = data1.assign_coords(lya=data_lya).swap_dims(dict(time='lya'))
pf = da.polyfit(dim='lya', deg=1, cov=True)
poly_coeff = pf.polyfit_coefficients
poly_deg1_error = pf.polyfit_covariance.sel(cov_i=0, cov_j=0).pipe(np.sqrt)
poly_deg1 = poly_coeff.sel(degree=1)

# scatter plots
lat_seq = [-80, 80, -60, 60, -40, 40, -20, 20, 0]
fc = da.reindex(
    latitude_bins=lat_seq).plot.line(
        figsize=(8,9),
        y='lya', col='latitude_bins', col_wrap=2, 
        ls='-', marker='o', markersize=5,
    )

for j in range(fc.axes.shape[1]):
    for i in range(fc.axes.shape[0]):
        if (i+1,j+1) == fc.axes.shape:
            pass 
        else:
            # error bars
            fc.axes[i,j].errorbar(
                da.sel(**fc.name_dicts[i,j]), da.lya, 
                xerr=error1.sel(**fc.name_dicts[i,j]),
                yerr=error_lya,
                ecolor='C0', alpha=0.5, ls='', capsize=3,
            )
            # slopes
            xr.polyval(
                coord=da.lya, coeffs=poly_coeff.sel(**fc.name_dicts[i,j])
                ).plot(
                    y='lya', ax=fc.axes[i,j], 
                    color='C3', ls='-', lw=1,
                    label='s = {}\n$\pm${} ({}%)'.format(
                        poly_deg1.sel(**fc.name_dicts[i,j]).values.astype(int),
                        poly_deg1_error.sel(**fc.name_dicts[i,j]).values.astype(int),
                        abs(poly_deg1_error/poly_deg1*100).sel(**fc.name_dicts[i,j]).values.astype(int),
                        # var_units, data_lya.units,
                    ),
                )
            fc.axes[i,j].legend(handlelength=0.5, frameon=False)
            if j==1:
                fc.axes[i,j].set_ylabel('')

fc.set_axis_labels('{} [{}]'.format(var_label, var_units), 'Ly-a [{}]'.format(data_lya.units))
fc.axes[-2,1].tick_params(labelbottom=True, labelrotation=0)
fc.axes[-2,1].set_xlabel(fc.axes[-1,0].get_xlabel())
for j, sn in enumerate('S N'.split()):
    for i, lat in enumerate(np.linspace(80,0,num=len(fc.axes)).astype(int)):
        fc.axes[i,j].set_title('{} - {} $^\circ$ {}'.format(lat-10, lat+10, sn))
fc.axes[-1,0].ticklabel_format(axis='x', style='sci', scilimits=(3,3))

# %% correlation plot
var1 = 'mean_zenith_intensity'
var1_label = 'ZER'
var1_units = '$cm^{-2} s^{-1}$'

var2 = 'mean_peak_height' #'mean_apparent_solar_time'
var2_label = 'Height' #'LST'
var2_units = 'm' #'h'
data1 = mds[var1].resample(time='1Y', loffset='-6M').mean('time', keep_attrs=True)
data2 = mds[var2].resample(time='1Y', loffset='-6M').mean('time', keep_attrs=True)

fig, ax = plt.subplots(5, 2, figsize=(8,10), sharex=True, sharey=True,
            gridspec_kw=dict(hspace=0.5))
scatterplot_args = dict(y='y', ls='-', marker='o', markersize=5)
for i, lat in enumerate(range(80,0,-20)):
    da = data1.sel(latitude_bins=-lat).assign_coords(y=data2.sel(latitude_bins=-lat)).swap_dims(dict(time='y'))
    da = da.where(~da.y.isnull(), drop=True)
    pf = da.polyfit(dim='y', deg=1, cov=True)
    poly_coeff = pf.polyfit_coefficients
    poly_deg1_error = pf.polyfit_covariance.sel(cov_i=0, cov_j=0).pipe(np.sqrt)
    poly_deg1 = poly_coeff.sel(degree=1)
    da.plot.line(**scatterplot_args, ax=ax[i,0])
    xr.polyval(coord=da.y, coeffs=poly_coeff).plot(
        y='y', 
        ax=ax[i,0],
        color='C3',
        label='s = {}\n$\pm${} ({}%)'.format(
            poly_deg1.values.astype(int),
            poly_deg1_error.values.astype(int),
            abs(poly_deg1_error/poly_deg1*100).values.astype(int),
        ))
    ax[i,0].set_title('{} - {} $^\circ$ S'.format(lat-10, lat+10))
    ax[i,0].legend(handlelength=0.5, frameon=False)

    da = data1.sel(latitude_bins=lat).assign_coords(y=data2.sel(latitude_bins=lat)).swap_dims(dict(time='y'))
    da = da.where(~da.y.isnull(), drop=True)
    pf = da.polyfit(dim='y', deg=1, cov=True)
    poly_coeff = pf.polyfit_coefficients
    poly_deg1_error = pf.polyfit_covariance.sel(cov_i=0, cov_j=0).pipe(np.sqrt)
    poly_deg1 = poly_coeff.sel(degree=1)
    da.plot.line(**scatterplot_args, ax=ax[i,1])
    xr.polyval(coord=da.y, coeffs=poly_coeff).plot(
        y='y', 
        ax=ax[i,1],
        color='C3',
        label='s = {}\n$\pm${} ({}%)'.format(
            poly_deg1.values.astype(int),
            poly_deg1_error.values.astype(int),
            abs(poly_deg1_error/poly_deg1*100).values.astype(int),
        ))
    ax[i,1].set_title('{} - {} $^\circ$ N'.format(lat-10, lat+10))
    ax[i,1].legend(handlelength=0.5, frameon=False)

da = data1.sel(latitude_bins=0).assign_coords(y=data2.sel(latitude_bins=0)).swap_dims(dict(time='y'))
da = da.where(~da.y.isnull(), drop=True)
pf = da.polyfit(dim='y', deg=1, cov=True)
poly_coeff = pf.polyfit_coefficients
poly_deg1_error = pf.polyfit_covariance.sel(cov_i=0, cov_j=0).pipe(np.sqrt)
poly_deg1 = poly_coeff.sel(degree=1)
da.plot.line(**scatterplot_args, ax=ax[-1,0])
xr.polyval(coord=da.y, coeffs=poly_coeff).plot(
    y='y', 
    ax=ax[-1,0],
    color='C3',
    label='s = {}\n$\pm${} ({}%)'.format(
        poly_deg1.values.astype(int),
        poly_deg1_error.values.astype(int),
        abs(poly_deg1_error/poly_deg1*100).values.astype(int),
        ))
ax[-1,0].set_title('{} $^\circ$ N - {} $^\circ$ S'.format(10, 10))    
ax[-1,0].legend(handlelength=0.5, frameon=False)

ax[-1,1].set_axis_off()
ax[-2,1].tick_params(labelbottom=True)
ax[-1,0].set_xlabel('{} [{}]'.format(var1_label, var1_units))
ax[-2,1].set_xlabel('{} [{}]'.format(var1_label, var1_units))
[ax[i,0].set_ylabel('{} [{}]'.format(var2_label, var2_units)) for i in range(5)]
[ax[i,1].set_ylabel('') for i in range(5)]
ax[-1,0].ticklabel_format(axis='x', style='sci', scilimits=(3,3))
ax[-1,0].ticklabel_format(axis='y', style='sci', scilimits=(3,3))

# fig.suptitle('{} measurements'.format(am_pm), fontsize=16)

# %% look at each month at one latiude
var = 'mean_apparent_solar_time'
test = mds[var].resample(time='QS-Dec').mean('time').sel(latitude_bins=80)
mean = test.assign_coords(
    year=test.time.dt.year, season=test.time.dt.season).set_index(
        time=['year', 'season']).unstack().mean('year')
unstacked = test.assign_coords(
    year=test.time.dt.year, season=test.time.dt.season).set_index(
        time=['year', 'season']).unstack() - mean
unstacked.plot(x='year', hue='season')#, row='latitude_bins')


# %% scatter plot
var1 = 'mean_peak_intensity'
var2 = 'mean_peak_height'
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
var = 'mean_peak_intensity'
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

ax[0,0].set_xlim(5e4, 1.25e5)
# %% line plot
var = 'mean_peak_intensity'
data_running_mean = mds[var].rolling(time=365, min_periods=90, center=True).mean('time')
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
    data_running_mean.sel(latitude_bins=-lat).plot(ax=ax[i,0], **dataplot_args)
    plot_y107(ax[i,0], ticks=False)
    ax[i,0].set(ylabel='', xlabel='', title='{} - {} $^\circ$ S'.format(lat-10, lat+10))
    data_running_mean.sel(latitude_bins=lat).plot(ax=ax[i,1], **dataplot_args)
    plot_y107(ax[i,1], ticks=False)
    ax[i,1].set(ylabel='', xlabel='', title='{} - {} $^\circ$ N'.format(lat-10, lat+10)) 

data_running_mean.sel(latitude_bins=0).plot(ax=ax[-1,0], **dataplot_args)
plot_y107(ax[-1,0], ticks=False)
ax[-1,0].set(ylabel='', xlabel='', title='{} $^\circ$ S - {} $^\circ$ N'.format(10, 10))

ax[-1,1].set_axis_off()
ax[-2,1].tick_params(labelbottom=True, labelrotation=30)
ax[-1,0].set_xlabel('Time')
ax[-2,1].set_xlabel('Time')
[ax[i,0].set_ylabel('Peak Intensity') for i in range(5)]
# [ax[i,0].set_ylim(17, 19.5) for i in range(1,5)]
# [ax[i,0].set_ylabel(var) for i in range(5)]
[ax[i,1].set_ylabel('') for i in range(5)]

fig.suptitle('{} measurements'.format(am_pm), fontsize=16)

#%% Countourf plot
var = 'mean_peak_intensity'
mds[var].rolling(time=11, min_periods=1, center=True).mean('time').plot.contourf(
    x='time', y='latitude_bins', figsize=(15,3), robuts=True)
# %% Line plot
var = 'mean_sza'
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



