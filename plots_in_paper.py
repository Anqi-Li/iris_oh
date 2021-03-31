#%%
import numpy as np
import xarray as xr
import glob
from os import listdir
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d

orbit = 10122
ch = 1

#%% limb orbit plot example
path_limb = '/home/anqil/Documents/sshfs/oso_extra_storage/StrayLightCorrected/Channel{}/'.format(ch)
filename_limb = 'ir_slc_{}_ch{}.nc'.format(str(orbit).zfill(6), ch)
ds_limb = xr.open_dataset(path_limb+filename_limb).load()
ds_limb.close()

alts = np.arange(40e3, 110e3, 1.5e3)
ir_altitude = []
error_altitude = []
for (data, error, alt) in zip(ds_limb.data.T, ds_limb.error.T, ds_limb.altitude):
    f = interp1d(alt, data, bounds_error=False)
    ir_altitude.append(f(alts))
    f = interp1d(alt, error, bounds_error=False)
    error_altitude.append(f(alts))
ir_altitude = xr.DataArray(ir_altitude, coords=[ds_limb.time, alts], dims=['time', 'z'], 
                attrs=dict(long_name='IRI CH1 radiance', units=ds_limb.data.attrs['units']))
error_altitude = xr.DataArray(error_altitude, coords=[ds_limb.time, alts], dims=['time', 'z'], 
                attrs=dict(long_name='IRI CH1 error', units=ds_limb.data.attrs['units']))

ds = xr.Dataset(dict(radiance=ir_altitude, error=error_altitude,
        orbit=ds_limb.orbit, channel=ds_limb.channel, 
        latitude=ds_limb.latitude, 
        longitude=ds_limb.longitude,
        sza=ds_limb.sza,
        apparent_solar_time=ds_limb.apparent_solar_time))
ds.z.attrs['units']='m'
ds.to_netcdf('ir_orbit_sample_{}.nc'.format(str(orbit).zfill(6)), mode='w')

#%%
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(lambda x: x*np.pi/180, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


ds = xr.open_dataset('ir_orbit_sample_{}.nc'.format(str(orbit).zfill(6))).load()
ds.close()

d = haversine(ds.longitude[0], ds.latitude[0], ds.longitude, ds.latitude)
ds = ds.update({'d' : (['time'], d, dict(units='km'))}).swap_dims({'time':'d'})
ds = ds.assign({'d': ds.d.diff('d').pipe(abs).cumsum().reindex(d=ds.d).fillna(0)}) 

fig, ax = plt.subplots(3,2, sharex='col', sharey=False, figsize=(10,10),
        gridspec_kw=dict(width_ratios=[1, 0.05]))

colorplot_args = dict(x='d', y='z', cmap='viridis', norm=LogNorm())
ds.radiance.plot(ax=ax[0,0], vmin=1e10, vmax=1e12, **colorplot_args, cbar_kwargs=dict(cax=ax[0,1], shrink=0.6))
ds.error.plot(ax=ax[1,0], **colorplot_args, cbar_kwargs=dict(cax=ax[1,1], shrink=0.6))
latlonplot_args = dict(x='d', c='k', ax=ax[2,0])
ds.latitude.plot(label='latitude', ls='-.', **latlonplot_args)
ds.longitude.plot(label='longitude', ls='--',**latlonplot_args)
ax[2,0].set(ylabel='[$^\circ$ N]\n[$^\circ$ E]', )
ax[2,0].legend()

ax_twin = ax[2,0].twinx()
szaplot_args = dict(x='d', c='r', ax=ax_twin)
ds.sza.plot(label='SZA', **szaplot_args)
ax_twin.set_ylabel('SZA [$^\circ$]', color=szaplot_args['c'])
ax_twin.tick_params(axis='y', colors=szaplot_args['c'])

ax[-1,0].set(xlabel='Horizontal distance long the orbit [km]')
ax[2,1].set_axis_off()

#%%
# ds_limb.pixel.attrs['long_name'] = 'IRI Ch{} pixel number'.format(ch)
# ds_limb.data.attrs['long_name'] = 'IRI CH{} radiance'.format(ch)

# fig, ax = plt.subplots(3,1, sharex=True, sharey=False, figsize=(15,15))
# # ds_limb.data.plot(x='time', ax=ax[0], 
# #             norm=LogNorm(), vmin=1e8, vmax=1e15, extend='both',
# #             cmap='viridis', robust=True)

# ir_altitude.plot(x='time', y='Altitude', ax=ax[1],
#                 norm=LogNorm(), vmin=1e10, vmax=1e12)
# error_altitude.plot(x='time', y='Altitude', ax=ax[2],
#                 norm=LogNorm())

# %%
path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/'
filename_ver = 'iri_ch{}_ver_{}.nc'.format(ch, str(orbit).zfill(6), ch)

ds_ver = xr.open_dataset(path_ver+filename_ver)
ds_ver.close()
ds_ver.where(ds_ver.A_diag>0.8).isel(time=slice(200)).ver.plot.line(
    y='z', add_legend=False, ls='', marker='.')
plt.show()

# %% VER heatmap plot
from xhistogram.xarray import histogram
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap

# Choose colormap
cmap = pl.cm.Reds
# Get the colormap colors
my_cmap = cmap(np.arange(cmap.N))
# Set alpha
my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
# Create new colormap
my_cmap = ListedColormap(my_cmap)

h_z = histogram(ds_ver.ver.isel(time=slice(200)), bins=[np.arange(-15e3, 15e3, .3e3)], dim=['time'])
h_z.rename('Histogram').plot(vmax=h_z.sel(z=93e3).max(), cmap=my_cmap)

ver_mean = ds_ver.ver.where(ds_ver.A_diag>0.8).isel(time=slice(200)).mean(dim='time', keep_attrs=True)
# error_mean = ds_ver.error2_retrieval.where(ds_ver.A_diag>0.8).isel(time=slice(200)).mean(dim='time', keep_attrs=True)
error_p_mean = (ds_ver.ver + ds_ver.error2_retrieval.pipe(np.sqrt)).where(ds_ver.A_diag>0.8).isel(time=slice(200)).mean(dim='time', keep_attrs=True)
error_m_mean = (ds_ver.ver - ds_ver.error2_retrieval.pipe(np.sqrt)).where(ds_ver.A_diag>0.8).isel(time=slice(200)).mean(dim='time', keep_attrs=True)

ver_mean.plot(y='z', color='k',ls='-',
                    label='Averaged')
# (ver_mean + np.sqrt(error_mean)).plot(y='z', color='k', ls='--')#, label='mean + error')
# (ver_mean - np.sqrt(error_mean)).plot(y='z', color='k', ls='--', label='mean $\pm$ error')
error_p_mean.plot(y='z', color='k', ls='--')
error_m_mean.plot(y='z', color='k', ls='--', label='Averaged $\pm$ error')

# a priori error
tan_low = 60e3
tan_up = 95e3
peak_apprx = 5e3/0.55 #7.5e3  # approximation of the airglow peak, used in Sa
z = np.arange(tan_low-5e3, tan_up+20e3, 1e3) # m
z_top = z[-1] + (z[-1]-z[-2]) #m
xa = np.ones(len(z)) * 0 
sigma_a = np.ones_like(xa) * peak_apprx
# sigma_a[np.where(np.logical_or(z<tan_low, z>tan_up))] = 1e-1
sigma_a = np.ones_like(xa) * peak_apprx
n = (z<tan_low).sum()
sigma_a[np.where(z<tan_low)] = np.logspace(-1, np.log10(peak_apprx), n)
n = (z>tan_up).sum()
sigma_a[np.where(z>tan_up)] = np.logspace(np.log10(peak_apprx), -1, n)
plt.fill_betweenx(z, xa-sigma_a, xa+sigma_a, 
    label='a priori VER', color='k', alpha=0.1)

plt.gca().set(xlabel='VER [photons cm-3 s-1]', 
    #    ylabel='Altitdue grid',
       title='Serial of 200 images')
plt.legend(loc='upper left')
plt.show()
# %%
from VER_invert_routine import invert_1d
path_limb = '/home/anqil/Documents/sshfs/oso_extra_storage/StrayLightCorrected/Channel{}/'.format(ch)
result_1d, AVK = invert_1d(orbit, ch, path_limb, save_file=False, im_lst=[0], return_AVK=True)

# %%
fig, ax = plt.subplots(1,2, sharey=True, sharex=True)
lineplot_args = dict(ax=ax[0], hue='row_z', y='col_z', add_legend=False, alpha=0.5)
l0 = AVK.sel(row_z=slice(60e3, 95e3)).plot.line(color='r', **lineplot_args)
l1 = AVK.sel(row_z=slice(96e3, 120e3)).plot.line(color='k', **lineplot_args)
l2 = AVK.sel(row_z=slice(59e3)).plot.line(color='k', **lineplot_args)
ax[0].legend((l0[0], l1[0]), ('60 - 95km', 'others'))
ax[0].set(ylabel='z [m]', xlabel='AVK row')
# ax_twin = ax[0].twiny()
# fwhm.plot(ax=ax_twin, y='z')

ax_twin = ax[1].twiny()
# l0 = result_1d.A_diag.plot(ax=ax[1], y='z', color='C0', label='AVK diaganol')
l1 = result_1d.A_peak.plot(ax=ax[1], label='AVK max at row', y='z', color='k')
l2 = result_1d.A_peak_height.plot(ax=ax_twin, label='Height at AVK max', y='z', color='C3', ls=':')
# ax[1].legend(loc=(0.05, 0.9)) #loc='upper center')
# ax_twin.legend(loc=(0.05, 0.8))#loc='lower center')
ax[1].set(xlabel='AVK max at row', ylabel='', title='')
ax[1].xaxis.get_label().set_color(l1[0].get_color())

ax_twin.set(xlabel='Height at AVK max [m]', xlim=(50e3, 101e3), ylabel='', title='')
ax_twin.xaxis.get_label().set_color(l2[0].get_color())
ax_twin.tick_params(axis='x', colors=l2[0].get_color())
plt.show()

# %% plot peak widths (resolutions)
from scipy.signal import chirp, find_peaks, peak_widths
fwhm = []
for i in range(len(result_1d.z)):
    x = AVK.isel(row_z=i)#.plot(xlim=(82e3, 88e3))
    peaks, _ = find_peaks(x, height=result_1d.A_peak.isel(z=i).item()/2)
    results_half = peak_widths(x, peaks, rel_height=0.5)
    if len(results_half[0]) == 0:
        fwhm.append(np.nan)
    else:
        fwhm.append(results_half[0].item())
fwhm = xr.DataArray(fwhm, dims='z').assign_coords(z=result_1d.z) * 1e3
fwhm.plot(y='z')
plt.xlabel('width [m]')
# %% Monthly mean SZA
# path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/daily/'
# filename = 'daily_ver_clima_{}.nc'.format('*')
# with xr.open_mfdataset(path+filename) as mds:
#     mds = mds.reindex(
#         latitude_bins=mds.latitude_bins[::-1])
#     mds.mean_sza.plot.contourf(x='time', figsize=(10,3), levels=[90, 96, 100],
#         cbar_kwargs=dict(label='Mean SZA [$^\circ$]'))
#     plt.xlabel('Year')
#     plt.ylabel('Latitude bins $\pm$ 10 $^\circ$')

#%% OH daily and resample monthly means
am_pm = 'All'
min_sza = 96
d_m = 'daily'
path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/'
filename = '{2}/{0}/{1}_{0}_{2}_ver_clima_{3}.nc'.format(min_sza, am_pm, d_m, '{}')
with xr.open_mfdataset(path+filename.format('*')) as mds:
    # mds = mds.assign_coords(z=mds.z*1e-3)
    # mds.z.attrs['units'] = 'km'
    mds['mean_ver'] /= 0.55
    mds['std_ver'] /= 0.55
    mds = mds.reindex(latitude_bins=mds.latitude_bins[::-1])
    #resample into monthly mean
    mds = mds.where(mds.count_ver>100).resample(time="1M").mean()

    # mds.mean_ver.plot.contourf(
    #         x='time', y='z', row='latitude_bins', ylim=(75,95),
    #         vmin=0, vmax=4865, robust=True, cmap='viridis', figsize=(10,10)
    #         )
    # plt.show()

    #stack years
    mds = mds.assign_coords(
        dict(year=mds.time.dt.year, month=mds.time.dt.month)).set_index(
        time=['year', 'month']).unstack()

    #shift NH 6 months then mean
    mds_nh = mds.sel(latitude_bins=slice(80,20)).roll(month=6, roll_coords=False).mean('year')
    mds_sh = mds.sel(latitude_bins=slice(0,-80)).mean('year')

    fc = mds_nh.merge(mds_sh).reindex(latitude_bins = mds.latitude_bins).mean_ver.plot.contourf(
        y='z', x='month', row='latitude_bins',
        figsize=(6,15), ylim=(75e3, 95e3), cmap='viridis', vmin=0, vmax=12e3,
        robust=True,
        cbar_kwargs=dict(shrink=0.6, label='Mean VER'))
    
    for i in range(len(mds.latitude_bins)):
        fc.axes[i][0].set_title('{} to {} $^\circ$'.format(
            mds.latitude_bins[i].values-10, mds.latitude_bins[i].values+10))
    fc.axes[-1][0].set_xlabel('Month eq. to NH')
    plt.show()
    
# %%
    mds.mean_ver.mean('year').plot.contourf(y='z', x='month', row='latitude_bins',
        figsize=(6,15), ylim=(75e3, 95e3), cmap='viridis', vmin=0, vmax=12e3,
        cbar_kwargs=dict(shrink=0.6))
    plt.show()
#%%
mds.roll(time=0, roll_coords=False).assign_coords(
    dict(year=mds.time.dt.year, doy=mds.time.dt.dayofyear)).set_index(
        time=['year', 'doy']).unstack()

# %%

# %%
