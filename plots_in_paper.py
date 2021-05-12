#%%
import numpy as np
import xarray as xr
import glob
from os import listdir
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, NoNorm, Normalize
from scipy.interpolate import interp1d

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

orbit = 10122
ch = 1

#%% limb orbit plot example
path_limb = '/home/anqil/Documents/sshfs/oso_extra_storage/StrayLightCorrected/Channel{}/'.format(ch)
filename_limb = 'ir_slc_{}_ch{}.nc'.format(str(orbit).zfill(6), ch)
ds_limb = xr.open_dataset(path_limb+filename_limb).load()
ds_limb.close()

#%%
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
ds = xr.open_dataset('ir_orbit_sample_{}.nc'.format(str(orbit).zfill(6))).load()
ds.close()

d = haversine(ds.longitude[0], ds.latitude[0], ds.longitude, ds.latitude)
ds = ds.update({'d' : (['time'], d, dict(units='km'))}).swap_dims({'time':'d'})
ds = ds.assign({'d': ds.d.diff('d').pipe(abs).cumsum().reindex(d=ds.d).fillna(0)}) 
#%% limb_orbit.png
fig, ax = plt.subplots(3,2, sharex='col', sharey=False, figsize=(8,8),
        gridspec_kw=dict(width_ratios=[1, 0.05], wspace=0.1))

colorplot_args = dict(x='d', y='z', cmap='viridis', add_colorbar=False)
rad_plot = ds.radiance.plot(ax=ax[0,0], norm=Normalize(vmin=1e10, vmax=6e11), **colorplot_args)
error_plot = ds.error.plot(ax=ax[1,0], **colorplot_args)
cbar_kwargs = dict(shrink=0.6, drawedges=False)
cb = fig.colorbar(rad_plot, cax=ax[0,1], **cbar_kwargs, label='[{}]'.format(ds.radiance.units))
cb.outline.set_visible(False)
cb = fig.colorbar(error_plot, cax=ax[1,1], **cbar_kwargs, label='[{}]'.format(ds.error.units))
cb.outline.set_visible(False)
ax[0,0].set(xlabel='', title=ds.radiance.long_name)
ax[1,0].set(xlabel='', title=ds.error.long_name)
ax[0,0].ticklabel_format(axis='y', style='sci', scilimits=(3,3))
ax[1,0].ticklabel_format(axis='y', style='sci', scilimits=(3,3))

latlonplot_args = dict(x='d', c='k', ax=ax[2,0])
ds.latitude.plot(label='latitude', ls='-.', **latlonplot_args)
ds.longitude.plot(label='longitude', ls='--',**latlonplot_args)
ax[2,0].set(ylabel='[$^\circ$ N]\n[$^\circ$ E]')
ax[2,0].legend()

ax_twin = ax[2,0].twinx()
szaplot_args = dict(x='d', c='r', ax=ax_twin)
ds.sza.plot(label='SZA', **szaplot_args)
ax_twin.set_ylabel('SZA [$^\circ$]', color=szaplot_args['c'])
ax_twin.tick_params(axis='y', colors=szaplot_args['c'])

ax[-1,0].set(xlabel='Horizontal distance long the orbit [km]')
ax[2,1].set_axis_off()

# %% open VER file
path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/4pi/'
filename_ver = 'iri_ch{}_ver_{}.nc'.format(ch, str(orbit).zfill(6), ch)

ds_ver = xr.open_dataset(path_ver+filename_ver)
ds_ver.close()
ds_ver = ds_ver.update({'tp' : (['time'],
        (ds_ver.time - ds_ver.time[0])/(ds_ver.time[-1]-ds_ver.time[0]))})
d = haversine(ds_ver.longitude[0], ds_ver.latitude[0], ds_ver.longitude, ds_ver.latitude)
ds_ver = ds_ver.update({'d' : (['time'], d, dict(units='km'))})

# be careful here!
# ds_ver['error2_retrieval'] *= (np.pi*4)**2
# ds_ver['error2_smoothing'] *= (np.pi*4)**2
# ds_ver['ver'] *= np.pi*4
ds_ver_4pi = ds_ver.copy()

# %% VER_histogram.png
from xhistogram.xarray import histogram
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap

fig, ax = plt.subplots(1,1)

# Choose colormap
cmap = pl.cm.Reds
# Get the colormap colors
my_cmap = cmap(np.arange(cmap.N))
# Set alpha
my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
# Create new colormap
my_cmap = ListedColormap(my_cmap)

h_z = histogram(ds_ver.ver.isel(time=slice(200)), bins=[np.arange(-15e3, 15e3, .3e3)*np.pi*4], dim=['time'])
h_z.rename('Histogram').plot(ax=ax, vmax=h_z.sel(z=93e3).max(), cmap=my_cmap)

ver_mean = ds_ver.ver.where(ds_ver.A_diag>0.8).isel(time=slice(200)).mean(dim='time', keep_attrs=True)
# error_mean = ds_ver.error2_retrieval.where(ds_ver.A_diag>0.8).isel(time=slice(200)).mean(dim='time', keep_attrs=True)
error_p_mean = (ds_ver.ver + ds_ver.error2_retrieval.pipe(np.sqrt)).where(ds_ver.A_diag>0.8).isel(time=slice(200)).mean(dim='time', keep_attrs=True)
error_m_mean = (ds_ver.ver - ds_ver.error2_retrieval.pipe(np.sqrt)).where(ds_ver.A_diag>0.8).isel(time=slice(200)).mean(dim='time', keep_attrs=True)

ver_mean.plot(ax=ax, y='z', color='k',ls='-',
                    label='Averaged')
# (ver_mean + np.sqrt(error_mean)).plot(y='z', color='k', ls='--')#, label='mean + error')
# (ver_mean - np.sqrt(error_mean)).plot(y='z', color='k', ls='--', label='mean $\pm$ error')
error_p_mean.plot(ax=ax, y='z', color='k', ls='--')
error_m_mean.plot(ax=ax, y='z', color='k', ls='--', label='Averaged $\pm$ error')

# a priori error
tan_low = 60e3
tan_up = 95e3
peak_apprx = 5e3/0.55*np.pi*4 #7.5e3  # approximation of the airglow peak, used in Sa
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
ax.fill_betweenx(z, xa-sigma_a, xa+sigma_a, 
    label='a priori VER', color='k', alpha=0.1)

ax.set(xlabel='VER [photons cm-3 s-1]', 
    #    ylabel='Altitdue grid',
       title='Serial of 200 images')
ax.ticklabel_format(axis='both', style='sci', scilimits=(3,3))
ax.legend(loc='upper left')

plt.show()
# %% access AVK 
from VER_invert_routine import invert_1d
path_limb = '/home/anqil/Documents/sshfs/oso_extra_storage/StrayLightCorrected/Channel{}/'.format(ch)
result_1d, AVK = invert_1d(orbit, ch, path_limb, save_file=False, im_lst=[0], return_AVK=True)

# %% AVK_example.png
fig, ax = plt.subplots(1,2, sharey=True, sharex=True)
lineplot_args = dict(ax=ax[0], hue='row_z', y='col_z', add_legend=False, alpha=0.5)
l0 = AVK.sel(row_z=slice(60e3, 95e3)).plot.line(color='r', **lineplot_args)
l1 = AVK.sel(row_z=slice(96e3, 120e3)).plot.line(color='k', **lineplot_args)
l2 = AVK.sel(row_z=slice(59e3)).plot.line(color='k', **lineplot_args)
ax[0].legend((l0[0], l1[0]), ('60 - 95km', 'others'))
ax[0].set(ylabel='z [m]', xlabel='AVK row')
ax[0].ticklabel_format(axis='y', style='sci', scilimits=(3,3))


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
ax_twin.ticklabel_format(axis='x', style='sci', scilimits=(3,3))

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

#%% open gauss file
path = '~/Documents/osiris_database/iris_oh/gauss_character/4pi/'
ds_agc = xr.open_dataset(path+'gauss_{}.nc'.format(str(orbit).zfill(6)))
ds_agc.close()

# be careful here!
ds_agc['thickness'] = ds_agc.peak_sigma.reduce(lambda x: 2*x, keep_attrs=True)
ds_agc['thickness_error'] = ds_agc.peak_sigma_error.reduce(lambda x: 2*x, keep_attrs=True)
# ds_agc['peak_intensity'] *= np.pi*4
# ds_agc['peak_intensity_error'] *= np.pi*4
# ds_agc['cov_peak_intensity_peak_sigma'] *= np.pi*4
# ds_agc['cov_peak_intensity_peak_height'] *= np.pi*4
# ds_agc['zenith_intensity'] *= np.pi*4
# ds_agc['zenith_intensity_error'] *= np.pi*4

ds_ver = ds_ver_4pi.update(ds_agc)
ds_ver = ds_ver.swap_dims({'time': 'd'})

if orbit == 10122:
    ds_ver = ds_ver.where(ds_ver.tp<0.4, drop=True)
elif orbit == 38710:
    ds_ver = ds_ver.where(ds_ver.tp>0.55, drop=True)

cond_peak_height = (ds_ver.peak_height>60e3)*(ds_ver.peak_height<95e3)
cond_chisq = ds_ver.chisq<2
cond_gauss = cond_peak_height * cond_chisq

#%% Gauss_profile_example.png
from characterise_agc_routine import gauss
fig, ax = plt.subplots(1,1)
isel_args = dict(d=0)
ds_ver.ver.isel(**isel_args).plot(
    y='z', color='k', marker='.', ls='-', alpha=0.4, label='VER data')
ds_ver.error2_retrieval.pipe(np.sqrt).isel(**isel_args).plot(
    y='z', color='b', marker='', ls='-', alpha=0.4, label='VER error')
ds_ver.z.pipe(
    gauss, *ds_ver[['peak_intensity', 'peak_height', 'peak_sigma']].isel(
        **isel_args).to_array()).plot(
        y='z', color='C3', lw=4, label='Gaussian model'
        )
ax.ticklabel_format(axis='both', style='sci', scilimits=(3,3))

ax.legend(loc='upper right')
ax.set_xlabel('VER [{}]'.format(ds_ver.ver.units))
ax.set_title('')

#%%
fig, ax = plt.subplots(2,1, gridspec_kw=dict(height_ratios=[.1,1], hspace=0.3))
colorplot_args = dict(x='d', vmin=0, vmax=12e3 *np.pi*4, ylim=[60e3, 95e3], 
                cmap='viridis', add_colorbar=True)
cbar_kwargs = dict(cax=ax[0], shrink=0.6, orientation='horizontal', label='')

ds_ver.ver.where(ds_ver.A_peak>0.8).plot(ax=ax[1], **colorplot_args, cbar_kwargs=cbar_kwargs)

ax[0].tick_params(tick2On=True, label2On=True, tick1On=False, label1On=False)
ax[0].set_title('[{}]'.format(ds_ver.ver.units))

ax[-1].set(xlabel='Horizontal distance long the orbit [km]',
            ylabel='Altitude [m]',
            title='OH (3-1) volume emission rate')

ax[0].ticklabel_format(axis='x', style='sci', scilimits=(3,3))
ax[0].ticklabel_format(axis='y', style='sci', scilimits=(3,3))
ax[1].ticklabel_format(axis='x', style='sci', scilimits=(3,3))
ax[1].ticklabel_format(axis='y', style='sci', scilimits=(3,3))

#%%
import cartopy.crs as ccrs
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
plt.plot(ds_ver.longitude, ds_ver.latitude, 'r*', markersize=10,
             transform=ccrs.PlateCarree())
ax.coastlines()
ax.set_global()

#%% Gauss_orbit_example.png
from characterise_agc_routine import gauss
fig, ax = plt.subplots(5,2, sharex=False, figsize=(10,10), 
    gridspec_kw=dict(height_ratios=[0.1, 1,1,1,1], top=0.85, hspace=0.3, wspace=0.4))
ax_twin = np.array([[ax[i,j].twinx() for j in range(2)] for i in range(len(ax))])

panel_labels = 'a b c d e f g h'.split()
k = 0
for j in range(2):
    for i in range(1,5):
        s = panel_labels[k]
        ax[i,j].text(0.05, 0.9, '{})'.format(s), 
            backgroundcolor='w', fontweight='bold', transform=ax[i,j].transAxes) 
        k += 1

colorplot_args = dict(x='d', vmin=0, vmax=12e3 *np.pi*4, ylim=[60e3, 95e3], 
                cmap='viridis', add_colorbar=False)

ds_ver.error2_retrieval.pipe(np.sqrt).where(
        ds_ver.error2_retrieval!=0, other=1e-1).where(
            ds_ver.A_peak>0.8).plot(ax=ax[1,0], **colorplot_args)
ds_ver.ver.where(ds_ver.A_peak>0.8).plot(ax=ax[2,0], **colorplot_args)
colorplot_args['add_colorbar'] = True
cbar_kwargs = dict(cax=ax[0,0], shrink=0.6, orientation='horizontal', label='[photons cm-3 s-1]')
ds_ver.z.pipe(gauss, ds_ver.peak_intensity, ds_ver.peak_height, ds_ver.peak_sigma).where(
    cond_gauss).plot(
    ax=ax[3,0], **colorplot_args, cbar_kwargs=cbar_kwargs)

lineplot_args = dict(x='d', color='k', alpha=0.9)
ds_ver.peak_intensity.plot(ax=ax[1,1], **lineplot_args, ylim=(0,3e4 *np.pi*4))
ds_ver.peak_height.plot(ax=ax[2,1], **lineplot_args, ylim=(8e4, 9e4))
ds_ver.thickness.plot(ax=ax[3,1], **lineplot_args, ylim=(0, 2e4))
ds_ver.zenith_intensity.plot(ax=ax[4,1], **lineplot_args, ylim=(0, 6e4 *np.pi*4))
# ds_ver.chisq.plot(ax=ax[4,1], **lineplot_args, ylim=(0, 80))
l_lon, = ds_ver.longitude.plot(ax=ax[4,0], label='longitude', x='d', c='k', ls='--', alpha=0.9)
l_lat, = ds_ver.latitude.plot(ax=ax[4,0], label='latitude', x='d', c='k', ls='-.', alpha=0.9)
l_sza, = ds_ver.sza.plot(ax=ax_twin[4,0], label='SZA', x='d', c='r', alpha=0.9)

fillplot_args = dict(color='r', alpha=0.3)
ax[1,1].fill_between(ds_ver.d, 
    ds_ver.peak_intensity + ds_ver.peak_intensity_error,
    ds_ver.peak_intensity - ds_ver.peak_intensity_error, **fillplot_args)
ax[2,1].fill_between(ds_ver.d, 
    ds_ver.peak_height + ds_ver.peak_height_error,
    ds_ver.peak_height - ds_ver.peak_height_error, **fillplot_args)
ax[3,1].fill_between(ds_ver.d, 
    ds_ver.thickness + ds_ver.thickness_error,
    ds_ver.thickness - ds_ver.thickness_error, **fillplot_args)
ax[4,1].fill_between(ds_ver.d, 
    ds_ver.zenith_intensity + ds_ver.zenith_intensity_error,
    ds_ver.zenith_intensity - ds_ver.zenith_intensity_error, **fillplot_args)

errorplot_args = dict(x='d', color='b', ylim=(0,1), alpha=0.6)
(ds_ver.peak_intensity_error/ds_ver.peak_intensity).plot(ax=ax_twin[1,1], **errorplot_args)
(ds_ver.peak_height_error/ds_ver.peak_height).plot(ax=ax_twin[2,1], **errorplot_args)
(ds_ver.thickness_error/ds_ver.thickness).plot(ax=ax_twin[3,1], **errorplot_args)
(ds_ver.zenith_intensity_error/ds_ver.zenith_intensity).plot(ax=ax_twin[4,1], **errorplot_args)

ax[1,0].set(title='VER error')
ax[2,0].set(title='Original VER')
ax[3,0].set(title='Gaussian modelled VER')
ax[1,1].set(title='Peak intensity', ylabel='$V_{peak}$ '+'[{}]'.format(ds_ver.peak_intensity.units))
ax[2,1].set(title='Peak height', ylabel='$z_{peak}$ '+'[{}]'.format(ds_ver.peak_height.units))
ax[3,1].set(title='Thickness', ylabel='$2 \sigma$ '+'[{}]'.format(ds_ver.thickness.units))
ax[4,1].set(title='Zenith intensity', ylabel='$V_{zenith}$ '+'[{}]'.format(ds_ver.zenith_intensity.units))
# ax[4,1].set(title='$\chi^2$', ylabel='')
ax[4,0].set(ylabel='[$^\circ$ N]\n[$^\circ$ E]')
ax[4,0].legend()
[ax[-1,j].set(xlabel='Horizontal distance long the orbit [km]') for j in range(2)]

ax[0,0].tick_params(tick2On=True, label2On=True, tick1On=False, label1On=False)
ax[0,0].set_title('[{}]'.format(ds_ver.ver.units))
[[ax[i,j].set(xlabel='') for i in range(len(ax)-1)] for j in range(2)]
[[ax[i,j].tick_params(bottom=True, labelbottom=False) for i in range(1,len(ax)-1)] for j in range(2)]
[[ax[i, j].set_xlim(ax[1,0].get_xlim()) for i in range(1, len(ax))] for j in range(2)]
ax[0,1].set_axis_off()
ax_twin[0,1].set_axis_off()
[ax_twin[i,0].set_axis_off() for i in range(4)]
[ax_twin[i,1].set_ylabel('rel. error', color=errorplot_args['color']) for i in range(1,len(ax))]
[ax_twin[i,1].tick_params(axis='y', colors=errorplot_args['color']) for i in range(1,len(ax))]
ax_twin[4,0].set_ylabel('SZA [$^\circ$]', color=l_sza.get_color())
ax_twin[4,0].tick_params(axis='y', colors=l_sza.get_color())

ax[0,0].ticklabel_format(axis='x', style='sci', scilimits=(3,3))
[ax[i,0].ticklabel_format(axis='y', style='sci', scilimits=(3,3)) for i in range(4)]
[ax[i,1].ticklabel_format(axis='y', style='sci', scilimits=(3,3)) for i in range(5)]
[ax[-1,j].ticklabel_format(axis='x', style='sci', scilimits=(3,3)) for j in range(2)]

fig.suptitle('Orbit no. {} \n {} \n to \n {}'.format(
    str(orbit).zfill(6), ds_ver.time[0].values, ds_ver.time[-1].values))

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

#%% Monthly_VER.png
am_pm = 'All'
min_sza = 96
d_m = 'daily'
path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/ver/'
filename = '{2}/{0}/{1}_{0}_{2}_ver_clima_{3}.nc'.format(min_sza, am_pm, d_m, '{}')
with xr.open_mfdataset(path+filename.format('*')) as mds:
    # mds = mds.assign_coords(z=mds.z*1e-3)
    # mds.z.attrs['units'] = 'km'
    mds['mean_ver'] *= np.pi*4/0.55 
    mds['std_ver'] *= np.pi*4/0.55
    mds = mds.reindex(latitude_bins=mds.latitude_bins[::-1])
    #resample into monthly mean
    mds = mds.where(mds.count_ver>50).resample(time="1M").mean()

    #stack years
    mds = mds.assign_coords(
        dict(year=mds.time.dt.year, month=mds.time.dt.month)).set_index(
        time=['year', 'month']).unstack()

    #shift NH 6 months then mean
    mds_nh = mds.sel(latitude_bins=slice(80,20)).roll(month=6, roll_coords=False).mean('year')
    mds_sh = mds.sel(latitude_bins=slice(0, -80)).mean('year')

    # fc = mds_nh.merge(mds_sh).reindex(latitude_bins = mds.latitude_bins).mean_ver.plot.contourf(
    #     y='z', x='month', row='latitude_bins',
    #     figsize=(6,15), ylim=(75e3, 95e3), cmap='viridis', vmin=0, vmax=12e3*np.pi*4,
    #     robust=True,
    #     cbar_kwargs=dict(shrink=0.6, label='Mean VER'))
    
    # for i in range(len(mds.latitude_bins)):
    #     fc.axes[i][0].set_title('{}$^\circ$ - {}$^\circ$'.format(
    #         mds.latitude_bins[i].values-10, mds.latitude_bins[i].values+10))
    #     fc.axes[i][0].ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    # fc.axes[-1][0].set_xlabel('Month eq. to NH')
    # plt.show()
    fig, ax = plt.subplots(5,2, figsize=(10,10), gridspec_kw=dict(hspace=0.4))
    plt_args = dict(y='z', x='month', cmap='viridis',
                    ylim=(75e3, 95e3), xlim=(1,12),  
                    vmin=0, vmax=15e4, #12e3*np.pi*4,
                    add_colorbar=False)
    for i, lat in enumerate(range(80, 0, -20)):
        mds_sh.sel(latitude_bins=-lat).mean_ver.plot.contourf(ax=ax[i,0], **plt_args)
        mds_nh.sel(latitude_bins=lat).mean_ver.plot.contourf(ax=ax[i,1], **plt_args)
        ax[i,0].set(title='{} - {}$^\circ S$'.format(lat-10, lat+10), xlabel='', xticklabels='')
        ax[i,1].set(title='{} - {}$^\circ N$'.format(lat-10, lat+10), xlabel='', xticklabels='')
    c = mds_sh.sel(latitude_bins=0).mean_ver.plot.contourf(ax=ax[-1,0], **plt_args)#, cbar_kwargs=cbar_kwargs)
    ax[-1,0].set(title='{}$^\circ N$ - {}$^\circ S$'.format(10, 10), 
            xlabel='Month')
    # ax[-1,0].set(xlabel='Month')
    ax[-2,1].set(xlabel='Month', xticklabels='6 8 10 12 2 4 6'.split())
    for i in range(5):
        for j in range(2):
            ax[i,j].ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    cbar_kwargs = dict(cax=ax[-1,1], 
            # format='%1.1e', ticks=[0, 7.5e04, 1.5e05], 
            shrink=0.9, orientation='horizontal', label='[photons cm-3 s-1]')
    cbar = fig.colorbar(c, **cbar_kwargs)
    cbar.formatter.set_powerlimits((0,0))
    
    # ax[-1,1].ticklabel_format(axis='x', style='sci', scilimits=(3,3))
    ax[-1,1].set_box_aspect(1/7)
    
# %% Monthly Gauss parameters
path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/gauss/ALL/'
filename = 'gauss_ALL_D_96_{}.nc'
# years = list(range(2001, 2008)) + list(range(2009, 2018))
years = list(range(2001,2018))# + list(range(2012,2018))

with xr.open_mfdataset([path+filename.format(y) for y in years]) as mds:
    
    mds['mean_peak_intensity'] *= 4*np.pi
    mds['mean_peak_intensity'].attrs['units'] = 'photons cm-3 s-1'
    mds['mean_peak_height'].attrs['units'] = 'm'
    mds['mean_peak_sigma'].attrs['units'] = 'm'
    mds['mean_thickness'] = 2*mds['mean_peak_sigma']
    mds['mean_thickness'].attrs['units'] = 'm'
    mds['std_thickness'] = 2*mds['std_peak_sigma']
    
    # mds = mds.where(mds.count_sza>10).resample(time='1M').mean(keep_attrs=True)
    mds = mds.where(mds.count_sza>10).rolling(time=30, min_periods=10, center=True).mean('time')

    mds = mds.assign_coords(
        dict(year=mds.time.dt.year, doy=mds.time.dt.dayofyear)).set_index(
        time=['year', 'doy']).unstack()

    #shift NH 6 months
    mds_nh = mds.sel(latitude_bins=slice(20,80)).roll(doy=180, roll_coords=False)
    mds_sh = mds.sel(latitude_bins=slice(-80,0))

#%%
    # years_ssw = [2002, 2004, 2006, 2009, 2012, 2013]
    data_vars = 'peak_intensity peak_height peak_sigma'.split()
    data_sym = '$V_{peak}$ $z_{peak}$ $2\sigma$'.split()
    # data_vars = 'apparent_solar_time sza'.split()
    # data_sym = 'LST SZA'.split()
    fig, ax = plt.subplots(5, len(data_vars), figsize=(15,10), sharex=True, sharey='col', gridspec_kw=dict(hspace=0.5))
    plot_args = dict(x='doy', hue='year', add_legend=False, alpha=0.3)
    for j,d in enumerate(data_vars):
        # mds_nh.sel(latitude_bins=80, year=years_ssw)['mean_{}'.format(d)].plot.line(ax=ax[0,j], c='g', **plot_args)
        # mds_nh.sel(latitude_bins=60, year=years_ssw)['mean_{}'.format(d)].plot.line(ax=ax[1,j], c='g', **plot_args)
        for i,lat in enumerate(range(80,0,-20)):
            l_nh = mds_nh.sel(latitude_bins=lat)['mean_{}'.format(d)].plot.line(ax=ax[i, j], c='r', **plot_args)
            l_sh = mds_sh.sel(latitude_bins=-lat)['mean_{}'.format(d)].plot.line(ax=ax[i, j], c='b', **plot_args)
            ax[i, j].set(
                ylabel='[{}]'.format(mds['mean_{}'.format(d)].units), 
                xlabel='', 
                title='{}$^\circ$ - {}$^\circ$'.format(lat-10, lat+10),
                )
            if i==0:
                ax[i,j].set_title(data_sym[j]+'\n'+'{}$^\circ$ - {}$^\circ$'.format(lat-10, lat+10))
            ax[i, j].legend([l_nh[0], l_sh[0]], ['NH', 'SH'], loc='upper right')
        mds_sh.sel(latitude_bins=0)['mean_{}'.format(d)].plot.line(ax=ax[-1, j], c='k', **plot_args)
        ax[-1, j].set(
            xlabel='SH-eq DOY', 
            ylabel='[{}]'.format(mds['mean_{}'.format(d)].units), 
            title='{}$^\circ$ S - {}$^\circ$ N'.format(10, 10),
            )
    [[ax[i,j].ticklabel_format(axis='y', style='sci', scilimits=(3,3)) for i in range(ax.shape[0])] for j in range(ax.shape[1])]
    ax[0,0].set_ylim(50e3, 200e3)
    ax[0,1].set_ylim(80e3, 87e3)
    ax[0,2].set_ylim(4.5e3, 12e3)

    plt.show()

#%%
var = 'mean_apparent_solar_time'
lats = np.arange(80, 20,-20)
years=slice(2018)
plot_args = dict(hue='year', alpha=0.3, add_legend=False)
fig, ax = plt.subplots(4,1, sharex=True, sharey=True, figsize=(3,8))
for i, lat in enumerate(lats):
    l_sh = mds[var].sel(latitude_bins=-lat, year=years).plot.line(ax=ax[i], c='b', **plot_args)
    l_nh = mds[var].sel(latitude_bins=lat, year=years).roll(
        doy=180, roll_coords=False).plot.line(ax=ax[i], c='r', **plot_args)
mds[var].sel(latitude_bins=0, year=years).plot.line(ax=ax[-1], c='k', **plot_args)

# years_ssw = [2002, 2004, 2006, 2009, 2012, 2013]
# mds.mean_sza.sel(latitude_bins=80, year=years_ssw).roll(
#     doy=180, roll_coords=False).plot.line(
#         x='doy', hue='year', c='k')

# %% seasonal lattiudinal variations
am_pm = 'All'
min_sza = 96
d_m = 'daily'
path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/ver/'
filename = '{2}/{0}/{1}_{0}_{2}_ver_clima_{3}.nc'.format(min_sza, am_pm, d_m, '{}')

with xr.open_mfdataset(path+filename.format('*')) as mds:
    # mds = mds.assign_coords(z=mds.z*1e-3)
    # mds.z.attrs['units'] = 'km'
    mds['mean_ver'] *= np.pi*4/0.55 
    mds['std_ver'] *= np.pi*4/0.55
    mds = mds.reindex(latitude_bins=mds.latitude_bins[::-1])
    #resample into monthly mean
    mds = mds.where(mds.count_ver>50).resample(time="QS-JAN").mean()

    #stack years
    mds = mds.assign_coords(
        dict(year=mds.time.dt.year, season=mds.time.dt.season)).set_index(
        time=['year', 'season']).unstack().reindex(season='MAM JJA SON DJF'.split())

    contourf_args = dict(vmin=0, cmap='viridis', figsize=(5,8), x='latitude_bins', y='z', row='season')
    fc = mds.mean_ver.mean('year').sel(z=slice(75e3, 95e3)).plot.contourf(**contourf_args)
    [fc.axes[i][0].ticklabel_format(axis='y', style='sci', scilimits=(3,3)) for i in range(4)]

#%%
year = 2012
path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/gauss/'
filename = 'gauss_D_{}.nc'
with xr.open_dataset(path+filename.format(year)) as ds:
    ds.close()
    ds = ds.resample(time='1M').mean()
    
    # plt.figure()
    # ds.mean_peak_intensity.where(ds.count_sza>100).plot(x='time', row='latitude_bins', marker = '*', figsize=(5,8))
    
    plt.figure()
    ds.mean_peak_intensity.plot.contourf(x='time', y='latitude_bins', robust=True)

#%%
path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/gauss/PM/'
filename = 'gauss_PM_geo_D_96_2001.nc'
with xr.open_dataset(path+filename) as ds:
    print(ds)
# %%
