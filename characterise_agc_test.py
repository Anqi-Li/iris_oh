#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from xhistogram.xarray import histogram
import glob

# %% open VER file
orbit = 10122 #38710 #10122 #4814 #3713
orbit_num = str(orbit).zfill(6)
# path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/oh/'
# filename = 'iri_oh_ver_{}.nc'
path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/'
filename = 'iri_ch1_ver_{}.nc'
ds_ver = xr.open_dataset(path+filename.format(orbit_num))
ds_ver.close()
ds_ver = ds_ver.update({'tp' : (['time'],
        (ds_ver.time - ds_ver.time[0])/(ds_ver.time[-1]-ds_ver.time[0]))})

# ds_ver['ver'] /= 0.55
# ds_ver['error2_retrieval'] /= 0.55**2
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

d = haversine(ds_ver.longitude[0], ds_ver.latitude[0], ds_ver.longitude, ds_ver.latitude)
ds_ver = ds_ver.update({'d' : (['time'], d, dict(units='km'))})

# %% gaussian fit the whole orbit
from characterise_agc_routine import process_file
f = path+filename.format(orbit_num)
ds_agc = process_file(f, save_file=False)
ds_ver = ds_ver.update(ds_agc)
ds_ver['thickness'] = ds_ver.peak_sigma.reduce(lambda x: 2*x, keep_attrs=True)
ds_ver['thickness_error'] = ds_ver.peak_sigma_error.reduce(lambda x: 2*x, keep_attrs=True)

ds_ver = ds_ver.swap_dims(dict(time='d'))
#%% First profile
from characterise_agc_routine import gauss
isel_args = dict(d=0)
ds_ver.ver.where(ds_ver.A_peak>-1).isel(**isel_args).plot(
    y='z', color='k', marker='.', ls='-', alpha=0.4, label='VER data')
ds_ver.error2_retrieval.pipe(np.sqrt).where(ds_ver.A_peak>-1).isel(**isel_args).plot(
    y='z', color='b', marker='', ls='-', alpha=0.4, label='VER error')
ds_ver.z.pipe(
    gauss, *ds_ver[['peak_intensity', 'peak_height', 'peak_sigma']].isel(
        **isel_args).to_array()).plot(
        y='z', color='C3', lw=4, label='Gaussian model'
        )
# ds_ver.ver.where(ds_ver.A_peak>0.8).isel(**isel_args).plot
plt.legend(loc='upper right')
plt.xlabel('VER [{}]'.format(ds_ver.ver.units))
plt.title('')
#%% fileter profiles
cond_peak_height = (ds_ver.peak_height>60e3)*(ds_ver.peak_height<95e3)
cond_chisq = ds_ver.chisq<2
cond_gauss = cond_peak_height * cond_chisq
#%%
if orbit == 10122:
    ds_ver = ds_ver.where(ds_ver.tp<0.4, drop=True)
elif orbit == 38710:
    ds_ver = ds_ver.where(ds_ver.tp>0.55, drop=True)

from characterise_agc_routine import gauss
fig, ax = plt.subplots(9,1, sharex=False, figsize=(6,15), 
    gridspec_kw=dict(height_ratios=[0.1, 1,1,1,1,1,1,1,1], hspace=0.5))
ax_twin = [ax[i].twinx() for i in range(0,len(ax))]

colorplot_args = dict(x='d', vmin=0, vmax=12e3, ylim=[60e3, 95e3], 
                cmap='viridis', add_colorbar=False)
#option 1:
# ds_ver.error2_retrieval.pipe(np.sqrt).where(
#     ds_ver.error2_retrieval!=0, other=1e-1).plot(ax=ax[1], **colorplot_args)
# ds_ver.ver.plot(ax=ax[2], **colorplot_args)

#option 2:
# ds_ver.error2_retrieval.pipe(np.sqrt).where(
#     ds_ver.error2_retrieval!=0, other=1e-1).where(
#         ds_ver.A_peak>0.8, other=6e3).plot(ax=ax[1], **colorplot_args)
# ds_ver.ver.plot(ax=ax[2], **colorplot_args)

#option 3:
ds_ver.error2_retrieval.pipe(np.sqrt).where(
        ds_ver.error2_retrieval!=0, other=1e-1).where(
            ds_ver.A_peak>0.8).plot(ax=ax[1], **colorplot_args)
ds_ver.ver.where(ds_ver.A_peak>0.8).plot(ax=ax[2], **colorplot_args)
colorplot_args['add_colorbar'] = True
cbar_kwargs = dict(cax=ax[0], shrink=0.6, orientation='horizontal')
ds_ver.z.pipe(gauss, ds_ver.peak_intensity, ds_ver.peak_height, ds_ver.peak_sigma).where(
    cond_gauss).plot(
    ax=ax[3], **colorplot_args, cbar_kwargs=cbar_kwargs)

lineplot_args = dict(x='d', color='k', alpha=0.9)
ds_ver.peak_intensity.plot(ax=ax[4], **lineplot_args, ylim=(0,3e4))
ds_ver.peak_height.plot(ax=ax[5], **lineplot_args, ylim=(8e4, 9e4))
ds_ver.thickness.plot(ax=ax[6], **lineplot_args, ylim=(0, 2e4))
ds_ver.zenith_intensity.plot(ax=ax[7], **lineplot_args, ylim=(0, 6e4))
# ds_ver.chisq.plot(ax=ax[7], **lineplot_args, ylim=(0, 80))
l_lon, = ds_ver.longitude.plot(ax=ax[8], label='longitude', x='d', c='k', ls='--', alpha=0.9)
l_lat, = ds_ver.latitude.plot(ax=ax[8], label='latitude', x='d', c='k', ls='-.', alpha=0.9)
l_sza, = ds_ver.sza.plot(ax=ax_twin[8], label='SZA', x='d', c='r', alpha=0.9)

fillplot_args = dict(color='r', alpha=0.3)
ax[4].fill_between(ds_ver.d, 
    ds_ver.peak_intensity + ds_ver.peak_intensity_error,
    ds_ver.peak_intensity - ds_ver.peak_intensity_error, **fillplot_args)
ax[5].fill_between(ds_ver.d, 
    ds_ver.peak_height + ds_ver.peak_height_error,
    ds_ver.peak_height - ds_ver.peak_height_error, **fillplot_args)
ax[6].fill_between(ds_ver.d, 
    ds_ver.thickness + ds_ver.thickness_error,
    ds_ver.thickness - ds_ver.thickness_error, **fillplot_args)
ax[7].fill_between(ds_ver.d, 
    ds_ver.zenith_intensity + ds_ver.zenith_intensity_error,
    ds_ver.zenith_intensity - ds_ver.zenith_intensity_error, **fillplot_args)

errorplot_args = dict(x='d', color='b', ylim=(0,1), alpha=0.6)
(ds_ver.peak_intensity_error/ds_ver.peak_intensity).plot(ax=ax_twin[4], **errorplot_args)
(ds_ver.peak_height_error/ds_ver.peak_height).plot(ax=ax_twin[5], **errorplot_args)
(ds_ver.thickness_error/ds_ver.thickness).plot(ax=ax_twin[6], **errorplot_args)
(ds_ver.zenith_intensity_error/ds_ver.zenith_intensity).plot(ax=ax_twin[7], **errorplot_args)

ax[1].set(title='VER error')
ax[2].set(title='Original VER')
ax[3].set(title='Gaussian modelled VER')
ax[4].set(title='Peak intensity $V_{peak}$', ylabel='[{}]'.format(ds_ver.peak_intensity.units))
ax[5].set(title='Peak height $z_{peak}$', ylabel='[{}]'.format(ds_ver.peak_height.units))
ax[6].set(title='Thickness $2 \sigma$', ylabel='[{}]'.format(ds_ver.thickness.units))
ax[7].set(title='Zenith intensity $V_{zenith}$', ylabel='[{}]'.format(ds_ver.zenith_intensity.units))
# ax[7].set(title='$\chi^2$', ylabel='')
ax[8].set(ylabel='[$^\circ$ N]\n[$^\circ$ E]')
ax[8].legend()
ax[-1].set(xlabel='Horizontal distance long the orbit [km]')

ax[0].tick_params(tick2On=True, label2On=True, tick1On=False, label1On=False)
[ax[i].set(xlabel='') for i in range(1,len(ax)-1)]
[ax[i].tick_params(bottom=True, labelbottom=False) for i in range(1,len(ax)-1)]
[ax[i].set_xlim(ax[1].get_xlim()) for i in range(1, len(ax))]
[ax_twin[i].set_axis_off() for i in range(4)]
[ax_twin[i].set_ylabel('rel. error', color=errorplot_args['color']) for i in range(3,len(ax)-1)]
[ax_twin[i].tick_params(axis='y', colors=errorplot_args['color']) for i in range(3,len(ax)-1)]
ax_twin[8].set_ylabel('SZA [$^\circ$]', color=l_sza.get_color())
ax_twin[8].tick_params(axis='y', colors=l_sza.get_color())

fig.suptitle('Orbit no. {} \n {} \n to \n {}'.format(
    orbit_num, ds_ver.time[0].values, ds_ver.time[-1].values))

#%% Two columns
if orbit == 10122:
    ds_ver = ds_ver.where(ds_ver.tp<0.4, drop=True)
elif orbit == 38710:
    ds_ver = ds_ver.where(ds_ver.tp>0.55, drop=True)

from characterise_agc_routine import gauss
fig, ax = plt.subplots(5,2, sharex=False, figsize=(10,10), 
    gridspec_kw=dict(height_ratios=[0.1, 1,1,1,1], top=0.85, hspace=0.25, wspace=0.5))
ax_twin = np.array([[ax[i,j].twinx() for j in range(2)] for i in range(len(ax))])
 
colorplot_args = dict(x='d', vmin=0, vmax=12e3, ylim=[60e3, 95e3], 
                cmap='viridis', add_colorbar=False)

#option 3:
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
ds_ver.peak_intensity.plot(ax=ax[1,1], **lineplot_args, ylim=(0,3e4))
ds_ver.peak_height.plot(ax=ax[2,1], **lineplot_args, ylim=(8e4, 9e4))
ds_ver.thickness.plot(ax=ax[3,1], **lineplot_args, ylim=(0, 2e4))
ds_ver.zenith_intensity.plot(ax=ax[4,1], **lineplot_args, ylim=(0, 6e4))
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
ax[1,1].set(title='Peak intensity $V_{peak}$', ylabel='[{}]'.format(ds_ver.peak_intensity.units))
ax[2,1].set(title='Peak height $z_{peak}$', ylabel='[{}]'.format(ds_ver.peak_height.units))
ax[3,1].set(title='Thickness $2 \sigma$', ylabel='[{}]'.format(ds_ver.thickness.units))
ax[4,1].set(title='Zenith intensity $V_{zenith}$', ylabel='[{}]'.format(ds_ver.zenith_intensity.units))
# ax[4,1].set(title='$\chi^2$', ylabel='')
ax[4,0].set(ylabel='[$^\circ$ N]\n[$^\circ$ E]')
ax[4,0].legend()
[ax[-1,j].set(xlabel='Horizontal distance long the orbit [km]') for j in range(2)]

ax[0,0].tick_params(tick2On=True, label2On=True, tick1On=False, label1On=False)
[[ax[i,j].set(xlabel='') for i in range(1,len(ax)-1)] for j in range(2)]
[[ax[i,j].tick_params(bottom=True, labelbottom=False) for i in range(1,len(ax)-1)] for j in range(2)]
[[ax[i, j].set_xlim(ax[1,0].get_xlim()) for i in range(1, len(ax))] for j in range(2)]
ax[0,1].set_axis_off()
ax_twin[0,1].set_axis_off()
[ax_twin[i,0].set_axis_off() for i in range(4)]
[ax_twin[i,1].set_ylabel('rel. error', color=errorplot_args['color']) for i in range(1,len(ax))]
[ax_twin[i,1].tick_params(axis='y', colors=errorplot_args['color']) for i in range(1,len(ax))]
ax_twin[4,0].set_ylabel('SZA [$^\circ$]', color=l_sza.get_color())
ax_twin[4,0].tick_params(axis='y', colors=l_sza.get_color())

fig.suptitle('Orbit no. {} \n {} \n to \n {}'.format(
    orbit_num, ds_ver.time[0].values, ds_ver.time[-1].values))

#%% visualise covariances
# (ds_ver.cov_peak_intensity_peak_height/ds_ver.peak_intensity_error/ds_ver.peak_height_error).plot(
#     label='$V_{peak}$ $z_{peak}$', x='d')
# (ds_ver.cov_peak_intensity_peak_sigma/ds_ver.peak_intensity_error/ds_ver.peak_sigma_error).plot(
#     label='$V_{peak}$ $\sigma$', x='d')
# (ds_ver.cov_peak_height_peak_sigma/ds_ver.peak_height_error/ds_ver.peak_sigma_error).plot(
#     label='$z_{peak}$ $\sigma$',x='d')
# plt.legend()
# plt.ylabel('correlation coefficient')
# plt.xlabel('fraction into the night')

#%% characterise the airglow layer by a gaussian fit
# from characterise_agc_routine import gauss, gauss_integral#, characterise_layer
# from scipy.optimize import curve_fit

# def characterise_layer(da_ver_profile, a0=5e3, mean0=85e3, sigma0=5e3, ver_error_profile=None):
#     y = da_ver_profile.dropna(dim='z')
#     x = y.z
#     if ver_error_profile is None:
#         absolute_sigma = False
#     else:
#         absolute_sigma = True

#     popt, pcov = curve_fit(gauss, x, y, 
#                     p0=[a0, mean0, sigma0], 
#                     sigma=ver_error_profile, absolute_sigma=absolute_sigma,
#                     # bounds=([0, 70e3, 0], [1e5, 100e3, 40e3]) #some reasonable ranges for the airglow characteristics
#                     )
#     residual = np.sqrt((y - gauss(x, *popt))**2).sum()
#     peak_intensity, peak_height, thickness_sigma = popt
#     peak_intensity_error, peak_height_error, thickness_sigma_error = np.sqrt(np.diag(pcov))
#     total_intensity, total_intensity_error = gauss_integral(peak_intensity, thickness_sigma, peak_intensity_error, thickness_sigma_error)
#     cov_peak_intensity_peak_height, cov_peak_height_peak_sigma = np.diag(pcov, 1)
#     cov_peak_intensity_peak_sigma = np.diag(pcov, 2)

#     return (peak_intensity, peak_height, abs(thickness_sigma), 
#         peak_intensity_error, peak_height_error, thickness_sigma_error,
#         residual, total_intensity, total_intensity_error, 
#         cov_peak_intensity_peak_height, cov_peak_intensity_peak_sigma, cov_peak_height_peak_sigma)

# isel_args = dict(time=0)

# da_ver_profile = ds_ver.ver.isel(**isel_args)
# ver_error_profile = ds_ver.error2_retrieval.isel(**isel_args).pipe(np.sqrt)
# ver_error_profile = xr.where(ver_error_profile==0, 1e-1, ver_error_profile).values
# result0 = characterise_layer(da_ver_profile, ver_error_profile=ver_error_profile)

# da_ver_profile = ds_ver.ver.isel(**isel_args).where(ds_ver.A_peak.isel(**isel_args)>0.8, drop=True)
# ver_error_profile = ds_ver.error2_retrieval.isel(**isel_args).pipe(np.sqrt).where(ds_ver.A_peak.isel(**isel_args)>0.8, drop=True)
# ver_error_profile = xr.where(ver_error_profile==0, 1e-1, ver_error_profile).values
# result1 = characterise_layer(da_ver_profile, ver_error_profile=ver_error_profile)

# da_ver_profile = ds_ver.ver.isel(**isel_args)
# ver_error_profile = ds_ver.error2_retrieval.isel(**isel_args).pipe(np.sqrt).where(ds_ver.A_peak.isel(**isel_args)>0.8, other=5e3)
# result2 = characterise_layer(da_ver_profile, ver_error_profile=ver_error_profile)

# da_ver_profile.plot(y='z', color='k', marker='.', ls='-', alpha=0.4, 
#     label='VER data')
# ds_ver.error2_retrieval.isel(**isel_args).pipe(np.sqrt).plot(
#     y='z', color='b', marker='', ls='-', alpha=0.4, label='VER error')
# ds_ver.z.pipe(gauss, *result0[:3]).plot(y='z', color='C2', lw=4,
#     label='Gaussian model org'#: \n $V_{peak}$=%1.0f [cm-3 s-1], \n $z_{peak}$=%1.0f [m], \n $\sigma$=%1.0f [m]' % tuple(result0[:3])
#     )
# ds_ver.z.pipe(gauss, *result1[:3]).plot(y='z', color='C3', lw=4,
#     label='Gaussian model MR filtered'#: \n $V_{peak}$=%1.0f [cm-3 s-1], \n $z_{peak}$=%1.0f [m], \n $\sigma$=%1.0f [m]' % tuple(result1[:3])
#     )
# ds_ver.z.pipe(gauss, *result2[:3]).plot(y='z', color='C4', lw=4,
#     label='Gaussian model error modified'#: \n $V_{peak}$=%1.0f [cm-3 s-1], \n $z_{peak}$=%1.0f [m], \n $\sigma$=%1.0f [m]' % tuple(result1[:3])
#     )

# # result = characterise_layer(da_ver_profile, ver_error_profile=None)
# # ds_ver.z.pipe(gauss, *result[:3]).plot(y='z', color='k', 
# #     label='Gaussian fit (LS): \n $V_{peak}$=%1.0f, $z_{peak}$=%1.0f, $\sigma$=%1.0f' % tuple(result[:3]))
# plt.legend(loc='upper right')
# plt.xlabel('VER [{}]'.format(ds_ver.ver.units))
