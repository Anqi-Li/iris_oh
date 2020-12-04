#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from oem_functions import linear_oem
from geometry_functions import pathl1d_iris

#%%
def invert_1d(orbit):
    orbit_num = str(orbit).zfill(6)
    filename = 'ir_slc_{}_ch{}.nc'.format(orbit_num, ch)
    ir = xr.open_dataset(path+filename).sel(pixel=slice(21,128))
    l1 = ir.data.where(ir.data.notnull(), drop=True).where(ir.sza>90, drop=True)
    time = l1.time
    error = ir.error.sel(time=time)
    tan_alt = ir.altitude.sel(time=time)

    im_lst = range(0,len(time))
    tan_low = 60e3
    tan_up = 95e3
    pixel_map = ((tan_alt>tan_low)*(tan_alt<tan_up))
    if ch==1:
        peak_apprx = 5e3 #7.5e3  # approximation of the airglow peak, used in Sa
    elif ch==3:
        peak_apprx = 5e4 

    #%% 1D inversion
    z = np.arange(tan_low-5e3, tan_up+30e3, 1e3) # m
    z_top = z[-1] + (z[-1]-z[-2]) #m
    xa = np.ones(len(z)) * 0 
    sigma_a = np.ones_like(xa) * peak_apprx
    # sigma_a[np.where(np.logical_or(z<tan_low, z>tan_up))] = 1e-1
    sigma_a = np.ones_like(xa) * peak_apprx
    n = (z<tan_low).sum()
    sigma_a[np.where(z<tan_low)] = np.logspace(-1, np.log10(peak_apprx), n)
    n = (z>tan_up).sum()
    sigma_a[np.where(z>tan_up)] = np.logspace(np.log10(peak_apprx), -1, n)
    Sa = np.diag(sigma_a ** 2)

    mr = []
    error2_retrieval = []
    ver = []
    y_fit = []
    for i in range(len(im_lst)):
        # print(i, '/', len(im_lst), '/', orbit)
        print('{}/{}/{}'.format(i, len(im_lst), orbit))
        isel_args = dict(time=im_lst[i])
        h = tan_alt.isel(**isel_args).where(pixel_map.isel(**isel_args), drop=True)
        K = pathl1d_iris(h.values, z, z_top)  *1e2 #m->cm
        y = l1.sel(pixel=h.pixel, time=h.time).values
        Se = np.diag(error.sel(pixel=h.pixel, time=h.time).values**2)
        x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
        ver.append(x)
        mr.append(A.sum(axis=1)) #sum over rows 
        error2_retrieval.append(np.diag(Sm))
        y_fit = y - K.dot(x)

    result_1d = xr.Dataset().update({
        'time': time[im_lst],
        'z': (['z',], z, {'units': 'm'}),
        'ver': (['time','z'], ver, {'long name': 'VER', 'units': 'photons cm-3 s-1'}),
        'mr': (['time','z'], mr),
        'error2_retrieval': (['time','z'], error2_retrieval),
        'y_fit': (['time','z'], y_fit),
        'latitude': (['time',], ir.latitude.sel(time=time)),
        'longitude': (['time',], ir.longitude.sel(time=time)),
        'orbit': ir.orbit,
        'channel': ir.channel,
        })

    result_1d.to_netcdf('~/Documents/osiris_database/iris_oh/iri_oh_ver_{}.nc'.format(orbit_num))

ch = 1
path = '~/Documents/osiris_database/globus/StrayLightCorrected/Channel{}/'.format(ch)
orbit = 3713
while orbit < 10000:
    print(orbit)
    try:
        invert_1d(orbit)
        orbit += 15
    except FileNotFoundError:
        orbit += 1
        print('invert the next orbit')
        
#%%
# mr_threshold = 0.8
# ver_1d_mean = result_1d.ver.where(result_1d.mr>mr_threshold).mean(dim='time')
# error_1d_mean = result_1d.error2_retrieval.where(result_1d.mr>mr_threshold).mean('time')

# #% plot VER results, profiles
# plt.figure()
# result_1d.ver.plot.line(y='z', marker='.', ls='', add_legend=False)
# ver_1d_mean.plot(y='z', color='k',ls='-',
#                     label='averaged with MR>.{}'.format(mr_threshold))
# (ver_1d_mean + np.sqrt(error_1d_mean)).plot(y='z', color='k', ls='--', label='averaged + error')
# (ver_1d_mean - np.sqrt(error_1d_mean)).plot(y='z', color='k', ls='--', label='averaged - error')

# plt.fill_betweenx(z, xa+np.sqrt(np.diag(Sa)), xa-np.sqrt(np.diag(Sa)), alpha=0.2, label='xa, Sa')
# # plt.fill_betweenx(z, ver_1d_mean+np.sqrt(error_1d_mean), ver_1d_mean-np.sqrt(error_1d_mean), alpha=0.9, label='x_hat, Sm')

# plt.gca().set(xlabel='VER / [photons cm-3 s-1]', 
#     #    ylabel='Altitdue grid',
#        title='1D retrieval of images {}'.format(im_lst))
# plt.legend()
# plt.show()

# # heatmap plot
# # from xhistogram.xarray import histogram
# # import matplotlib.pylab as pl
# # from matplotlib.colors import ListedColormap

# # # Choose colormap
# # cmap = pl.cm.viridis
# # # Get the colormap colors
# # my_cmap = cmap(np.arange(cmap.N))
# # # Set alpha
# # my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
# # # Create new colormap
# # my_cmap = ListedColormap(my_cmap)

# # h_z = histogram(result_1d.ver, bins=[np.linspace(-4e3, 8e3)], dim=['time'])
# # h_z.plot(vmax=im_lst[-1]*0.4, cmap=my_cmap)

# # ver_1d_mean.plot(y='z', color='k',ls='-',
# #                     label='averaged with MR>.{}'.format(mr_threshold))
# # (ver_1d_mean + np.sqrt(error_1d_mean)).plot(y='z', color='k', ls='--', label='averaged + error')
# # (ver_1d_mean - np.sqrt(error_1d_mean)).plot(y='z', color='k', ls='--', label='averaged - error')
# # plt.fill_betweenx(z, xa+np.sqrt(np.diag(Sa)), xa-np.sqrt(np.diag(Sa)), alpha=0.1, label='xa, Sa')

# # plt.gca().set(xlabel='VER / [photons cm-3 s-1]', 
# #     #    ylabel='Altitdue grid',
# #        title='1D retrieval of images {}'.format(im_lst))
# # plt.legend()
# # plt.show()

# # AVK plot
# A = xr.DataArray(A, coords=(z,z), dims=('row', 'col'), name='AVKs')
# plt.figure()
# A.plot.line(y='row', hue='col', add_legend=False)
# A.sum('col').plot(y='row', color='k', label='MR')
# # result_1d.mr.isel(time=-1).plot.line(y='z')
# plt.gca().set(xlabel='',
#             title='AVKs and MR \n {}'.format(im_lst[i]))
# plt.show()

# # countour plot
# fig, ax = plt.subplots(2,1,sharex=True, sharey=True)
# plt_args = dict(x='time', y='z')
# map_args = dict(cond=result_1d.mr>mr_threshold, drop=True)
# # result_1d.ver.where(**map_args).plot(ax=ax[0], vmin=0, vmax=7.5e3, **plt_args)
# result_1d.ver.where(**map_args).plot(ax=ax[0], vmin=0, vmax=peak_apprx, **plt_args)
# result_1d.error2_retrieval.where(**map_args).pipe(lambda x: 100*np.sqrt(x)/result_1d.ver
#             ).rename('% error').plot(ax=ax[1], vmin=0, vmax=20, **plt_args)
# plt.show()

# #%%
# result_1d_save = result_1d.copy()
