# %%
from OH_invert_routine import invert_1d
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
#%%
# %%time
ch = 1
path = '~/Documents/osiris_database/globus/StrayLightCorrected/Channel{}/'.format(ch)
# path = '~/Documents/sshfs/oso_extra_storage/StrayLightCorrected/Channel{}/'.format(ch)
orbit = 3713
result_1d = invert_1d(orbit, ch, path, save_file=False, im_lst=range(10))

#%%
isel_args = dict(time=0)
result_1d.isel(**isel_args).ver.plot(y='z')
#%% characterise the airglow layer by a gaussian fit
from characterise_ver_routine import characterise_layer, gauss

da_ver_profile = result_1d.ver.where(result_1d.A_diag>=0).isel(**isel_args)
popt = characterise_layer(da_ver_profile)

result_1d.ver.isel(**isel_args).plot(y='z', label='VER data')
result_1d.z.pipe(gauss, *popt).plot(y='z', label='Gaussian fit: \n a=%1.0f, x0=%1.0f, sigma=%1.0f' % tuple(popt))
result_1d.ver.isel(**isel_args).rolling(z=5, center=True).mean(
    ).plot(y='z', label='SMA 5km')
p0 = (5e3, 85e3, 5e3)
result_1d.z.pipe(gauss, *p0).plot(y='z', label='Initial guess: \n a=%1.0f, x0=%1.0f, sigma=%1.0f' % tuple(p0))
plt.legend()

#%%

# # %% plot VER results
# # heatmap plot
# from xhistogram.xarray import histogram
# import matplotlib.pylab as pl
# from matplotlib.colors import ListedColormap

# # Choose colormap
# cmap = pl.cm.viridis
# # Get the colormap colors
# my_cmap = cmap(np.arange(cmap.N))
# # Set alpha
# my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
# # Create new colormap
# my_cmap = ListedColormap(my_cmap)

# h_z = histogram(result_1d.ver, bins=[np.linspace(-4e3, 8e3)], dim=['time'])
# h_z.plot(vmax=h_z.sel(z=85e3).max(), cmap=my_cmap)

# ver_1d_mean = result_1d.ver.where(result_1d.A_diag>0.8).mean(dim='time', keep_attrs=True)
# error_1d_mean = result_1d.error2_retrieval.where(result_1d.A_diag>0.8).mean(dim='time', keep_attrs=True)

# ver_1d_mean.plot(y='z', color='k',ls='-',
#                     label='mean')
# (ver_1d_mean + np.sqrt(error_1d_mean)).plot(y='z', color='k', ls='--', label='mean + error')
# (ver_1d_mean - np.sqrt(error_1d_mean)).plot(y='z', color='k', ls='--', label='mean - error')

# plt.gca().set(xlabel='VER / [photons cm-3 s-1]', 
#     #    ylabel='Altitdue grid',
#        title='1D retrieval of images {}'.format(im_lst))
# plt.legend()
# plt.show()

# #%%
# # AVK plot
# isel_args = dict(time=0)
# result_1d.isel(**isel_args).A_diag.plot(y='z', label='diaganol')
# result_1d.isel(**isel_args).A_peak.plot(y='z', label='peak at row')
# plt.legend()
# plt.gca().twiny()
# result_1d.isel(**isel_args).A_peak_height.plot(y='z', color='k', ls=':', label='height at peak')
# plt.legend()
# # %%
