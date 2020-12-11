#%%
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
import glob
import matplotlib.pyplot as plt

# %%
def gauss(x, a, x0, sigma):
    '''
    x: data
    a: amplitude
    x0: mean of data
    sigma: FWHM
    '''
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def weighted_arithmetic_mean(x,y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y)) #this might become nan!
    return mean, sigma

def characterise_layer(da_ver_profile, a0=5e3, mean0=85e3, sigma0=5e3):
    y = da_ver_profile.dropna(dim='z')
    x = y.z
    popt, pcov = curve_fit(gauss, x, y, 
                    p0=[a0, mean0, sigma0], 
                    # bounds=([0, 70e3, 0], [1e4, 100e3, 40e3])
                    )
    
    amplitude, peak_height, thickness_FWHM = popt
    return amplitude, peak_height, thickness_FWHM

def process_file(f, save_file=True):
    result_1d = xr.open_dataset(f)
    ver_data = result_1d.ver#.where(result_1d.A_diag>0.8) #masking might cause issue in fitting!

    time_save = []
    amplitude, peak_height, thickness = [], [], []
    for i in range(len(ver_data.time)):
        print('{}/{}'.format(i, result_1d.orbit.item()))
        try:
            isel_args = dict(time = i)
            char = characterise_layer(ver_data.isel(**isel_args))
            amplitude.append(char[0])
            peak_height.append(char[1])
            thickness.append(char[2])
            time_save.append(ver_data.time.isel(**isel_args))
        except:
            pass
    if len(time_save) > 0:
        time_save = xr.concat(time_save, dim='time')
        ver_character = xr.Dataset({
            'time': (['time'], time_save),
            'longitude': (['time'], result_1d.longitude.sel(time=time_save)),
            'latitude': (['time'], result_1d.latitude.sel(time=time_save)),
            'amplitude': (['time'], amplitude, dict(units=result_1d.ver.units)),
            'peak_height': (['time'], peak_height, dict(units=result_1d.z.units)),
            'thickness': (['time'], thickness, dict(units=result_1d.z.units)),
        })
        if save_file:
            ver_character.to_netcdf('/home/anqil/Documents/osiris_database/iris_oh/airglow_character/agc_{}.nc'.format(
                str(result_1d.orbit.item()).zfill(6)))
#%%
if __name__ == '__main__':
    path = '/home/anqil/Documents/osiris_database/iris_oh/'
    ver_file_lst = glob.glob(path + '*nc')
    agc_file_lst = glob.glob(path + 'airglow_character/agc_*.nc')
    for f in ver_file_lst:
        orbit_num = f[-9:-3]
        if orbit_num in [k[-9:-3] for k in agc_file_lst]:
            pass
        else:
            process_file(f)

#%% plot stuff

# fig, ax = plt.subplots(3,1, sharex=True)
# plot_args = dict(x='time')
# ver_character.amplitude.plot(ax=ax[0], **plot_args)
# ver_character.peak_height.plot(ax=ax[1], **plot_args)
# ver_character.thickness.plot(ax=ax[2], **plot_args)
# # %%
# gauss_fit = []
# for a, x0, sigma in zip(ver_character.amplitude, 
#     ver_character.peak_height, ver_character.thickness):
#     gauss_fit.append(ver_data.z.pipe(gauss, a, x0, sigma))

# gauss_fit = xr.concat(gauss_fit, 'time').assign_coords(
#     time=ver_character.time).rename('gaussian ver')

# #%%
# fig, ax = plt.subplots(3,1, sharex=True, sharey=True)
# plot_args = dict(cmap='viridis', x='time', robust=True, vmin=0, vmax=7e3)
# ver_data.plot(ax=ax[0], **plot_args)
# gauss_fit.plot(ax=ax[1], **plot_args)
# (ver_data - gauss_fit).plot(ax=ax[2], x='time')

# ax[0].set(title='Original VER',
#             xlabel='')
# ax[1].set(title='Gaussian fitted VER',
#             xlabel='')
# ax[2].set(title='Org - Gauss')
# %%
