#%%
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
import glob
import matplotlib.pyplot as plt
from multiprocessing import Pool
from random import shuffle

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
                    # bounds=([0, 70e3, 0], [1e5, 100e3, 40e3]) #some reasonable ranges for the airglow characteristics
                    )
    residual = np.sqrt((y - gauss(x, *popt))**2).sum()
    amplitude, peak_height, thickness_FWHM = popt
    amplitude_error, peak_height_error, thickness_FWHM_error = np.sqrt(np.diag(pcov))
    return (amplitude, peak_height, thickness_FWHM, 
        amplitude_error, peak_height_error, thickness_FWHM_error,
        residual)

def process_file(ver_file, save_file=False, agc_file_pattern=None):
    result_1d = xr.open_dataset(ver_file)
    result_1d.close()
    ver_data = result_1d.ver#.where(result_1d.A_diag>0.8) #masking might cause issue in fitting!

    time_save = []
    amplitude, peak_height, thickness = [], [], []
    amplitude_error, peak_height_error, thickness_error = [], [], []
    residual = []
    for i in range(len(ver_data.time)):
        # print('{}/{}'.format(i, result_1d.orbit.item()))
        try:
            isel_args = dict(time = i)
            char = characterise_layer(ver_data.isel(**isel_args))
            amplitude.append(char[0])
            peak_height.append(char[1])
            thickness.append(char[2])
            amplitude_error.append(char[3])
            peak_height_error.append(char[4])
            thickness_error.append(char[5])
            residual.append(char[6])
            time_save.append(ver_data.time.isel(**isel_args))
        except:
            pass
    if len(time_save) > 0:
        time_save = xr.concat(time_save, dim='time')
        agc_character = xr.Dataset({
            'time': (['time'], time_save),
            'longitude': (['time'], result_1d.longitude.sel(time=time_save)),
            'latitude': (['time'], result_1d.latitude.sel(time=time_save)),
            'amplitude': (['time'], amplitude, dict(units=result_1d.ver.units)),
            'peak_height': (['time'], peak_height, dict(units=result_1d.z.units)),
            'thickness': (['time'], thickness, dict(units=result_1d.z.units)),
            'amplitude_error': (['time'], amplitude_error, dict(units=result_1d.ver.units)),
            'peak_height_error': (['time'], peak_height_error, dict(units=result_1d.z.units)),
            'thickness_error': (['time'], thickness_error, dict(units=result_1d.z.units)),
            'residual':(['time'], residual, dict(units=result_1d.ver.units)),
            'orbit': result_1d.orbit,
            'channel': result_1d.channel,
        })
        if save_file:
            orbit_num = str(result_1d.orbit.item()).zfill(6)
            print('saving agc file {}'.format(orbit_num))
            if agc_file_pattern == None:
                agc_file_pattern = '/home/anqil/Documents/osiris_database/iris_oh/airglow_character/agc_{}.nc'
            else:
                agc_character.to_netcdf(agc_file_pattern.format(orbit_num))
        return agc_character

#%%
if __name__ == '__main__':
    ver_path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/oh/'
    ver_file_lst = glob.glob(ver_path + '*nc')
    agc_path = '/home/anqil/Documents/osiris_database/iris_oh/'
    agc_filename_pattern = 'airglow_character/agc_{}.nc'
    # agc_file_lst = glob.glob(agc_path + agc_filename_pattern.format('*'))
    # for f in ver_file_lst:
    def fun(f):
        orbit_num = f[-9:-3]
        agc_file_lst = glob.glob(agc_path + agc_filename_pattern.format('*'))
        
        if orbit_num in [k[-9:-3] for k in agc_file_lst]:
            print('{} already done'.format(orbit_num))
            pass
        else:
            print('process orbit {}'.format(orbit_num))
            process_file(f, save_file=True, agc_file_pattern=agc_path + agc_filename_pattern)
            # process_file(f, save_file=False)
    
    shuffle(ver_file_lst)
    with Pool(processes=6) as p:
        p.map(fun, ver_file_lst)
# %%
