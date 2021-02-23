#%%
import numpy as np
import xarray as xr
import glob 
from multiprocessing import Pool
from random import shuffle

#%%
def Gauss_integral(amplitude, sigma):
    return amplitude * abs(sigma) * np.sqrt(2*np.pi)

def process_file(agc_file, new_file_pattern, save_file=False):
    with xr.open_dataset(agc_file) as ds:
        ds['thickness'] = abs(ds.thickness)
        ds['peak_height'] = abs(ds.peak_height)
        integral = Gauss_integral(ds.amplitude, ds.thickness)
        ds = ds.update({'total_intensity': integral})
        ds = ds.rename({'amplitude': 'peak_intensity',
                        'amplitude_error': 'peak_intensity_error'})
        if save_file:
            ds.to_netcdf(new_file_pattern)

#%%
if __name__ == '__main__':
    agc_path = '/home/anqil/Documents/osiris_database/iris_oh/airglow_character/'
    agc_filename_pattern = 'agc_{}.nc'
    agc_file_lst = glob.glob(agc_path + agc_filename_pattern.format('*'))

    gauss_path = '/home/anqil/Documents/osiris_database/iris_oh/gauss_character/'
    gauss_filename_pattern = 'gauss_{}.nc'
    def fun(f):
        orbit_num = f[-9:-3]
        gauss_file_lst = glob.glob(gauss_path + gauss_filename_pattern.format('*'))
        
        if orbit_num in [k[-9:-3] for k in gauss_file_lst]:
            print('{} already done'.format(orbit_num))
            pass
        else:
            print('process orbit {}'.format(orbit_num))
            process_file(f, 
                new_file_pattern=gauss_path+gauss_filename_pattern.format(orbit_num),
                save_file=True)
    
    shuffle(agc_file_lst)
    with Pool(processes=12) as p:
        p.map(fun, agc_file_lst)
    # fun(agc_file_lst[0])


# %%
