#%%
import numpy as np
import xarray as xr
import glob 
from multiprocessing import Pool
from random import shuffle
from characterise_agc_routine import process_file as fit_gauss_from_ver_file

#%%
def Gauss_integral(amplitude, sigma):
    return amplitude * abs(sigma) * np.sqrt(2*np.pi)

def process_file(agc_file, new_file_pattern, save_file=False):
    with xr.open_dataset(agc_file) as ds:
        # ds['thickness'] = abs(ds.thickness)
        # ds['peak_height'] = abs(ds.peak_height)
        # integral = Gauss_integral(ds.amplitude, ds.thickness)
        # ds = ds.update({'total_intensity': integral})
        # ds = ds.rename({'amplitude': 'peak_intensity',
        #                 'amplitude_error': 'peak_intensity_error'})
        ds['peak_intensity'] *= np.pi*4
        ds['peak_intensity_error'] *= np.pi*4
        ds['cov_peak_intensity_peak_sigma'] *= np.pi*4
        ds['cov_peak_intensity_peak_height'] *= np.pi*4
        ds['zenith_intensity'] *= np.pi*4
        ds['zenith_intensity_error'] *= np.pi*4

        if save_file:
            ds.to_netcdf(new_file_pattern)

#%%
if __name__ == '__main__':
    old_path = '/home/anqil/Documents/osiris_database/iris_oh/gauss_character/A_filtered/'
    old_filename_pattern = 'gauss_{}.nc'
    old_file_lst = glob.glob(old_path + old_filename_pattern.format('*'))

    new_path = '/home/anqil/Documents/osiris_database/iris_oh/gauss_character/4pi/'
    new_filename_pattern = 'gauss_{}.nc'
    def fun(f):
        orbit_num = f[-9:-3]
        new_file_lst = glob.glob(new_path + new_filename_pattern.format('*'))
        try:
            if orbit_num in [k[-9:-3] for k in new_file_lst]:
                print('{} already done'.format(orbit_num))
                pass
            else:
                print('process orbit {}'.format(orbit_num))
                process_file(f, 
                    new_file_pattern=new_path+new_filename_pattern.format(orbit_num),
                    save_file=True)
        # except OSError:    
        #     ver_path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/'
        #     ver_file = ver_path + 'iri_ch1_ver_{}.nc'.format(orbit_num)
        #     gauss_path = '/home/anqil/Documents/osiris_database/iris_oh/'
        #     gauss_filename_pattern = 'gauss_character/A_filtered/gauss_{}.nc'

        #     _ = fit_gauss_from_ver_file(ver_file, save_file=True, agc_file_pattern=gauss_path + gauss_filename_pattern)
        except:
            raise
    shuffle(old_file_lst)
    with Pool(processes=8) as p:
        p.map(fun, old_file_lst)
    # fun(old_file_lst[0])


# %%
