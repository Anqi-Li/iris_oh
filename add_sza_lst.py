#%%
import numpy as np
import xarray as xr
import glob
from os import listdir
from random import shuffle
from multiprocessing import Pool

#%%
if __name__ == '__main__':
    orbit_year = xr.open_dataset('/home/anqil/Documents/osiris_database/odin_rough_orbit_year.nc')
    orbit_year.close()

    ch = 1
    # % Inverted VER files
    # if ch == 3:
    #     path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/'
    # elif ch == 1:
    #     path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/oh/'

    path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel{}/nightglow/'.format(ch)
    files_ver = [f for f in listdir(path_ver) if 'nc' in f]
    #files_ver.sort()
    #shuffle(files_ver)

    # Check limb files
    # path_limb = '/home/anqil/Documents/sshfs/oso_extra_storage/StrayLightCorrected/Channel{}/'.format(ch)
    
    dir_save = path_ver + '4pi/iri_ch{}_ver_{}.nc'

    # # for year in range(2001,2002):
    # def fun(year):
    #     files_ver_year = [f for f in files_ver 
    #         if int(f[-9:-3]) in range(*tuple(orbit_year.orbit.sel(
    #             year=slice(year,year+1)).values))]
    #     for i in range(len(files_ver_year)):
    #         orbit_num = files_ver_year[i][-9:-3]
    #         print(orbit_num)
    #         saved_file_lst = glob.glob(dir_save.format(ch, ch, '*'))
    #         if dir_save.format(ch, ch, orbit_num) in saved_file_lst:
    #             print('orbit {} already exist'.format(orbit_num))
    #             pass 
    #         else:   
    #             with xr.open_dataset(path_ver+'iri_oh_ver_{}.nc'.format(orbit_num)) as ds_ver:
    #                 with xr.open_dataset(path_limb+'ir_slc_{}_ch{}.nc'.format(orbit_num, ch)) as ds_limb:
    #                     ds_ver = ds_ver.assign(ds_limb.reindex_like(ds_ver)[['sza', 'apparent_solar_time']])
    #                     ds_ver['ver'] /= 0.5515
    #                     ds_ver['error2_retrieval'] /= 0.5515**2
    #                     ds_ver['error2_smoothing'] /= 0.5515**2

    #                     ds_ver.to_netcdf(
    #                         dir_save.format(ch, ch, orbit_num))
    
    def fun(year):
        files_ver_year = [f for f in files_ver 
            if int(f[-9:-3]) in range(*tuple(orbit_year.orbit.sel(
                year=slice(year,year+1)).values))]
        for i in range(len(files_ver_year)):
            orbit_num = files_ver_year[i][-9:-3]
            print(orbit_num)
            saved_file_lst = glob.glob(dir_save.format(ch, '*'))
            if dir_save.format(ch, orbit_num) in saved_file_lst:
                print('orbit {} already exist'.format(orbit_num))
                pass 
            else:   
                with xr.open_dataset(path_ver + files_ver_year[i]) as ds_ver:
                    ds_ver['ver'] *= 4*np.pi
                    ds_ver['error2_retrieval'] *= (4*np.pi)**2
                    ds_ver['error2_smoothing'] *= (4*np.pi)**2

                    ds_ver.to_netcdf(
                        dir_save.format(ch, orbit_num))

    with Pool(processes=8) as p:
        p.map(fun, range(2009, 2018))
# %%
