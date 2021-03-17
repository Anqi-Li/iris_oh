#%%
import numpy as np
import xarray as xr
import glob
from os import listdir
from random import shuffle
from multiprocessing import Pool

#%%
if __name__ == '__main__':

    ch = 1
    # % Check inverted VER files
    if ch == 3:
        path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/'
    elif ch == 1:
        path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/oh/'

    files_ver = [f for f in listdir(path_ver) if 'nc' in f]
    #files_ver.sort()
    #shuffle(files_ver)

    # Check limb files
    path_limb = '/home/anqil/Documents/sshfs/oso_extra_storage/StrayLightCorrected/Channel{}/'.format(ch)

    path_saved = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel{}/nightglow/iri_ch{}_ver_{}.nc'

    #for i in range(len(files_ver)):
    def fun(i):
        orbit_num = files_ver[i][-9:-3]
        print(orbit_num)
        saved_file_lst = glob.glob(path_saved.format(ch, ch, '*'))
        if path_saved.format(ch, ch, orbit_num) in saved_file_lst:
            print('orbit {} already exist'.format(orbit_num))
            pass 
        else:   
            with xr.open_dataset(path_ver+files_ver[i]) as ds_ver:
                with xr.open_dataset(path_limb+'ir_slc_{}_ch{}.nc'.format(orbit_num, ch)) as ds_limb:
                    ds_ver = ds_ver.assign(
                        ds_limb.reindex_like(ds_ver)[['sza', 'apparent_solar_time']]).to_netcdf(
                        '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel{}/nightglow/iri_ch{}_ver_{}.nc'.format(
                        ch, ch, orbit_num
                ))

    with Pool(processes=12) as p:
        p.map(fun, range(len(files_ver)))
# %%
