#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Jan 2021

@author: anqil
"""
# %%
from characterise_agc_routine import process_file as process_agc
from characterise_spectral_routine import process_file as process_sp
import glob
from multiprocessing import Pool

#%%
if __name__ == '__main__':
    
    ver_path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/oh/'
    ver_file_lst = glob.glob(ver_path + '*nc')

    character_path = '/home/anqil/Documents/osiris_database/iris_oh/'
    agc_filename_pattern = 'airglow_character/agc_{}.nc'
    agc_file_lst = glob.glob(character_path + agc_filename_pattern.format('*'))
    sp_filename_pattern = 'spectral_character/sp_{}.nc'
    sp_file_lst = glob.glob(character_path + sp_filename_pattern.format('*'))
    # for ver_file in ver_file_lst:
    def fun(ver_file):
        orbit_num = ver_file[-9:-3]

        # for airglow-layer character (agc) already done
        if orbit_num in [k[-9:-3] for k in agc_file_lst]:
            print('{}, done for agc'.format(orbit_num))
            # for spectral character (sp) already done
            if orbit_num in [k[-9:-3] for k in sp_file_lst]:
                print('{}, done for sp'.format(orbit_num))
                pass
            # for sp not yet done
            else:
                print('sp: process orbit {}'.format(orbit_num))
                process_sp(ver_file)
        # for agc not yet done
        else:
            print('agc: process orbit {}'.format(orbit_num))
            process_agc(ver_file, save_file=True, agc_file_pattern=character_path+agc_filename_pattern)
            print('sp: process orbit {}'.format(orbit_num))
            process_sp(ver_file, agc_f=character_path+agc_filename_pattern.format(orbit_num), 
                save_file=True, sp_file_pattern=character_path+sp_filename_pattern)

    with Pool(processes=6) as p:
        p.map(fun, ver_file_lst)

#%%
# def benchmark(orbit_num=None, num_of_orbits=1):
#     path = '/home/anqil/Documents/osiris_database/iris_oh/'
#     ver_file_lst = glob.glob(path + '*nc')
#     if orbit_num is None:
#         for ver_file in ver_file_lst[:num_of_orbits]:
#             a=process_agc(ver_file, save_file=False)
#             b=process_sp(ver_file, save_file=False)
#     else:
#         ver_file = next(ver_file for ver_file in ver_file_lst if orbit_num in ver_file)
#         a = process_agc(ver_file, save_file=False)
#         b = process_sp(ver_file, save_file=False)
#     return a, b
# %%
# %%time
# benchmark(orbit_num='005844')
# %%
