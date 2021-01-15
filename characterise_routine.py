# %%
from characterise_agc_routine import process_file as process_agc
from characterise_spectral_routine import process_file as process_sp
import glob

#%%
if __name__ == '__main__':
    path = '/home/anqil/Documents/osiris_database/iris_oh/'
    ver_file_lst = glob.glob(path + '*nc')
    agc_file_lst = glob.glob(path + 'airglow_character/agc_*.nc')
    sp_file_lst = glob.glob(path + 'spectral_character/sp_*.nc')
    for f in ver_file_lst:
        orbit_num = f[-9:-3]

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
                process_sp(f)
        # for agc not yet done
        else:
            print('agc: process orbit {}'.format(orbit_num))
            process_agc(f)
            print('sp: process orbit {}'.format(orbit_num))
            process_sp(f)

#%%
def benchmark(orbit_num=None, num_of_orbits=1):
    path = '/home/anqil/Documents/osiris_database/iris_oh/'
    ver_file_lst = glob.glob(path + '*nc')
    if orbit_num is None:
        for f in ver_file_lst[:num_of_orbits]:
            a=process_agc(f, save_file=False)
            b=process_sp(f, save_file=False)
    else:
        f = next(f for f in ver_file_lst if orbit_num in f)
        a=process_agc(f, save_file=False)
        b=process_sp(f, save_file=False)
    return a, b
# %%
# %%time
# benchmark(orbit_num='006196')
# %%
