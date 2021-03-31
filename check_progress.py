#%%
import numpy as np
import xarray as xr
import glob
from os import listdir
import matplotlib.pyplot as plt

# %% rough estimates of odin year-orbits
# from astropy.time import Time
# orbits_rough = xr.open_dataset('/home/anqil/Documents/osiris_database/odin_rough_orbit.nc')
# orbits_rough = orbits_rough.set_coords('orbit').swap_dims({'mjd': 'orbit'}).reset_coords()
# orbits_rough = orbits_rough.update({'time': ('orbit', Time(orbits_rough.mjd, format='mjd').datetime64)})
# ref_year = np.arange(2001, 2019)
# ref_orbit = []
# for year in ref_year:
#     try:
#         ref_orbit.append(next(t.orbit.item() for t in orbits_rough.time if t.dt.year==year))
#     except:
#         break
    
# check_orbit = 8181
# check_year = ref_year[np.array(ref_orbit).searchsorted(check_orbit)-1]
# %% rough estimates of odin year-orbits
orbit_year = xr.open_dataset('/home/anqil/Documents/osiris_database/odin_rough_orbit_year.nc')
orbit_year.close()
ref_orbit = orbit_year.orbit
ref_year = orbit_year.year

#%%
# ch = 1
# path_limb = '/home/anqil/Documents/sshfs/oso_extra_storage/StrayLightCorrected/Channel{}/'.format(ch)
# files_limb = [f for f in listdir(path_limb) if 'nc' in f]
# files_limb.sort()
# orbits_limb = [int(s[-13:-7]) for s in files_limb]
# orbits_save_year_idx = []
# for i in range(len(ref_orbit)):
#     orbits_save_year_idx.append(abs(np.array(orbits_limb)-ref_orbit[i].values).argmin())

# with xr.open_mfdataset([path_limb+f for f in np.array(files_limb)[orbits_save_year_idx]]) as mds:
#     print(mds)
    
# %% Check downloaded limb files
ch = 1
path_limb = '/home/anqil/Documents/sshfs/oso_extra_storage/StrayLightCorrected/Channel{}/'.format(ch)
files_limb = [f for f in listdir(path_limb) if 'nc' in f]
orbits_limb = [int(s[-13:-7]) for s in files_limb]

# % Check inverted VER files
if ch == 3:
    path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/'
elif ch == 1:
    path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/oh/'
    path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/'

files_ver = [f for f in listdir(path_ver) if 'nc' in f]
orbits_ver = [int(s[-9:-3]) for s in files_ver]

path_char = '/home/anqil/Documents/osiris_database/iris_oh/'
# % Check airglow character files
path_agc = path_char + 'gauss_character/A_filtered/' #+ 'archive/'
files_agc = [f for f in listdir(path_agc) if 'nc' in f]
orbits_agc = [int(s[-9:-3]) for s in files_agc]

# % Check spectral character files
# path_sp = path_char + 'spectral_character/' + 'archive/'
# files_sp = [f for f in listdir(path_sp) if 'nc' in f]
# orbits_sp = [int(s[-9:-3]) for s in files_sp]

# % visualise
# orbit_bins = np.linspace(3e3, 4e4)
orbit_bins = ref_orbit
hist_args = dict(histtype='step', bins=orbit_bins,)
x0,*_ = plt.hist(orbits_limb, **hist_args, label='Limb radiance', color='k', alpha=0.3)
x1,*_ = plt.hist(orbits_ver, **hist_args, label='Inverted VER', color='C3', alpha=0.8)

x2,*_ = plt.hist(orbits_agc, **hist_args, label='Layer character', color='k')
# plt.hist(orbits_sp, **hist_args, label='Spectral character')

# plt.step(orbit_bins[1:], x0*0.5, label='50% of limb', where='pre', color='k', ls=':')

plt.legend()
plt.xticks(ref_orbit[1:], ref_year.values[1:], rotation=40)
plt.ylabel('Num. of orbits available')
plt.xlabel('Year')

plt.show()

# %% make monthly files
import pandas as pd
from astropy.time import Time

time_stamp = pd.date_range(start='2007-12-31 23:59:59', end='2008-12-31 23:59:59', freq='M')
with xr.open_dataset('~/Documents/osiris_database/odin_rough_orbit.nc') as rough_orbit:
    rough_orbit = rough_orbit.rename({'mjd':'time'}).assign(
        time=Time(rough_orbit.mjd, format='mjd').datetime64
        ).interp(time=time_stamp).astype(int).orbit

def remove_duplicate(ds):
    ds.sel(time=~ds.indexes['time'].duplicated())
    return ds
# i = 0
files_ver = [f for f in listdir(path_ver) if 'nc' in f]

for i in [2, 8]:
    print(i+1)
    try:
        with xr.open_mfdataset(
            [path_ver + f for f in files_ver 
            if int(f[-9:-3]) in range(*tuple(
                rough_orbit.isel(time=slice(i,i+2)).values)
                )], preprocess=remove_duplicate) as mds:
            mds.to_netcdf(
                '~/Documents/osiris_database/iris_oh/VER/iri_ch1_2008{}.nc'.format(
                str(i+1).zfill(2)
                ))
    except:
        # pass
        raise