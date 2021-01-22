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
ref_orbit = orbit_year.orbit
ref_year = orbit_year.year

# %% Check downloaded limb files
ch = 1
path_limb = '/home/anqil/Documents/sshfs/oso_extra_storage/StrayLightCorrected/Channel{}/'.format(ch)
files_limb = [f for f in listdir(path_limb) if 'nc' in f]
orbits_downloaded = [int(s[-13:-7]) for s in files_limb]

# % Check inverted VER files
path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/oh/'
files_ver = [f for f in listdir(path_ver) if 'nc' in f]
orbits_ver = [int(s[-9:-3]) for s in files_ver]

path_char = '/home/anqil/Documents/osiris_database/iris_oh/'
# % Check airglow character files
path_agc = path_char + 'airglow_character/'# + 'archive/'
files_agc = [f for f in listdir(path_agc) if 'nc' in f]
orbits_agc = [int(s[-9:-3]) for s in files_agc]

# % Check spectral character files
path_sp = path_char + 'spectral_character/'# + 'archive/'
files_sp = [f for f in listdir(path_sp) if 'nc' in f]
orbits_sp = [int(s[-9:-3]) for s in files_sp]

# % visualise
# orbit_bins = np.linspace(3e3, 4e4)
orbit_bins = ref_orbit
plt.hist(orbits_downloaded, bins=orbit_bins, label='Downloaded to OSO')
plt.hist(orbits_ver, bins=orbit_bins, label='Inverted VER')
plt.hist(orbits_agc, bins=orbit_bins, label='Layer character')
plt.hist(orbits_sp, bins=orbit_bins, label='Spectral character')
plt.legend()
plt.xticks(ref_orbit, ref_year.values, rotation=40)
plt.ylabel('Num. of orbits processed')
plt.show()

# %%