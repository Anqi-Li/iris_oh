#%%
import numpy as np
import xarray as xr
import glob 
from multiprocessing import Pool
from astropy.time import Time
import pandas as pd
from os import listdir
# %%
def zonal_average(mds, dlat=20):
    latitude_bins = np.arange(-90, 90+dlat, dlat)
    latitude_labels = latitude_bins[1:]-dlat/2

    groups = mds.ver.where(mds.A_diag>0.8).groupby_bins(
        mds.latitude, bins=latitude_bins, labels=latitude_labels)
    mean = groups.mean('time').rename('mean_ver')
    std = groups.std('time').rename('std_ver')
    count = groups.count('time').rename('count_ver')
    return mean, std, count

#%%
if __name__ == '__main__':
    time_stamp = pd.date_range(start='2001-01', end='2018', freq='M')
    
    with xr.open_dataset('~/Documents/osiris_database/odin_rough_orbit.nc') as rough_orbit:
        # rough_orbit = rough_orbit.update({'time': ('mjd', Time(rough_orbit.mjd, format='mjd').datetime64)}).set_coords('time')
        rough_orbit = rough_orbit.rename({'mjd':'time'}).assign(time=Time(rough_orbit.mjd, format='mjd').datetime64)

        rough_orbit = rough_orbit.interp(time=time_stamp, kwargs=dict(fill_value='extrapolate')).round()
        rough_orbit = rough_orbit.where(rough_orbit.orbit>0, drop=True).astype(int)

    def clima(year_int):
        year = str(year_int)
        rough_orbit_in_previous_year = rough_orbit.sel(time=str(int(year)-1)).isel(time=-1)
        rough_orbit_in_year = rough_orbit.sel(time=year)
        path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/oh/'
        files_ver = [f for f in listdir(path_ver) if 'nc' in f]
        mean_year, std_year, count_year = [], [], []
        time_coord = []
        for time_idx in range(len(rough_orbit_in_year.time)):
            print(rough_orbit_in_year.time[time_idx].values)
            if time_idx == 0:
                orbits_ver = [f for f in files_ver 
                    if int(f[-9:-3]) > rough_orbit_in_previous_year.orbit 
                        and int(f[-9:-3]) < rough_orbit_in_year.orbit.isel(time=time_idx)]
            else:
                orbits_ver = [f for f in files_ver 
                    if int(f[-9:-3]) > rough_orbit_in_year.orbit.isel(time=time_idx-1) 
                        and int(f[-9:-3]) < rough_orbit_in_year.orbit.isel(time=time_idx)]

            with xr.open_mfdataset([path_ver+f for f in orbits_ver]) as mds:
                mean, std, count = zonal_average(mds)
            mean_year.append(mean)
            std_year.append(std)
            count_year.append(count)
            time_coord.append(rough_orbit_in_year.time[time_idx].values)
        mean_year = xr.concat(mean_year, dim='time').assign_coords(time=time_coord)
        std_year = xr.concat(std_year, dim='time').assign_coords(time=time_coord)
        count_year = xr.concat(count_year, dim='time').assign_coords(time=time_coord)

        print('saving year {}'.format(year))
        path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/'
        filename = 'ver_clima_{}.nc'.format(year)
        mean_year.to_netcdf(path+filename, mode='w')
        std_year.to_netcdf(path+filename, mode='a')
        count_year.to_netcdf(path+filename, mode='a')

    with Pool(processes=6) as p:
        p.map(clima, range(2003, 2008))



# %%
