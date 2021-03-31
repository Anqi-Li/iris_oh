#%%
import numpy as np
import xarray as xr
import glob 
from multiprocessing import Pool
from astropy.time import Time
import pandas as pd
from os import listdir

# %%
def zonal_average(mds, var=None, dlat=20):
    latitude_bins = np.arange(-90, 90+dlat, dlat)
    latitude_labels = latitude_bins[1:]-dlat/2

    if var == None:
        groups = mds.groupby_bins(
            mds.latitude, bins=latitude_bins, labels=latitude_labels)
        mean = groups.mean('time').rename({k: 'mean_{}'.format(k) for k in mds.keys()})
        std = groups.std('time').rename({k: 'std_{}'.format(k) for k in mds.keys()})
        count = groups.count('time').rename({k: 'count_{}'.format(k) for k in mds.keys()})
    else:
        groups = mds[var].groupby_bins(
            mds.latitude, bins=latitude_bins, labels=latitude_labels)
        mean = groups.mean('time').rename('mean_{}'.format(var))
        std = groups.std('time').rename('std_{}'.format(var))
        count = groups.count('time').rename('count_{}'.format(var))

    return mean, std, count

#%%
if __name__ == '__main__':
    time_stamp = pd.date_range(start='2001-01', end='2018', freq='D')
    
    with xr.open_dataset('~/Documents/osiris_database/odin_rough_orbit.nc') as rough_orbit:
        # rough_orbit = rough_orbit.update({'time': ('mjd', Time(rough_orbit.mjd, format='mjd').datetime64)}).set_coords('time')
        rough_orbit = rough_orbit.rename({'mjd':'time'}).assign(time=Time(rough_orbit.mjd, format='mjd').datetime64)

        rough_orbit = rough_orbit.interp(time=time_stamp, kwargs=dict(fill_value='extrapolate')).round()
        rough_orbit = rough_orbit.where(rough_orbit.orbit>0, drop=True).astype(int)
    path_limb = '/home/anqil/Documents/sshfs/oso_extra_storage/StrayLightCorrected/Channel1/'
    files_limb = [f for f in listdir(path_limb) if 'nc' in f]
    path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/oh/'
    # path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/'
    files_ver = [f for f in listdir(path_ver) if 'nc' in f]
    path_agc = '/home/anqil/Documents/osiris_database/iris_oh/gauss_character/'
    files_agc = [f for f in listdir(path_agc) if 'nc' in f]    
    def clima_OH(year_int):
        year = str(year_int)
        if year_int > 2001:
            rough_orbit_in_previous_year = rough_orbit.sel(time=str(int(year)-1)).isel(time=-1)
        elif year_int == 2001:
            rough_orbit_in_previous_year = rough_orbit.isel(time=0)

        rough_orbit_in_year = rough_orbit.sel(time=year)
        
        mean_year, std_year, count_year = [], [], []
        time_coord = []
        for time_idx in range(len(rough_orbit_in_year.time)):
            print(rough_orbit_in_year.time[time_idx].values)
            if time_idx == 0:
                orbits_ver = [f for f in files_ver 
                    if int(f[-9:-3]) > rough_orbit_in_previous_year.orbit 
                        and int(f[-9:-3]) < rough_orbit_in_year.orbit.isel(time=time_idx)]
                orbits_limb = [f for f in files_limb 
                    if int(f[-13:-7]) > rough_orbit_in_previous_year.orbit 
                        and int(f[-13:-7]) < rough_orbit_in_year.orbit.isel(time=time_idx)]
            else:
                orbits_ver = [f for f in files_ver 
                    if int(f[-9:-3]) > rough_orbit_in_year.orbit.isel(time=time_idx-1) 
                        and int(f[-9:-3]) < rough_orbit_in_year.orbit.isel(time=time_idx)]
                orbits_limb = [f for f in files_limb 
                    if int(f[-13:-7]) > rough_orbit_in_year.orbit.isel(time=time_idx-1) 
                        and int(f[-13:-7]) < rough_orbit_in_year.orbit.isel(time=time_idx)]

            try:
                with xr.open_mfdataset([path_ver+f for f in orbits_ver], concat_dim="time",
                    data_vars='minimal', coords='minimal', compat='override') as mds:
                    with xr.open_mfdataset([path_limb+f for f in orbits_limb], concat_dim="time",
                        data_vars='minimal', coords='minimal', compat='override') as mds_limb:
                        mds.load()
                        mds_limb.load()
                        mds = mds.sel(time=~mds.indexes['time'].duplicated())
                        mds_limb = mds_limb.reindex_like(mds)[['latitude','sza','apparent_solar_time']]
                        mds = mds.combine_first(mds_limb)
                        
                        sza_min = 96 # 97.8(60km) 99.8(95km) 99.6(90km) 
                        am_pm = 'All' # define AM or PM

                        sza_cond = mds.sza > sza_min 
                        if am_pm == 'AM':
                            lst_cond = mds.apparent_solar_time < 12 
                        elif am_pm == 'PM':
                            lst_cond = mds.apparent_solar_time > 12 
                        elif am_pm == 'All':
                            lst_cond = mds.apparent_solar_time > 0
                        ver_cond = (mds.A_diag>0.8)*(mds.mr>0.8)*(mds.error2_retrieval<3e6)
                        mds = mds[['ver']].where(ver_cond * sza_cond * lst_cond).combine_first(
                            mds[['latitude','sza','apparent_solar_time']].where(sza_cond * lst_cond))
                        mean, std, count = zonal_average(mds)
                mean_year.append(mean)
                std_year.append(std)
                count_year.append(count)
                time_coord.append(rough_orbit_in_year.time[time_idx].values)
            except OSError:
                pass
            except ValueError:
                pass
        mean_year = xr.concat(mean_year, dim='time').assign_coords(time=time_coord)
        std_year = xr.concat(std_year, dim='time').assign_coords(time=time_coord)
        count_year = xr.concat(count_year, dim='time').assign_coords(time=time_coord)

        print('saving VER year {}'.format(year))
        path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/daily/'
        filename = '{}_{}_daily_halfglobe_ver_clima_{}.nc'.format(am_pm, sza_min, year)
        xr.merge([mean_year, std_year, count_year]).assign_attrs(
            am_pm=am_pm, sza_min=sza_min).to_netcdf(
            path+filename, mode='w')
        # mean_year.to_netcdf(path+filename, mode='w')
        # std_year.to_netcdf(path+filename, mode='a')
        # count_year.to_netcdf(path+filename, mode='a')

    def clima_agc(year_int):
        # year_int = 2001
        year = str(year_int)
        if year_int > 2001:
            rough_orbit_in_previous_year = rough_orbit.sel(time=str(int(year)-1)).isel(time=-1)
        elif year_int == 2001:
            rough_orbit_in_previous_year = rough_orbit.isel(time=0)

        rough_orbit_in_year = rough_orbit.sel(time=year)

        mean_year, std_year, count_year = [], [], []
        cond_amplitude_year, cond_thickness_year, cond_peak_height_year, cond_residual_year = [], [], [], []
        cond_verz_year = []
        time_coord = []
        for time_idx in range(len(rough_orbit_in_year.time)):
            print(rough_orbit_in_year.time[time_idx].values)
            if time_idx == 0:
                orbits_agc = [f for f in files_agc 
                    if int(f[-9:-3]) > rough_orbit_in_previous_year.orbit 
                        and int(f[-9:-3]) < rough_orbit_in_year.orbit.isel(time=time_idx)]
                orbits_ver = [f for f in files_ver 
                    if int(f[-9:-3]) > rough_orbit_in_previous_year.orbit 
                        and int(f[-9:-3]) < rough_orbit_in_year.orbit.isel(time=time_idx)]
            else:
                orbits_agc = [f for f in files_agc 
                    if int(f[-9:-3]) > rough_orbit_in_year.orbit.isel(time=time_idx-1) 
                        and int(f[-9:-3]) < rough_orbit_in_year.orbit.isel(time=time_idx)]
                orbits_ver = [f for f in files_ver 
                    if int(f[-9:-3]) > rough_orbit_in_year.orbit.isel(time=time_idx-1) 
                        and int(f[-9:-3]) < rough_orbit_in_year.orbit.isel(time=time_idx)]

            try:
                with xr.open_mfdataset([path_agc+f for f in orbits_agc]) as agc_ds:
                    agc_ds = agc_ds.sel(time=~agc_ds.indexes['time'].duplicated())
                    with xr.open_mfdataset([path_ver+f for f in orbits_ver]) as ver_ds:
                        ver_ds = ver_ds.reindex(time=agc_ds.time)
                        # print(ver_ds)
                    #%% filter low quality ver data
                    cond_ver = ((ver_ds.A_diag>0.8)*(
                                ver_ds.mr>0.8)*(
                                ver_ds.error2_retrieval<3e6)
                                ).rename('cond_ver')
                    cond_min_z = ver_ds.z.where(cond_ver).min(dim='z')<70e3
                    cond_max_z = ver_ds.z.where(cond_ver).max(dim='z')>90e3
                    cond_verz = (cond_min_z*cond_max_z).rename('cond_verz')
                    #%% filter unphysical agc data
                    cond_amplitude = np.logical_and(agc_ds.peak_intensity<1e5, agc_ds.peak_intensity>0).rename('cond_amplitude')
                    cond_thickness = np.logical_and(agc_ds.thickness<40e3, agc_ds.thickness>0).rename('cond_thickness')
                    cond_peak_height = np.logical_and(agc_ds.peak_height<100e3, agc_ds.peak_height>60e3).rename('cond_peak_height')
                    cond_residual = (agc_ds.residual<1e5).rename('cond_residual')
                    agc_ds = agc_ds.where(cond_min_z * cond_max_z 
                        * cond_amplitude * cond_thickness * cond_peak_height * cond_residual) 

                    mean, std, count = zonal_average(agc_ds, var=None)

                mean_year.append(mean)
                std_year.append(std)
                count_year.append(count)
                time_coord.append(rough_orbit_in_year.time[time_idx].values)
                # cond_verz_year.append(cond_verz)
                # cond_amplitude_year.append(cond_amplitude)
                # cond_peak_height_year.append(cond_peak_height)
                # cond_thickness_year.append(cond_thickness)
                # cond_residual_year.append(cond_residual)

            except OSError:
                pass
            except ValueError:
                pass
        
        print('saving agc year {}'.format(year))
        mean_year = xr.concat(mean_year, dim='time').assign_coords(time=time_coord)
        std_year = xr.concat(std_year, dim='time').assign_coords(time=time_coord)
        count_year = xr.concat(count_year, dim='time').assign_coords(time=time_coord)

        path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/'
        filename = 'gauss_clima_{}.nc'.format(year)
        mean_year.to_netcdf(path+filename, mode='w')
        std_year.to_netcdf(path+filename, mode='a')
        count_year.to_netcdf(path+filename, mode='a')

        # print('saving filterin year {}'.format(year))
        # cond_verz_year = xr.concat(cond_verz_year, dim='time')
        # cond_amplitude_year = xr.concat(cond_amplitude_year, dim='time')#.assign_coords(time=time_coord)
        # cond_peak_height_year = xr.concat(cond_peak_height_year, dim='time')#.assign_coords(time=time_coord)
        # cond_thickness_year = xr.concat(cond_thickness_year, dim='time')#.assign_coords(time=time_coord)
        # cond_residual_year = xr.concat(cond_residual_year, dim='time')#.assign_coords(time=time_coord)
    
        # path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/'
        # filename = 'filterin_{}.nc'.format(year)
        # cond_amplitude_year.to_netcdf(path+filename, mode='w')
        # cond_peak_height_year.to_netcdf(path+filename, mode='a')
        # cond_thickness_year.to_netcdf(path+filename, mode='a')
        # cond_residual_year.to_netcdf(path+filename, mode='a')
        # cond_verz_year.to_netcdf(path+filename, mode='a')

    # clima_OH(2013)
    year_lst = list(range(2001,2018))
    with Pool(processes=6) as p:
        p.map(clima_OH, year_lst)
    # for year_int in year_lst:
    #     clima_OH(year_int)
    #with Pool(processes=6) as p:
    #    p.map(clima_agc, year_lst)
    

# %%
