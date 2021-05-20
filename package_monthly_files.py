
# %% make monthly files
import pandas as pd
from astropy.time import Time
import xarray as xr
import glob
from os import listdir
from multiprocessing import Pool
from dask.diagnostics import ProgressBar
import sys

#%%
def find_orbit_stamps(year):
    time_stamp = pd.date_range(start='{}-12-31 23:59:59'.format(year-1), 
                                end='{}-01-31 23:59:59'.format(year+1),
                                freq='M')
    with xr.open_dataset('~/Documents/osiris_database/odin_rough_orbit.nc') as rough_orbit:
        rough_orbit = rough_orbit.rename({'mjd':'time'}).assign(
            time=Time(rough_orbit.mjd, format='mjd').datetime64
            ).interp(time=time_stamp).astype(int).orbit
    return rough_orbit

# def remove_duplicate(ds):
#     ds.sel(time=~ds.indexes['time'].duplicated())
#     return ds
# def clip_start_end(ds, start_str, end_str):
#     return ds.sel(time=slice(start_str, end_str))

# %% test 2
def modify_var_attrs(ds, var, long_name=None, units=None, description=None):
        ds[var] = ds[var].assign_attrs(dict(long_name=long_name, units=units, description=description))
        return ds

def package_ver(year, month): 
    print(year, str(month).zfill(2))
    rough_orbit = find_orbit_stamps(year=year)
    range_orbit = range(*tuple(rough_orbit.isel(time=slice(month-1,month+1)).values))
    path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/4pi/'
    files_ver_in_month = [f for f in glob.glob(path_ver+'*nc') if int(f[-9:-3]) in range_orbit]
    path_save = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/months/'

    if path_save+'iri_ch1_ver_{}{}.nc'.format(year, str(month).zfill(2)) in glob.glob(path_save+'*.nc'):
        print('already done')
        return
    else:
        # read data
        with xr.open_mfdataset(files_ver_in_month, parallel=True, concat_dim='time') as mds:
            mds = mds.drop_vars(['limb_fit', 'channel', 'pixel'])
            del mds.ver.attrs['long name']
            mds = modify_var_attrs(mds, 'z', 'altitude', 'm', 'Altitude grid of VER retrieval')
            mds = modify_var_attrs(mds, 'ver', 'volume_emission_rate', 'photons cm-3 s-1', 'IRI OH(3-1) volume emission rate ')
            mds = modify_var_attrs(mds, 'mr', 'measurement_response', '1', 'Measurement response')
            mds = modify_var_attrs(mds, 'A_diag', 'AVK_diagonal', '1','Averaging kernal matrix diagonal elements')
            mds = modify_var_attrs(mds, 'A_peak','AVK_maximum', '1', 'Averaging kernal maximum in each row')
            mds = modify_var_attrs(mds, 'A_peak_height', 'AVK_max_height', 'm', 'Corresponding altitude of the averaging kernal maximum in each row')
            mds = modify_var_attrs(mds, 'error2_retrieval', 'variance_measurement', '(photons cm-3 s-1)^2', 'Setireval noise S_m diagonal elements (Rodgers (2000))')
            mds = modify_var_attrs(mds, 'error2_smoothing', 'variance_a-priori', '(photons cm-3 s-1)^2', 'Smoothing error S_s diagonal elements (Rodgers (2000))')
            mds = modify_var_attrs(mds, 'latitude', 'latitude', 'degrees north', 'Latitude at the tangent point')
            mds = modify_var_attrs(mds, 'longitude', 'longitude', 'degrees east', 'Longitude at the tangent point')
            mds = mds.assign_attrs(dict(channel=1))
            
            # save data
            encoding = dict(
                orbit=dict(dtype='int32'), 
                time=dict(units='days since 1858-11-17'),
                )
            encoding.update({v: {'zlib': True, 'dtype': 'float32'} for v in mds.data_vars 
                            if mds[v].dtype == 'float64'})

            delayed_obj = mds.to_netcdf(path_save+'iri_ch1_ver_{}{}.nc'.format(year, str(month).zfill(2)),
                        encoding=encoding, compute=False)
        with ProgressBar():
            delayed_obj.compute()
    
#%%
def package_gauss(year, month):
# year = 2001
# month = 12
    print(year, str(month).zfill(2))
    rough_orbit = find_orbit_stamps(year=year)
    range_orbit = range(*tuple(rough_orbit.isel(time=slice(month-1,month+1)).values))
    path_gauss = '/home/anqil/Documents/sshfs/oso_extra_storage/Gaussian_OH_night/orbits/'
    files_gauss_in_month = [f for f in glob.glob(path_gauss+'*nc') if int(f[-9:-3]) in range_orbit]
    path_save = '/home/anqil/Documents/sshfs/oso_extra_storage/Gaussian_OH_night/months/'
    if path_save+'gauss_{}{}.nc'.format(year, str(month).zfill(2)) in glob.glob(path_save+'*.nc'):
        print('already done')
    else:
        # read data
        with xr.open_mfdataset(files_gauss_in_month, parallel=True, concat_dim='time') as mds:
            mds = mds.drop_vars('channel')
            mds = modify_var_attrs(mds, 'latitude', 'latitude', 'degrees north', 'Latitude at the tangent point')
            mds = modify_var_attrs(mds, 'longitude', 'longitude', 'degrees east', 'Longitude at the tangent point')
            mds = modify_var_attrs(mds, 'apparent_solar_time', 'apparent_solar_time', 'hour', 'Apparent Solar Time at the line-of-sight tangent point')
            mds = modify_var_attrs(mds, 'sza', 'solar_zenith_angle', 'degrees', 'Solar Zenith Angle between the satellite line-of-sight and the sun')
            mds = modify_var_attrs(mds, 'chisq', 'chi_square', '1', 'chi square of the Gaussian function fitting')
            mds['cov_peak_intensity_peak_height'] = mds['cov_peak_intensity_peak_height'].assign_attrs(dict(description='error covariance of peak_intensity and peak_height'))
            mds['cov_peak_intensity_peak_sigma'] = mds['cov_peak_intensity_peak_sigma'].assign_attrs(dict(description='error covariance of peak_intensity and peak_sigma'))
            mds['cov_peak_height_peak_sigma'] = mds['cov_peak_height_peak_sigma'].assign_attrs(dict(description='error covariance of peak_height and peak_sigma'))
            mds = mds.assign_attrs(dict(channel=1))

            # save data
            encoding = dict(
                orbit=dict(dtype='int32'), 
                time=dict(units='days since 1858-11-17'),

                )
            encoding.update({v: {'zlib': True, 'dtype': 'float32'} for v in mds.data_vars 
                            if mds[v].dtype == 'float64'})

            delayed_obj = mds.to_netcdf(path_save+'gauss_{}{}.nc'.format(year, str(month).zfill(2)),
                        encoding=encoding, compute=False)
        with ProgressBar():
            delayed_obj.compute()



#%%
# year = int(sys.argv[1]) #2002
months = range(1,13) #[1, 2, 4, 5,6,7,8,9,10,11,12]
years = range(2003,2018)
for year in years:
    for month in months:
        try:
            package_gauss(year, month)
        except OSError:
            pass
        except:
            raise


# %%
