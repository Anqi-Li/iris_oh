
# %% make monthly files
import pandas as pd
from astropy.time import Time
import xarray as xr
import glob
from os import listdir
from multiprocessing import Pool
from dask.diagnostics import ProgressBar
import sys
import numpy as np

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

def remove_duplicate(ds):
    return ds.sel(time=~ds.indexes['time'].duplicated())
# def clip_start_end(ds, start_str, end_str):
#     return ds.sel(time=slice(start_str, end_str))

# %% 
def modify_var_attrs(ds, var, long_name=None, units=None, description=None):
        ds[var] = ds[var].assign_attrs(dict(long_name=long_name, units=units, description=description))
        return ds

def package_ver(year, month): 
    print(year, str(month).zfill(2))
    rough_orbit = find_orbit_stamps(year=year)
    range_orbit = range(*tuple(rough_orbit.isel(time=slice(month-1,month+1)).values))
    path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/archive/'
    files_ver_in_month = [f for f in glob.glob(path_ver+'*nc') if int(f[-9:-3]) in range_orbit]
    path_save = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/months/'

    if path_save+'iri_ch1_ver_{}{}.nc'.format(year, str(month).zfill(2)) in glob.glob(path_save+'*.nc'):
        print('already done')
        return
    elif path_save+'iri_ch1_ver_{}{}_.nc'.format(year, str(month).zfill(2)) in glob.glob(path_save+'*.nc'):
        print('2nd already done')
        return
    else:
        # read data
        with xr.open_mfdataset(files_ver_in_month, parallel=True, 
                preprocess=remove_duplicate, concat_dim='time') as mds_ver:
            
            mds_ver['ver'] *= 4*np.pi
            mds_ver['error2_retrieval'] *= (4*np.pi)**2
            mds_ver['error2_smoothing'] *= (4*np.pi)**2

            mds_ver = mds_ver.drop_vars(['limb_fit', 'channel', 'pixel'])
            del mds_ver.ver.attrs['long name']
            mds_ver = modify_var_attrs(mds_ver, 'z', 'altitude', 'm', 'Altitude grid of VER retrieval')
            mds_ver = modify_var_attrs(mds_ver, 'ver', 'volume_emission_rate', 'photons cm-3 s-1', 'IRI OH(3-1) volume emission rate ')
            mds_ver = modify_var_attrs(mds_ver, 'mr', 'measurement_response', '1', 'Measurement response')
            mds_ver = modify_var_attrs(mds_ver, 'A_diag', 'AVK_diagonal', '1','Averaging kernal matrix diagonal elements')
            mds_ver = modify_var_attrs(mds_ver, 'A_peak','AVK_maximum', '1', 'Averaging kernal maximum in each row')
            mds_ver = modify_var_attrs(mds_ver, 'A_peak_height', 'AVK_max_height', 'm', 'Corresponding altitude of the averaging kernal maximum in each row')
            mds_ver = modify_var_attrs(mds_ver, 'error2_retrieval', 'variance_measurement', '(photons cm-3 s-1)^2', 'Setireval noise S_m diagonal elements (Rodgers (2000))')
            mds_ver = modify_var_attrs(mds_ver, 'error2_smoothing', 'variance_a-priori', '(photons cm-3 s-1)^2', 'Smoothing error S_s diagonal elements (Rodgers (2000))')
            mds_ver = modify_var_attrs(mds_ver, 'latitude', 'latitude', 'degrees north', 'Latitude at the tangent point')
            mds_ver = modify_var_attrs(mds_ver, 'longitude', 'longitude', 'degrees east', 'Longitude at the tangent point')
            mds_ver = mds_ver.assign_attrs(dict(channel=1))
            
            # save data
            encoding = dict(
                orbit=dict(dtype='int32'), 
                time=dict(units='days since 1858-11-17'),
                )
            encoding.update({v: {'zlib': True, 'dtype': 'float32'} for v in mds_ver.data_vars 
                            if mds_ver[v].dtype == 'float64'})

            delayed_obj = mds_ver.to_netcdf(path_save+'iri_ch1_ver_{}{}_.nc'.format(year, str(month).zfill(2)),
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
        with xr.open_mfdataset(files_gauss_in_month, parallel=True, concat_dim='time') as mds_gauss:
            mds_gauss = mds_gauss.drop_vars('channel')
            mds_gauss = modify_var_attrs(mds_gauss, 'latitude', 'latitude', 'degrees north', 'Latitude at the tangent point')
            mds_gauss = modify_var_attrs(mds_gauss, 'longitude', 'longitude', 'degrees east', 'Longitude at the tangent point')
            mds_gauss = modify_var_attrs(mds_gauss, 'apparent_solar_time', 'apparent_solar_time', 'hour', 'Apparent Solar Time at the line-of-sight tangent point')
            mds_gauss = modify_var_attrs(mds_gauss, 'sza', 'solar_zenith_angle', 'degrees', 'Solar Zenith Angle between the satellite line-of-sight and the sun')
            mds_gauss = modify_var_attrs(mds_gauss, 'chisq', 'chi_square', '1', 'chi square of the Gaussian function fitting')
            mds_gauss['cov_peak_intensity_peak_height'] = mds_gauss['cov_peak_intensity_peak_height'].assign_attrs(dict(description='error covariance of peak_intensity and peak_height'))
            mds_gauss['cov_peak_intensity_peak_sigma'] = mds_gauss['cov_peak_intensity_peak_sigma'].assign_attrs(dict(description='error covariance of peak_intensity and peak_sigma'))
            mds_gauss['cov_peak_height_peak_sigma'] = mds_gauss['cov_peak_height_peak_sigma'].assign_attrs(dict(description='error covariance of peak_height and peak_sigma'))
            mds_gauss = mds_gauss.assign_attrs(dict(channel=1))

            # save data
            encoding = dict(
                orbit=dict(dtype='int32'), 
                time=dict(units='days since 1858-11-17'),

                )
            encoding.update({v: {'zlib': True, 'dtype': 'float32'} for v in mds_gauss.data_vars 
                            if mds_gauss[v].dtype == 'float64'})

            delayed_obj = mds_gauss.to_netcdf(path_save+'gauss_{}{}.nc'.format(year, str(month).zfill(2)),
                        encoding=encoding, compute=False)
        with ProgressBar():
            delayed_obj.compute()



#%%
# year = int(sys.argv[1]) #2002
# months = range(1,13) #[1, 2, 4, 5,6,7,8,9,10,11,12]
# # years = range(2009,2018)
# # for year in years:
# for month in months:
#         try:
#             package_ver(year, month)
#         except OSError:
#             pass
#         except:
#             # raise
#             pass


# %%
# year=2009
# month=7

# rough_orbit = find_orbit_stamps(year=year)
# range_orbit = range(*tuple(rough_orbit.isel(time=slice(month-1,month+1)).values))
# path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/archive/'
# files_ver_in_month = [f for f in glob.glob(path_ver+'*nc') if int(f[-9:-3]) in range_orbit]
# path_save = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/months/'

# xxx = []
# orbits = []
# for f in sorted(files_ver_in_month):
#     with xr.open_dataset(f) as ds:
#         xxx.append(ds.indexes['time'].duplicated().any())
#         orbits.append(ds.orbit.values)

# orbits[np.where(xxx)[0][0]]
# # %%
# with xr.open_dataset(path_ver+'iri_ch1_ver_{}.nc'.format(str(45665).zfill(6))) as ds:
#     print(ds)
# %%
def package_year(year):
    # year = 2001
    path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/months/'
    filename_ver_months = 'iri_ch1_ver_{year}{month}.nc'.format(year=year, month='*')
    path_gauss = '/home/anqil/Documents/sshfs/oso_extra_storage/Gaussian_OH_night/months/'
    filename_gauss_months = 'gauss_{year}{month}.nc'.format(year=year, month='*')

    path_save = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/years/'
    with xr.open_mfdataset(path_ver+filename_ver_months) as mds_ver:
        mds_ver = modify_var_attrs(mds_ver, 'z', 'altitude', 'm', 'Altitude grid of VER retrieval')
        mds_ver = modify_var_attrs(mds_ver, 'ver', 'volume_emission_rate', 'photons cm-3 s-1', 'IRI OH(3-1) volume emission rate ')
        mds_ver = modify_var_attrs(mds_ver, 'mr', 'measurement_response', '1', 'Measurement response')
        mds_ver = modify_var_attrs(mds_ver, 'A_diag', 'AVK_diagonal', '1','Averaging kernel matrix diagonal elements')
        mds_ver = modify_var_attrs(mds_ver, 'A_peak','AVK_maximum', '1', 'Averaging kernel maximum in each row')
        mds_ver = modify_var_attrs(mds_ver, 'A_peak_height', 'AVK_max_height', 'm', 'Corresponding altitude of the averaging kernel maximum in each row')
        mds_ver = modify_var_attrs(mds_ver, 'error2_retrieval', 'variance_measurement', '(photons cm-3 s-1)^2', 'Retrieval noise S_m diagonal elements (Rodgers (2000))')
        mds_ver = modify_var_attrs(mds_ver, 'error2_smoothing', 'variance_a-priori', '(photons cm-3 s-1)^2', 'Smoothing error S_s diagonal elements (Rodgers (2000))')
        mds_ver = modify_var_attrs(mds_ver, 'latitude', 'latitude', 'degrees north', 'Latitude at the tangent point')
        mds_ver = modify_var_attrs(mds_ver, 'longitude', 'longitude', 'degrees east', 'Longitude at the tangent point')
        mds_ver = mds_ver.assign_attrs(dict(channel=1))

    with xr.open_mfdataset(path_gauss+filename_gauss_months) as mds_gauss:
        mds_gauss = modify_var_attrs(mds_gauss, 'chisq', 'chi_square', '1', 'Cost of the Gaussian function fitting')
        mds_gauss = modify_var_attrs(mds_gauss, 'cov_peak_intensity_peak_height', 'covariance_intensity_height', 'photons cm-3 s-1 m', 'Error covariance of peak_intensity and peak_height')
        mds_gauss = modify_var_attrs(mds_gauss, 'cov_peak_intensity_peak_sigma', 'covariance_intensity_sigma', 'photons cm-3 s-1 m', 'Error covariance of peak_intensity and peak_sigma')
        mds_gauss = modify_var_attrs(mds_gauss, 'cov_peak_height_peak_sigma', 'covariance_height_sigma', 'm m', 'Error covariance of peak_height and peak_sigma')
        mds_gauss = modify_var_attrs(mds_gauss, 'zenith_intensity', 'gauss_zenith_intensity', 'photons cm-2 s-1', 'Integration of the Gaussian function')
        mds_gauss = modify_var_attrs(mds_gauss, 'zenith_intensity_error', 'gauss_zenith_intensity_error', 'photons cm-2 s-1', 'Error of Gaussian integration')
        mds_gauss = modify_var_attrs(mds_gauss, 'peak_intensity', 'gauss_peak_intensity', 'photons cm-3 s-1', 'Peak intensity (Gaussian fit)')
        mds_gauss = modify_var_attrs(mds_gauss, 'peak_intensity_error', 'gauss_peak_intensity_error', 'photons cm-3 s-1', 'Error of peak intensity (Gaussian fit)')
        mds_gauss = modify_var_attrs(mds_gauss, 'peak_height', 'gauss_peak_height', 'm', 'Peak height (Gaussian fit)')
        mds_gauss = modify_var_attrs(mds_gauss, 'peak_height_error', 'gauss_peak_height_error', 'm', 'Error of peak height (Gaussian fit)')
        mds_gauss = modify_var_attrs(mds_gauss, 'peak_sigma', 'gauss_peak_sigma', 'm', 'Half layer thickness (Gaussian fit)')
        mds_gauss = modify_var_attrs(mds_gauss, 'peak_sigma_error', 'gauss_peak_sigma_error', 'm', 'Error of half layer thickness (Gaussian fit)')

    mds = xr.merge([mds_ver, mds_gauss],
                    compat='override', join='left', combine_attrs='override')

    # save data
    encoding = dict(
        z=dict(dtype='float32', zlib=True),
        )

    delayed_obj = mds.to_netcdf(path_save+'iri_ch1_ver_{}.nc'.format(year),
                encoding=encoding, compute=False)

    with ProgressBar():
        delayed_obj.compute()

for year in range(2002,2018):
    print(year)
    path_save = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/years/'
    filename = 'iri_ch1_ver_{}.nc'.format(year)
    if path_save+filename in glob.glob(path_save+'*.nc'):
        pass
    else:
        package_year(year)

# %%
