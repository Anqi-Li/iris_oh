#%%
import numpy as np
import xarray as xr
from multiprocessing import Pool
import pandas as pd
from dask.diagnostics import ProgressBar

#%%
# year = 2001
# am_pm = 'PM'
def average_year(year, am_pm):
    print(year)
    ver_path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/years/false_zenith/'
    ver_filename = 'iri_ch1_ver_{}.nc'
    if year == 2001:
        file_lst = [ver_path+ver_filename.format(y) for y in [2001, 2002]]
    elif year == 2017:
        file_lst = [ver_path+ver_filename.format(y) for y in [2016, 2017]]
    else:
        file_lst = [ver_path+ver_filename.format(y) for y in range(year-1, year+2)]

    with xr.open_mfdataset(file_lst).sel(time=str(year)) as ds:
        if am_pm == 'PM':
            cond_lst = ds.apparent_solar_time>12 
        elif am_pm == 'AM':
            cond_lst = ds.apparent_solar_time<=12
        elif am_pm == 'ALL':
            cond_lst = ds.apparent_solar_time>0

        cond_sza = (ds.sza>96).rename('cond_sza')
        cond_intensity = np.logical_and(ds.peak_intensity<1e6*(4*np.pi), ds.peak_intensity>0).rename('cond_intensity')
        cond_sigma = np.logical_and(ds.peak_sigma<20e3, ds.peak_sigma>0).rename('cond_sigma')
        cond_height = np.logical_and(ds.peak_height<100e3, ds.peak_height>60e3).rename('cond_height')
        cond_chisq = (ds.chisq<10).rename('cond_chisq')
        vars = ['peak_intensity', 'peak_height', 'peak_sigma', 
                'latitude', 'longitude', 'sza', 'apparent_solar_time']
        ds_agc = ds[vars]
        ds_agc = ds_agc.where(cond_intensity * cond_sigma * cond_height * cond_chisq * cond_sza * cond_lst) 

        cond_mr = (ds.A_diag>0.8)*(ds.A_peak>0.8)*(ds.mr>0.8)
        cond_error2 = (ds.error2_retrieval<3e6 * (4*np.pi)**2)
        ds_ver = ds[['ver']].where(cond_mr * cond_error2)

        ds_year = xr.merge([ds_agc, ds_ver])

        #zonal - daily mean
        dlat = 20
        latitude_bins = np.arange(-90, 90+dlat, dlat)
        latitude_labels = latitude_bins[1:]-dlat/2
        groups_lat = ds_year.groupby_bins(
            ds_year.latitude, bins=latitude_bins, 
            labels=latitude_labels)
        lat_coord = []
        mean_daily, std_daily, count_daily = [], [], []
        for label, data in groups_lat:
            mean_daily.append(data.resample(time='D').mean('time', keep_attrs=True))
            std_daily.append(data.resample(time='D').std('time', keep_attrs=True))
            count_daily.append(data.resample(time='D').count('time', keep_attrs=True))
            lat_coord.append(label)
            print(label)
        mean_daily = xr.concat(mean_daily, dim='latitude_bin').assign_coords(latitude_bin=lat_coord).sortby('latitude_bin')
        std_daily = xr.concat(std_daily, dim='latitude_bin').assign_coords(latitude_bin=lat_coord).sortby('latitude_bin')
        count_daily = xr.concat(count_daily, dim='latitude_bin').assign_coords(latitude_bin=lat_coord).sortby('latitude_bin')
        final =  xr.merge(
            [mean_daily.rename({k: 'mean_{}'.format(k) for k in mean_daily.keys()}), 
            std_daily.rename({k: 'std_{}'.format(k) for k in std_daily.keys()}), 
            count_daily.rename({k: 'count_{}'.format(k) for k in count_daily.keys()})]
            ).assign_attrs(am_pm=am_pm)

        return final


#%%
# year = 2008
am_pm = 'ALL'
for year in range(2001, 2008):
    ds = average_year(year, am_pm)
    print('saving VER year {}'.format(year))
    path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/averages/'
    filename = '{}_daily_zonal_mean_{}.nc'.format(am_pm, year)
    delayed_obj = ds.to_netcdf(path+filename, mode='w', compute=False)
    with ProgressBar():
        delayed_obj.compute()

#%% test
# path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/averages/'
# filename = 'PM_daily_zonal_mean_{}.nc'.format(2001)
# with xr.open_dataset(path+filename) as ds:
#     # print(ds)
#     # ds.mean_peak_intensity.plot.line(x='time', hue='latitude_bin')
#     ds.mean_ver.plot(x='time', y='z', row='latitude_bin')

# %%
