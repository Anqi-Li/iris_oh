#%%
import numpy as np
import xarray as xr
import glob 


#%%
def groupby_time_lat(ds, var=None, dlat=20):
    print(var)
    latitude_bins = np.arange(-90,90+dlat,dlat)
    latitude_labels = latitude_bins[1:]-dlat/2

    if var==None:
        groups = ds.groupby(ds.time.dt.month)
    else:
        groups = ds[var].groupby(ds.time.dt.month)

    time_bins, mean, count, std = [], [], [], []
    for s, data in groups:
        print('month = {}'.format(s))
        time_bins.append(s)
        mean.append(data.groupby_bins(data.latitude, 
            bins=latitude_bins, labels=latitude_labels).mean(..., keep_attrs=True))
        count.append(data.groupby_bins(data.latitude, 
            bins=latitude_bins, labels=latitude_labels).count(...))
        std.append(data.groupby_bins(data.latitude, 
            bins=latitude_bins, labels=latitude_labels).std(..., keep_attrs=True))

    mean = xr.concat(mean, dim='time_bins').assign_coords(time_bins=time_bins)
    count = xr.concat(count, dim='time_bins').assign_coords(time_bins=time_bins)
    std = xr.concat(std, dim='time_bins').assign_coords(time_bins=time_bins)
    
    if var==None:
        mean = mean.rename_vars({k: 'mean_{}'.format(k) for k in mean.data_vars.keys()})
        std = std.rename_vars({k: 'std_{}'.format(k) for k in std.data_vars.keys()})
        count = count.rename_vars({k: 'count_{}'.format(k) for k in count.data_vars.keys()})
    else:
        mean = mean.rename('mean_{}'.format(var))
        count = count.rename('count_{}'.format(var))
        std = std.rename('std_{}'.format(var))

    return mean, count, std


def groupby_time_lat_lon(ds, var=None, dlat=20, dlon=20):
    print(var)
    latitude_bins = np.arange(-90,90+dlat,dlat)
    latitude_labels = latitude_bins[1:]-dlat/2

    longitude_bins = np.arange(0,360+dlon, dlon)
    longitude_labels = longitude_bins[1:] - dlon/2

    if var==None:
        groups = ds.groupby(ds.time.dt.season)
    else:
        groups = ds[var].groupby(ds.time.dt.season)
    time_bins, mean, count, std = [], [], [], []
    for t, t_data in groups:
        print('season = {}'.format(t))
        time_bins.append(t)

        longitude_lab, lat_lon_mean, lat_lon_count, lat_lon_std = [], [], [], []
        for lon, lon_t_data in t_data.groupby_bins(t_data.longitude, bins=longitude_bins, labels=longitude_labels):
            longitude_lab.append(lon)
            print('longitude = {}'.format(lon))
            lat_lon_mean.append(lon_t_data.groupby_bins(lon_t_data.latitude,
             bins=latitude_bins, labels=latitude_labels).mean(..., keep_attrs=True))
            lat_lon_count.append(lon_t_data.groupby_bins(lon_t_data.latitude, 
            bins=latitude_bins, labels=latitude_labels).count(...))
            lat_lon_std.append(lon_t_data.groupby_bins(lon_t_data.latitude, 
            bins=latitude_bins, labels=latitude_labels).std(..., keep_attrs=True))

        mean.append(xr.concat(lat_lon_mean, dim='longitude_bins').assign_coords(longitude_bins=longitude_lab).sortby('longitude_bins'))
        count.append(xr.concat(lat_lon_count, dim='longitude_bins').assign_coords(longitude_bins=longitude_lab).sortby('longitude_bins'))
        std.append(xr.concat(lat_lon_std, dim='longitude_bins').assign_coords(longitude_bins=longitude_lab).sortby('longitude_bins'))

    mean = xr.concat(mean, dim='time_bins').assign_coords(time_bins=time_bins)
    count = xr.concat(count, dim='time_bins').assign_coords(time_bins=time_bins)
    std = xr.concat(std, dim='time_bins').assign_coords(time_bins=time_bins)

    if var==None:
        mean = mean.rename_vars({k: 'mean_{}'.format(k) for k in mean.data_vars.keys()})
        std = std.rename_vars({k: 'std_{}'.format(k) for k in std.data_vars.keys()})
        count = count.rename_vars({k: 'count_{}'.format(k) for k in count.data_vars.keys()})
    else:
        mean = mean.rename('mean_{}'.format(var))
        count = count.rename('count_{}'.format(var))
        std = std.rename('std_{}'.format(var))

    return mean, count, std

#%%
if __name__ == '__main__':
    
    # %% rough estimates of odin year-orbits
    orbit_year = xr.open_dataset('/home/anqil/Documents/osiris_database/odin_rough_orbit_year.nc')
    orbit_year.close()
    
    #%% character files
    path = '/home/anqil/Documents/osiris_database/iris_oh/'
    sp_filelist = glob.glob(path+'spectral_character/archive/sp_*.nc')
    agc_filelist = glob.glob(path + 'airglow_character/archive/agc_*.nc')

    #%% select relevant data for analysis (year? dayofyear?)
    for year in range(2001, 2019):
        print('load year {} '.format(year))
        sp_filelist_year = [f for f in sp_filelist if int(f[-9:-3])<orbit_year.sel(year=year+1).orbit.item() and int(f[-9:-3])>orbit_year.sel(year=year).orbit.item()]
        sp_ds = xr.open_mfdataset(sp_filelist_year).set_coords(['longitude', 'latitude'])
        agc_filelist_year = [f for f in agc_filelist if int(f[-9:-3])<orbit_year.sel(year=year+1).orbit.item() and int(f[-9:-3])>orbit_year.sel(year=year).orbit.item()]
        agc_ds = xr.open_mfdataset(agc_filelist_year).set_coords(['longitude', 'latitude'])

        #%% time_lat
        print('time_lat')
        mean_agc, count_agc, std_agc = groupby_time_lat(agc_ds, var=None)
        mean_max_pw_freq, _, std_max_pw_freq= groupby_time_lat(sp_ds, 'max_pw_freq')
        mean_max_pw, sp_count, std_max_pw = groupby_time_lat(sp_ds, 'max_pw')
        print('Save files')
        filename = 'time_lat_{}.nc'.format(year)
        if len([f for f in glob.glob(path+'statistics/*.nc') if filename in f]) == 0:
            mean_max_pw_freq.to_netcdf(path+'statistics/'+filename, mode='w')
        else:
            mean_max_pw_freq.to_netcdf(path+'statistics/'+filename, mode='a')
        mean_max_pw.to_netcdf(path+'statistics/'+filename, mode='a')
        mean_agc.to_netcdf(path+'statistics/'+filename, mode='a')
        std_max_pw_freq.to_netcdf(path+'statistics/'+filename, mode='a')
        std_max_pw.to_netcdf(path+'statistics/'+filename, mode='a')
        std_agc.to_netcdf(path+'statistics/'+filename, mode='a')
        count_agc.to_netcdf(path+'statistics/'+filename, mode='a')
        sp_count.rename('sp_sample_count').to_netcdf(path+'statistics/'+filename, mode='a')

        #%% time_lat_lon
        print('time_lat_lon')
        mean_agc, count_agc, std_agc = groupby_time_lat_lon(agc_ds, var=None)
        mean_max_pw_freq, _, std_max_pw_freq = groupby_time_lat_lon(sp_ds, 'max_pw_freq')
        mean_max_pw, sp_count, std_max_pw = groupby_time_lat_lon(sp_ds, 'max_pw')
        print('Save files')
        filename = 'time_lat_lon_{}.nc'.format(year)
        if len([f for f in glob.glob(path+'statistics/*.nc') if filename in f]) == 0:
            mean_max_pw_freq.to_netcdf(path+'statistics/'+filename, mode='w')
        else:
            mean_max_pw_freq.to_netcdf(path+'statistics/'+filename, mode='a')
        mean_max_pw.to_netcdf(path+'statistics/'+filename, mode='a')
        mean_agc.to_netcdf(path+'statistics/'+filename, mode='a')
        std_max_pw_freq.to_netcdf(path+'statistics/'+filename, mode='a')
        std_max_pw.to_netcdf(path+'statistics/'+filename, mode='a')
        std_agc.to_netcdf(path+'statistics/'+filename, mode='a')
        count_agc.to_netcdf(path+'statistics/'+filename, mode='a')
        sp_count.rename('sp_sample_count').to_netcdf(path+'statistics/'+filename, mode='a')

#%%