#%%
import numpy as np
import xarray as xr
import glob 
from multiprocessing import Pool

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

#%% check histogram distributions
# import seaborn as sns
# import matplotlib.pyplot as plt
# # rough estimates of odin year-orbits
# orbit_year = xr.open_dataset('/home/anqil/Documents/osiris_database/odin_rough_orbit_year.nc')
# orbit_year.close()

# # character files
# path = '/home/anqil/Documents/osiris_database/iris_oh/'
# agc_filelist = glob.glob(path + 'airglow_character/agc_*.nc')

# hist_peak_height, hist_thickness, hist_amplitude, hist_residual = [], [], [], []
# for year in range(2003, 2018):
#     print(year)
#     agc_filelist_year = [f for f in agc_filelist 
#         if int(f[-9:-3])>orbit_year.sel(year=year).orbit.item()
#         and int(f[-9:-3])<orbit_year.sel(year=year+1).orbit.item()]
#     with xr.open_mfdataset(agc_filelist_year) as agc_ds:
#         agc_ds = agc_ds.set_coords(['longitude', 'latitude']).drop(('orbit', 'channel'))
#         agc_ds['thickness'] = abs(agc_ds.thickness)

#         hist_amplitude.append(agc_ds.amplitude.pipe(np.histogram, bins=[0,1e5, 2e10]))
#         hist_thickness.append(agc_ds.thickness.pipe(np.histogram, bins=[0, 40e3, 1e6]))
#         hist_peak_height.append(agc_ds.peak_height.pipe(np.histogram, bins=[-2e6, 0, 60e3, 100e3, 2e6]))
#         hist_residual.append(agc_ds.residual.pipe(np.histogram, bins=[0, 1e4, 1e5, 1e6, 1e7]))

# #%% plots
# title_lst = 'amplitude thickness peak_height residual'.split()
# fig, ax = plt.subplots(1,4, figsize=(10,4))
# for i, hist in enumerate([hist_amplitude, hist_thickness, hist_peak_height, hist_residual]):
#     year = 2003
#     for density, bins in hist:
#         ax[i].plot(bins[1:], density, label=year)#, where='pre') #align='edge')
#         year+=1
#     ax[i].set(title=title_lst[i], yscale='log', xscale='log')
# plt.legend()
# ax[0].set(xscale='log')
# ax[-1].set(xscale='log')
# # plt.show()

# #%% KDE plot
# par_lst = 'amplitude thickness peak_height residual'.split()
# fig, ax = plt.subplots(1,4, figsize=(10,4))
# kde = []
# for year in range(2002, 2018):
#     print(year)
#     agc_filelist_year = [f for f in agc_filelist 
#         if int(f[-9:-3])>orbit_year.sel(year=year).orbit.item()
#         and int(f[-9:-3])<orbit_year.sel(year=year+1).orbit.item()]
#     with xr.open_mfdataset(agc_filelist_year) as agc_ds:
#         agc_ds = agc_ds.set_coords(['longitude', 'latitude']).drop(('orbit', 'channel'))
#         agc_ds['thickness'] = abs(agc_ds.thickness)
#         for i, par in enumerate(par_lst):
#             kde.append(agc_ds[par].pipe(sns.kdeplot, ax=ax[i], label=year))

# plt.legend()

#%%
if __name__ == '__main__':

    # %% rough estimates of odin year-orbits
    orbit_year = xr.open_dataset('/home/anqil/Documents/osiris_database/odin_rough_orbit_year.nc')
    orbit_year.close()
    
    #%% character files
    path = '/home/anqil/Documents/osiris_database/iris_oh/'
    # sp_filelist = glob.glob(path+'spectral_character/archive/sp_*.nc')
    agc_filelist = glob.glob(path + 'airglow_character/agc_*.nc')

    #%% select relevant data for analysis (year? dayofyear?)
    # for year in range(2001, 2018):
    def statistics(year):
        print('load year {} '.format(year))
        # sp_filelist_year = [f for f in sp_filelist if int(f[-9:-3])<orbit_year.sel(year=year+1).orbit.item() and int(f[-9:-3])>orbit_year.sel(year=year).orbit.item()]
        # sp_ds = xr.open_mfdataset(sp_filelist_year).set_coords(['longitude', 'latitude'])
        agc_filelist_year = [f for f in agc_filelist 
            if int(f[-9:-3])>orbit_year.sel(year=year).orbit.item()
                 and int(f[-9:-3])<orbit_year.sel(year=year+1).orbit.item()]
        with xr.open_mfdataset(agc_filelist_year) as agc_ds:
            agc_ds = agc_ds.set_coords(['longitude', 'latitude']).drop(('orbit', 'channel'))
            agc_ds['thickness'] = abs(agc_ds.thickness)

            #%% filter unphysical data
            cond_amplitude = np.logical_and(agc_ds.amplitude<1e5, agc_ds.amplitude>0)
            cond_thickness = np.logical_and(agc_ds.thickness<40e3, agc_ds.thickness>0)
            cond_peak_height = np.logical_and(agc_ds.peak_height<100e3, agc_ds.peak_height>60e3)
            cond_residual = (agc_ds.residual<1e5)
            agc_ds = agc_ds.where(cond_amplitude * cond_thickness * cond_peak_height * cond_residual) 

            cond_amplitude.rename('cond_amplitude').to_netcdf(path+'statistics/filterin_{}.nc'.format(year), mode='w')
            cond_thickness.rename('cond_thickness').to_netcdf(path+'statistics/filterin_{}.nc'.format(year), mode='a')
            cond_peak_height.rename('cond_peak_height').to_netcdf(path+'statistics/filterin_{}.nc'.format(year), mode='a')
            cond_residual.rename('cond_residual').to_netcdf(path+'statistics/filterin_{}.nc'.format(year), mode='a')

        #%% time_lat
        print('time_lat')
        mean_agc, count_agc, std_agc = groupby_time_lat(agc_ds, var=None)
        # mean_max_pw_freq, _, std_max_pw_freq= groupby_time_lat(sp_ds, 'max_pw_freq')
        # mean_max_pw, sp_count, std_max_pw = groupby_time_lat(sp_ds, 'max_pw')
        
        filename = 'time_lat_{}.nc'.format(year)
        print('Save file {}'.format(filename))
        # if len([f for f in glob.glob(path+'statistics/*.nc') if filename in f]) == 0:
        mean_agc.to_netcdf(path+'statistics/'+filename, mode='w')
        # else:
            # mean_agc.to_netcdf(path+'statistics/'+filename, mode='a')
        std_agc.to_netcdf(path+'statistics/'+filename, mode='a')
        count_agc.to_netcdf(path+'statistics/'+filename, mode='a')

        # mean_max_pw_freq.to_netcdf(path+'statistics/'+filename, mode='a')
        # mean_max_pw.to_netcdf(path+'statistics/'+filename, mode='a')
        # std_max_pw_freq.to_netcdf(path+'statistics/'+filename, mode='a')
        # std_max_pw.to_netcdf(path+'statistics/'+filename, mode='a')
        # sp_count.rename('sp_sample_count').to_netcdf(path+'statistics/'+filename, mode='a')

        #%% time_lat_lon
        # print('time_lat_lon')
        # mean_agc, count_agc, std_agc = groupby_time_lat_lon(agc_ds, var=None)
        # # mean_max_pw_freq, _, std_max_pw_freq = groupby_time_lat_lon(sp_ds, 'max_pw_freq')
        # # mean_max_pw, sp_count, std_max_pw = groupby_time_lat_lon(sp_ds, 'max_pw')
        
        # filename = 'time_lat_lon_{}.nc'.format(year)
        # print('Save file {}'.format(filename))
        # # if len([f for f in glob.glob(path+'statistics/*.nc') if filename in f]) == 0:
        # mean_agc.to_netcdf(path+'statistics/'+filename, mode='w')
        # # else:
        #     # mean_agc.to_netcdf(path+'statistics/'+filename, mode='a')
        # std_agc.to_netcdf(path+'statistics/'+filename, mode='a')
        # count_agc.to_netcdf(path+'statistics/'+filename, mode='a')

        # # mean_max_pw_freq.to_netcdf(path+'statistics/'+filename, mode='a')
        # # mean_max_pw.to_netcdf(path+'statistics/'+filename, mode='a')
        # # std_max_pw_freq.to_netcdf(path+'statistics/'+filename, mode='a')
        # # std_max_pw.to_netcdf(path+'statistics/'+filename, mode='a')
        # # sp_count.rename('sp_sample_count').to_netcdf(path+'statistics/'+filename, mode='a')
    with Pool(processes=8) as p:
        p.map(statistics, range(2009,2017))
#%%