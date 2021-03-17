#%%
import numpy as np
import xarray as xr
from characterise_agc_routine import characterise_layer
from multiprocessing import Pool

#%% fit gauss to monthly zonal mean profiles
def fit_clima(mds):
    final = []
    for i in range(len(mds.latitude_bins)):
        result=[]
        for j in range(len(mds.time)):
            # print(i, j)
            ver_z = mds.mean_ver.isel(latitude_bins=i, time=j)
            if all(ver_z.isnull()):
                result.append(np.nan * np.ones(7))
            else:
                try:
                    result.append(characterise_layer(ver_z))
                except:
                    result.append(np.nan * np.ones(7))
        final.append(np.array(result))
    final = np.array(final)

    mean_amplitude = xr.DataArray(final[:,:,0], dims=('latitude', 'time'), coords=(mds.latitude_bins, mds.time))
    mean_peak_height = xr.DataArray(final[:,:,1], dims=('latitude', 'time'), coords=(mds.latitude_bins, mds.time))
    mean_thickness = xr.DataArray(final[:,:,2], dims=('latitude', 'time'), coords=(mds.latitude_bins, mds.time))
    std_amplitude = xr.DataArray(final[:,:,3], dims=('latitude', 'time'), coords=(mds.latitude_bins, mds.time))
    std_peak_height = xr.DataArray(final[:,:,4], dims=('latitude', 'time'), coords=(mds.latitude_bins, mds.time))
    std_thickness = xr.DataArray(final[:,:,5], dims=('latitude', 'time'), coords=(mds.latitude_bins, mds.time))
    residual = xr.DataArray(final[:,:,6], dims=('latitude', 'time'), coords=(mds.latitude_bins, mds.time))

    return xr.Dataset({'mean_amplitude': mean_amplitude, 
                    'mean_peak_height': mean_peak_height, 
                    'mean_thickness': mean_thickness,
                    'std_amplitude': std_amplitude,
                    'std_peak_height': std_peak_height,
                    'std_thickness': std_thickness,
                    'mean_residual': residual})
#%%
if __name__ == '__main__':
    def process(year):
        print(year)
        path = '/home/anqil/Documents/osiris_database/iris_oh/statistics/'
        filename = 'daily_ver_clima_{}.nc'
        with xr.open_dataset(path+filename.format(year)) as mds:
            ds_result = fit_clima(mds) 
            print('save', year)
            ds_result.to_netcdf(path + 'daily_agc_clima_{}.nc'.format(year))
    year_lst = range(2001, 2018)
    with Pool(processes=12) as p:
        p.map(process, year_lst)
