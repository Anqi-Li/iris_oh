#%%
import numpy as np
import xarray as xr
# from oem_functions import linear_oem
from geometry_functions import pathl1d_iris
import glob
from multiprocessing import Pool
from os import listdir
import pandas as pd
from astropy.time import Time
import sys

#%%
def linear_oem(y, K, Se_inv, Sa_inv, xa):
    if len(y.shape) == 1:
        y = y.reshape(len(y),1)
    if len(xa.shape) == 1:
        xa = xa.reshape(len(xa),1)
    G= np.linalg.solve(K.T.dot(Se_inv).dot(K) + Sa_inv, (K.T).dot(Se_inv))        
    x_hat = xa + G.dot(y - K.dot(xa)) 
    
    return x_hat.squeeze(), G

def mr_and_Sm(x_hat, K, Sa_inv, Se_inv, G):
    if len(x_hat.shape) == 1:
        x_hat = x_hat.reshape(len(x_hat),1)
    # A = Sa_inv + K.T.dot(Se_inv).dot(K)
    # b = K.T.dot(Se_inv)
    # G = np.linalg.solve(A, b) # gain matrix
    AVK = G.dot(K)
    MR = AVK.sum(axis=1)
    Se = np.linalg.inv(Se_inv)
#    Se = np.diag(1/np.diag(Se_inv)) #only works on diagonal matrix with no off-diagonal element
    Sm = G.dot(Se).dot(G.T) #retrieval noise covariance
    Ss = (AVK - np.eye(len(AVK))).dot(np.linalg.inv(Sa_inv)).dot((AVK - np.eye(len(AVK))).T)
#     Ss = np.linalg.inv(K.T.dot(Se_inv).dot(K) + Sa_inv).dot(Sa_inv).dot(np.linalg.inv(K.T.dot(Se_inv).dot(K) + Sa_inv))
    return MR, AVK, Sm, Ss

def oem_cost_pro(y, y_fit, x_hat, Se_inv, Sa_inv, xa, *other_args):
    if len(y.shape) == 1:
        y = y.reshape(len(y),1)
    if len(y_fit.shape) == 1:
        y_fit = y_fit.reshape(len(y_fit),1)
    if len(xa.shape) == 1:
        xa = xa.reshape(len(xa),1)
    if len(x_hat.shape) == 1:
        x_hat = x_hat.reshape(len(xa),1)    
    cost_x = (x_hat - xa).T.dot(Sa_inv).dot(x_hat - xa) / len(y)
    cost_y = (y-y_fit).T.dot(Se_inv).dot(y-y_fit) / len(y)
    return cost_x.squeeze(), cost_y.squeeze()


def invert_orbit_ch1(path_filename_limb, save_file=False, save_path_filename_ver=None, im_lst=None, return_AVK=False):
    
    orbit_num = path_filename_limb[-13:-7]
    print('process orbit {}'.format(orbit_num))
    ir = xr.open_dataset(path_filename_limb).sel(pixel=slice(21,128))
    ir.close()
    l1 = ir.data.where(ir.data.notnull(), drop=True).where(ir.sza>90, drop=True)
    time = l1.time
    if len(time)<1:
        print('orbit {} does not have sufficient images that satisfy criterion'.format(orbit_num))
        return
    error = ir.error.sel(time=time)
    tan_alt = ir.altitude.sel(time=time)

    if im_lst == None:
        im_lst = range(0,len(time))

    tan_low = 60e3
    tan_up = 95e3
    pixel_map = ((tan_alt>tan_low)*(tan_alt<tan_up))

    normalise = 4*np.pi/0.55 #optical filter overlap and radiance unit 4pi
    peak_apprx = 5e3 * normalise # approximation of the airglow peak, used in Sa

    #%% 1D inversion
    z = np.arange(tan_low-5e3, tan_up+20e3, 1e3) # m
    z_top = z[-1] + (z[-1]-z[-2]) #m
    xa = np.ones(len(z)) * 0 
    sigma_a = np.ones_like(xa) * peak_apprx
    sigma_a = np.ones_like(xa) * peak_apprx
    n = (z<tan_low).sum()
    sigma_a[np.where(z<tan_low)] = np.logspace(-1, np.log10(peak_apprx), n)
    n = (z>tan_up).sum()
    sigma_a[np.where(z>tan_up)] = np.logspace(np.log10(peak_apprx), -1, n)
    Sa_inv = np.diag(1 / sigma_a**2)

    mr = []
    error2_retrieval = []
    error2_smoothing = []
    ver = []
    time_save = []
    A_diag = []
    A_peak = []
    A_peak_height = []
    ver_cost_x, ver_cost_y = [], []
    for i in range(len(im_lst)):
        isel_args = dict(time=im_lst[i])
        h = tan_alt.isel(**isel_args).where(pixel_map.isel(**isel_args), drop=True)
        if len(h)<1:
            # print('image not enough pixels')
            continue
        K = pathl1d_iris(h.values, z, z_top)  *1e2 #m->cm
        y = l1.isel(**isel_args).reindex_like(h).values * normalise
        Se_inv = np.diag(1/(error.isel(**isel_args).reindex_like(h).values * normalise)**2)

        x, G = linear_oem(y, K, Se_inv, Sa_inv, xa)
        _, A, Sm, Ss = mr_and_Sm(x, K, Sa_inv, Se_inv, G)
        cost_x, cost_y = oem_cost_pro(y, y-K.dot(x), x, Se_inv, Sa_inv, xa)

        ver.append(x)
        A_diag.append(A.diagonal())
        A_peak.append(A.max(axis=1)) #max of each row
        A_peak_height.append(z[A.argmax(axis=1)]) #z of the A_peak
        mr.append(A.sum(axis=1)) #sum of each row
        error2_retrieval.append(np.diag(Sm))
        error2_smoothing.append(np.diag(Ss))
        ver_cost_x.append(cost_x)
        ver_cost_y.append(cost_y)
        time_save.append(time[im_lst[i]].values)

    if len(time_save) > 0:
        def attrs_dict(long_name, units, description):
            return dict(long_name=long_name, units=units, description=description)

        ds_ver = xr.Dataset().update({
            'time': (['time'], time_save),
            'z': (['z',], z, attrs_dict('altitude', 'm', 'Altitude grid of VER retrieval')),
            # 'pixel': (['pixel',], l1.pixel),
            'ver': (['time','z'], ver, attrs_dict('volume_emission_rate','photons cm-3 s-1', 'IRI OH(3-1) volume emission rate ')),
            'mr': (['time','z'], mr, attrs_dict('measurement_response', '1', 'Measurement response')),
            'A_diag': (['time','z'], A_diag, attrs_dict('AVK_diagonal', '1','Averaging kernel matrix diagonal elements')),
            'A_peak': (['time','z'], A_peak, attrs_dict('AVK_maximum', '1', 'Averaging kernel maximum in each row')),
            'A_peak_height': (['time','z'], A_peak_height, attrs_dict('AVK_max_height', 'm', 'Corresponding altitude of the averaging kernel maximum in each row')),
            'error2_retrieval': (['time','z'], error2_retrieval, attrs_dict('variance_measurement', '(photons cm-3 s-1)^2', 'Retrieval noise S_m diagonal elements (Rodgers (2000))')),
            'error2_smoothing': (['time','z'], error2_smoothing, attrs_dict('variance_smoothing', '(photons cm-3 s-1)^2', 'Smoothing error S_s diagonal elements (Rodgers (2000))')),
            'ver_cost_x': (['time',], ver_cost_x),
            'ver_cost_y': (['time',], ver_cost_y),
            'latitude': (['time',], ir.latitude.sel(time=time_save), attrs_dict('latitude', 'degrees north', 'Latitude at the tangent point')),
            'longitude': (['time',], ir.longitude.sel(time=time_save), attrs_dict('longitude', 'degrees east', 'Longitude at the tangent point')),
            'sza': (['time',], ir.sza.sel(time=time_save), attrs_dict('solar_zenith_angle', 'degrees', 'Solar Zenith Angle between the satellite line-of-sight and the sun')),
            'apparent_solar_time': (['time',], ir.apparent_solar_time.sel(time=time_save), attrs_dict('apparent_solar_time', 'hour', 'Apparent Solar Time at the line-of-sight tangent point')),
            'orbit': ir.orbit,
            })
        ds_ver = ds_ver.assign_attrs(dict(channel=ir.channel.values))
        
        if save_file:
            encoding = dict(
                orbit=dict(dtype='int32'), 
                time=dict(units='days since 1858-11-17'),
                )
            encoding.update({v: {'zlib': True, 'dtype': 'float32'} for v in ds_ver.data_vars 
                            if ds_ver[v].dtype == 'float64'})

            ds_ver.to_netcdf(save_path_filename_ver, encoding=encoding)
        
        if return_AVK:
            A = xr.DataArray(A, (('row_z',z), ('col_z', z)))
            return ds_ver, A
        else:
            return ds_ver
    
def find_orbit_stamps(year):
    time_stamp = pd.date_range(start='{}-12-31 23:59:59'.format(year-1), 
                                end='{}-01-31 23:59:59'.format(year+1),
                                freq='M')
    with xr.open_dataset('~/Documents/osiris_database/odin_rough_orbit.nc') as rough_orbit:
        rough_orbit = rough_orbit.rename({'mjd':'time'}).assign(
            time=Time(rough_orbit.mjd, format='mjd').datetime64
            ).interp(time=time_stamp).astype(int).orbit
    return rough_orbit

#%%
if __name__ == '__main__':
    path_limb = '/home/anqil/Documents/sshfs/oso_extra_storage/StrayLightCorrected/Channel1/'
    path_pattern_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/v2/{}{}/'
    filename_pattern_ver = 'iri_ch1_ver_{}.nc'
    orbit_error = []

    def process(year, month):
        rough_orbit = find_orbit_stamps(year=year)
        
        range_orbit = range(*tuple(rough_orbit.isel(time=slice(month-1,month+1)).values))
        orbit_num_in_month = [f[-13:-7] for f in sorted(glob.glob(path_limb+'*.nc')) 
                            if int(f[-13:-7]) in range_orbit]
        path_ver = path_pattern_ver.format(year, str(month).zfill(2))
        for orbit_num in orbit_num_in_month:
            path_filename_limb = path_limb + 'ir_slc_{}_ch1.nc'.format(orbit_num)
            ver_file_lst = glob.glob(path_ver + filename_pattern_ver.format('*'))
            ver_filename = filename_pattern_ver.format(orbit_num)
            if path_ver+ver_filename in ver_file_lst:
                print('orbit {} already exist'.format(orbit_num))
            else:
                try:
                    _ = invert_orbit_ch1(path_filename_limb, save_file=True,
                        save_path_filename_ver=path_ver+ver_filename)
                except:
                    raise
            
    # process(2002,1)
    def fun(month):
        year = int(sys.argv[1])
        process(year, month)
    if sys.argv[2] == 'a':
        months = range(1,7)
    elif sys.argv[2] == 'b':
        months = range(7,13)
    else:
        months = range(1,13)
    with Pool(processes=len(months)) as p:
        p.map(fun, months) 


# %%
# new_path_pattern_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/v2/{}{}/'
# test_new_file_ver = glob.glob(new_path_pattern_ver.format(2001,str(10).zfill(2))+'*.nc')[0]
# orbit_num = test_new_file_ver[-9:-3]
# old_path_pattern_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/4pi/'
# test_old_file_ver = old_path_pattern_ver+'iri_ch1_ver_{}.nc'.format(orbit_num)
# ds1 = xr.open_dataset(test_new_file_ver)
# ds1.close()
# ds2 = xr.open_dataset(test_old_file_ver)
# ds2.close()

# #%%
# var = 'A_diag'
# image = 100
# ds1[var].where(ds1.mr>0)[image].plot(y='z')
# ds2[var].where(ds2.mr>0)[image].plot(y='z')
# %%
