#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import glob
from characterise_agc_routine import gauss

#%%

def find_spectral_max(sp):
    test = sp.sel(freq=slice(1/40e3, 1/1e3))
    max_power_freq = test.freq.isel(freq=test.argmax('freq'))
    max_power_freq = max_power_freq.where(max_power_freq > max_power_freq.min())
    max_power = test.isel(freq=test.argmax('freq'))
    max_power = max_power.where(max_power_freq > max_power_freq.min())
    return max_power_freq.rename('maximum power freq'), max_power.rename('power')


def data_FFT(data):
    sp = data.transpose('time', 'z').interpolate_na('z').fillna(0).pipe(np.fft.fft, axis=1)                 
    freq = np.fft.fftfreq(len(data.z), d=1e3)
    # sp = xr.DataArray(abs(sp), coords=(('time', data.time), ('freq', freq))).sortby('freq')
    # sp['freq'] = sp.freq.assign_attrs(dict(units='m-1'))
    ds_sp = xr.Dataset({'data': data,
                        'freq': (['freq',], freq, {'units': 'm-1'}),
                        'sp': (['time', 'freq'], abs(sp))
                        })
    return ds_sp

def characterise_wave(ds, window_size=20, min_window_size=10):
    # reconstruct gaussion curve
    gauss_fit = []
    for a, x0, sigma in zip(ds.amplitude, 
        ds.peak_height, ds.thickness):
        gauss_fit.append(ds.z.pipe(gauss, a, x0, sigma))

    gauss_fit = xr.concat(gauss_fit, 'time').assign_coords(
        time=ds.time).rename('gaussian ver')
    
    ver_data = ds.ver.where(ds.A_diag>0.8)
    data = (ver_data - gauss_fit).dropna('z', 'all').rolling(time=window_size, center=True, min_periods=min_window_size).mean()
    ds_sp = data_FFT(data)
    return ds_sp


def process_file(ver_f, save_file=True):
    orbit_num = ver_f[-9:-3]
    # open VER file
    ds = xr.open_dataset(ver_f)

    window_size = 20
    min_window_size = 10
    if len(ds.time) <= min_window_size:
        return
    else:
        #open airglow character (agc) file
        path = '/home/anqil/Documents/osiris_database/iris_oh/'
        filename = '/airglow_character/agc_{}.nc'
        ds_agc = xr.open_dataset(path+filename.format(orbit_num))
        ds = ds.update(ds_agc)

        # spectral FFT
        ds_sp = characterise_wave(
            ds, window_size=window_size, min_window_size=min_window_size).reindex(
                z=ds.z).sortby('freq').sel(freq=slice(0,1))
        max_pw_freq, max_pw = find_spectral_max(ds_sp.sp)

        ds_save = ds_sp.update({'longitude': ds.longitude,
                                    'latitude': ds.latitude,
                                    'orbit': ds.orbit,
                                    'channel': ds.channel,
                                    'max_pw': max_pw,
                                    'max_pw_freq': max_pw_freq
                                    })
        # save data
        if save_file:
            ds_save.to_netcdf(path + 'spectral_character/sp_{}.nc'.format(orbit_num))
        
        return ds_save
            

#%%
def process_orbit(orbit_num):
    # open VER file
    path = '/home/anqil/Documents/osiris_database/iris_oh/'
    filename = 'iri_oh_ver_{}.nc'
    ds = xr.open_dataset(path+filename.format(orbit_num))

    # open airglow character (agc) file
    filename = 'airglow_character/agc_{}.nc'
    ds_agc = xr.open_dataset(path+filename.format(orbit_num))
    ds = ds.update(ds_agc)

    # spectral FFT
    ds_sp = characterise_wave(ds).reindex(z=ds.z).sortby('freq').sel(freq=slice(0,1))
    max_pw_freq, max_pw = find_spectral_max(ds_sp.sp)

    # save data
    ds_save = ds_sp.update({'longitude': ds.longitude,
                            'latitude': ds.latitude,
                            'orbit': ds.orbit,
                            'channel': ds.channel,
                            'max_pw': max_pw,
                            'max_pw_freq': max_pw_freq
                            })
    return ds_save

#%% test on one orbit
# orbit = 6196
# orbit_num = str(orbit).zfill(6)
# ds = process_orbit(orbit_num)
# ds = ds.update({'tp': (['time',], np.linspace(0,1,len(ds.time)))}).swap_dims({'time': 'tp'})
# fig, ax = plt.subplots(4,1, figsize=(6,8), sharex=True)
# ds.data.plot.contourf(x='tp', ax=ax[0], ylim=(60e3, 95e3), add_colorbar=False, cmap='RdBu', vmin=-1e3, vmax=1e3)
# ds.sp.plot.contourf(x='tp', ax=ax[1], add_colorbar=False)
# ds.max_pw.plot(x='tp', ax=ax[2])
# ds.max_pw_freq.plot(x='tp', ax=ax[3])

# [ax[i].set(xlabel='') for i in range(3)]
# ax[0].set(title='orbit {}'.format(orbit))

# %% 
if __name__ == '__main__':
    path = '/home/anqil/Documents/osiris_database/iris_oh/spectral_character/'
    sp_filelist = glob.glob(path + '*.nc')
    orbit = 3713 #4814 #3713
    while orbit < 30000:
        try:
            orbit_num = str(orbit).zfill(6)
            sp_filename = path + 'sp_{}.nc'.format(orbit_num)
            if sp_filename in sp_filelist:
                print('orbit {} already processed'.format(orbit_num))
            else:
                print('process orbit {}'.format(orbit_num))
                process_orbit(orbit_num).to_netcdf(sp_filename) 
            orbit += 1
        except FileNotFoundError:
            orbit += 1
        except ValueError:
            orbit +=1
            
# %%