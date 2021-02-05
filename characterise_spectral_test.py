#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from xhistogram.xarray import histogram
import glob

# %% open VER file
orbit = 3713#4814 #3713
orbit_num = str(orbit).zfill(6)
path = '/home/anqil/Documents//sshfs/oso_extra_storage/VER/oh/'
filename = 'iri_oh_ver_{}.nc'
ds = xr.open_dataset(path+filename.format(orbit_num))
ds.close()
ds = ds.update({'tp' : (['time'],
        (ds.time - ds.time[0])/(ds.time[-1]-ds.time[0]))})
#%% open airglow character (agc) file
path = '/home/anqil/Documents/osiris_database/iris_oh/'
filename = '/airglow_character/agc_{}.nc'
ds_agc = xr.open_dataset(path+filename.format(orbit_num))
ds_agc.close()
ds = ds.update(ds_agc).swap_dims({'time': 'tp'})

ver_data = ds.ver.where(ds.mr>0.8)

#%% reconstruct the gaussian function
from characterise_agc_routine import gauss
gauss_fit = []
for a, x0, sigma in zip(ds.amplitude, 
    ds.peak_height, ds.thickness):
    gauss_fit.append(ds.z.pipe(gauss, a, x0, sigma))

gauss_fit = xr.concat(gauss_fit, 'tp').assign_coords(
    tp=ds.tp).rename('gaussian ver')

#%% Plot gaussian fit and anomaly
fig, ax = plt.subplots(3,1, sharex=True, sharey=True)
plot_args = dict(cmap='viridis', x='tp', robust=True, vmin=0, vmax=7e3)
ver_data.plot(ax=ax[0], **plot_args)
gauss_fit.plot(ax=ax[1], **plot_args)
(ver_data - gauss_fit).plot(ax=ax[2], y='z', vmax=3e3, vmin=-3e3)

ax[0].set(title='Original VER',
            xlabel='')
ax[1].set(title='Gaussian fitted VER',
            xlabel='')
ax[2].set(title='Org - Gauss')


#%% Plot moving average of the anomaly
fig, ax = plt.subplots(3,1, sharex=True, sharey=True)
plot_args = dict(cmap='RdBu', y='z', robust=True)
(ver_data - gauss_fit).plot(ax=ax[0], **plot_args)
(ver_data - gauss_fit).rolling(tp=10).mean().plot(ax=ax[1], **plot_args)
(ver_data - gauss_fit).rolling(tp=20).mean().plot(ax=ax[2], **plot_args)

ax[0].set(title='Org - Gauss', xlabel='')
ax[1].set(title='moving window 10 image', xlabel='')
ax[2].set(title='moving window 20 image', xlabel='')

#%%
def data_FFT(data):
        sp = data.transpose('tp', 'z').interpolate_na('z').fillna(0).pipe(np.fft.fft, axis=1)                 
        freq = np.fft.fftfreq(len(data.z), d=1e3)
        sp = xr.DataArray(abs(sp), coords=(('tp', ds.tp), ('freq', freq))).sortby('freq')
        sp['freq'] = sp.freq.assign_attrs(dict(units='m-1'))
        return sp

window_size = 20
data = (ver_data - gauss_fit).dropna('z', 'all').rolling(tp=window_size, center=True).mean()
sp = data_FFT(data)
# sp = sp.assign_coords(freq=1/freq).rename({'freq': 'wavelength [m]'}).sortby('wavelength [m]')

# Plot spectral domain
fig, ax = plt.subplots(2,1, sharex=True)
data.plot.contourf(ax=ax[0], x='tp', cmap='RdBu', vmin=-1e3, vmax=1e3)
sp.plot.contourf(ax=ax[1], x='tp', cmap='viridis', levels=np.logspace(1,4))
plt.ylim(bottom=0)

ax[0].set(title='Data to transform with SMA ({})'.format(window_size), xlabel='')
ax[1].set(title='Spectral domain')

ax[1].set(ylabel='Ver. Wavelength [m]')
# ax[1].set_yticklabels(1/ax[1].get_yticks())
ax[1].set_yticks(np.linspace(0, sp.freq.max(), 4))
ax[1].set_yticklabels((1/np.linspace(0, sp.freq.max(), 4)).round(0))
# ax[1].axhline(y=1/20e3)
# ax[1].axhline(y=1/10e3)
ax[1].axhline(y=1/40e3, color='w')
plt.show()
#%%
def characterise_spectral(sp):
        test = sp.sel(freq=slice(1/40e3, 1/1e3))
        max_power_freq = test.freq.isel(freq=test.argmax('freq'))
        max_power_freq = max_power_freq.where(max_power_freq > max_power_freq.min())
        max_power = test.isel(freq=test.argmax('freq'))
        max_power = max_power.where(max_power_freq > max_power_freq.min())
        return max_power_freq.rename('maximum power freq'), max_power.rename('power')

max_power_freq, max_power = characterise_spectral(sp)

fig, ax = plt.subplots(2,1, sharex=True)
max_power_freq.pipe(lambda x: 1/x).rename('wavelength').assign_attrs(dict(units='m')).plot(ax=ax[0], x='tp')
plt.figure()
max_power.plot(ax=ax[1], x='tp')

#%% make zonal mean airglow profile over 1 day
ver_file_lst = glob.glob(path + '*nc')
result_1d = xr.open_mfdataset(ver_file_lst).set_coords(['longitude', 'latitude'])
result_1d_oneday = result_1d.where(
        result_1d.time.dt.dayofyear == ds.time.dt.dayofyear.pipe(np.unique)[0], drop=True)
ver_data_oneday = result_1d_oneday.ver.where(result_1d.A_peak>0.8, drop=True)

dlat = 10
latitude_bins = np.arange(-90,90+dlat,dlat)
latitude_labels = latitude_bins[1:]-dlat/2
zonal_mean = ver_data_oneday.groupby_bins(
        ver_data_oneday.latitude, 
        bins=latitude_bins, labels=latitude_labels).median(
                'time', keep_attrs=True)
zonal_count = ver_data_oneday.groupby_bins(
        ver_data_oneday.latitude, 
        bins=latitude_bins, labels=latitude_labels).count(
                'time', keep_attrs=True)
# zonal_mean.plot(y='z', cmap='viridis', vmin=0, vmax=7e3)

#%% Plot zonal mean and anomaly

# zonal_fit = zonal_mean.sel(latitude_bins=ds.latitude, method='nearest')
zonal_fit = zonal_mean.dropna('latitude_bins', 'all').fillna(0).load().interp(
        latitude_bins=ds.latitude.dropna('tp'))

fig, ax = plt.subplots(5,1, figsize=(6,8), sharex=True, sharey=True)
plot_ver_args = dict(cmap='viridis', y='z', vmin=0, vmax=7e3, ylim=(60e3, 95e3))
plot_anomaly_args = dict(cmap='RdBu', y='z', vmin=-1e3, vmax=1e3, ylim=(60e3, 95e3))
ver_data.plot(ax=ax[0], **plot_ver_args)
zonal_fit.plot(ax=ax[1], **plot_ver_args)
(ver_data - zonal_fit).rolling(
        tp=20, min_periods=10, center=True).mean().plot.contourf(
                ax=ax[2], **plot_anomaly_args)
gauss_fit.plot(ax=ax[3], **plot_ver_args)
(ver_data - gauss_fit).rolling(
        tp=20, min_periods=10, center=True).mean().plot.contourf(
                ax=ax[4], **plot_anomaly_args)


ax[0].set(title='Original VER orbit {}'.format(orbit))
ax[1].set(title='Daily zonal averaged VER')
ax[2].set(title='Org - Average')
ax[3].set(title='Gauss fit VER')
ax[4].set(title='Org - Gauss')
[ax[i].set(xlabel='') for i in range(len(ax)-1)]

#%%
import cartopy.crs as ccrs
fig, ax = plt.subplots(1,1, 
        subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=180)))
ax.plot(ver_data_oneday.longitude, ver_data_oneday.latitude, marker='*', ls='')
ax.coastlines()
ax.set_global()
ax.set(title='orbit path in day no.{} of 2002'.format(
        ds.time.dt.dayofyear.pipe(np.unique)[0]))




#%%


























#%% create moving average
moving_window_big = 200
ver_sma_big = ver_data.rolling(
    time=moving_window_big, center=True).mean().rename('ver_sma_big')
moving_window_small = 20
ver_sma_small = ver_data.rolling(
    time=moving_window_small, center=True).mean().rename('ver_sma_small')
#%% plot moving average
fig, ax = plt.subplots(3,1, figsize=(5,6), sharex=True, sharey=True)
plot_args = dict(y='z', vmin=0, vmax=1e4, cmap='viridis', robust=True, ylim=[60e3,100e3])

ver_data.plot(ax=ax[0], **plot_args)
ver_sma_small.plot(ax=ax[1], **plot_args)
ver_sma_big.plot(ax=ax[2], **plot_args)

ax[0].set(title='Original',
        xlabel='')
ax[1].set(title='Moving Average ({})'.format(moving_window_small),
        xlabel='')
ax[2].set(title='Moving Average ({})'.format(moving_window_big),
        )

# fig.savefig('/home/anqil/Documents/osiris_database/iris_oh/full_orbit_ver_{}.png'.format(ds.orbit.item()))
plt.show()

# %% plot differences 
fig, ax = plt.subplots(3,1, figsize=(5,6), sharex=True, sharey=True)
plot_args = dict(y='z', vmin=-2000, vmax=2000, ylim=(60e3, 100e3))

ver_mean = ver_data.mean('time')
(ver_data - ver_mean).plot(ax=ax[0], **plot_args)
(ver_data - ver_sma_big).plot(ax=ax[1], **plot_args)
(ver_sma_small - ver_sma_big).plot(ax=ax[2], **plot_args)

ax[0].set(title='Original - Total mean of the orbit',
        xlabel='')
ax[1].set(title='Original - Moving Average ({})'.format(moving_window_small),
        xlabel='')    
ax[2].set(title='MA ({}) - MA ({})'. format(moving_window_small, moving_window_big),
        )

# fig.savefig('/home/anqil/Documents/osiris_database/iris_oh/structure_{}.png'.format(ds.orbit.item()))
plt.show()

# %% Check mean, std, histogram
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
# Choose colormap
cmap = pl.cm.viridis
# Get the colormap colors
my_cmap = cmap(np.arange(cmap.N))
# Set alpha
my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
# Create new colormap
my_cmap = ListedColormap(my_cmap)

def plot_hist2d(data, ax):
    if data.name == None:
        data.rename('data')
    mean = data.mean('time')
    std = data.std('time')
    h_z = histogram(data, bins=[np.linspace(-1e3,8e3)], dim=['time'])
    h_z.plot(ax=ax, vmax=h_z.sel(z=85e3).max(), cmap=my_cmap)
    mean.plot(ax=ax, y='z', color='k', ls='-')
    std.pipe(lambda x: mean+x).plot(ax=ax, y='z', color='k', ls='--')
    std.pipe(lambda x: mean-x).plot(ax=ax, y='z', color='k', ls='--')

fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
plot_hist2d(ver_data, ax[0])
plot_hist2d(ver_sma_big, ax[1])

ax[0].set(title='Original',
        xlabel='')
ax[1].set(title='Running mean ({}-image-window)'.format(moving_window_big),
        ylim=(60e3,100e3))

ds.error2_retrieval.pipe(
    lambda x: np.sqrt(x) + ver_data).mean(
        'time').plot(ax=ax[0], y='z', color='b', ls=':')
ds.error2_retrieval.pipe(
    lambda x: -np.sqrt(x) + ver_data).mean(
        'time').plot(ax=ax[0], y='z', color='b', ls=':')


# %% FFT
omega = 1
omega_2 = 0.1
A = 1
phi = 0
dt = 0.1
t = np.arange(0, 256, dt)#np.arange(0, 2*np.pi*omega, 2*np.pi*omega/100) #np.arange(256)
x = A * (np.sin(omega*t + phi) + np.sin(omega_2*t + phi))
plt.figure()
plt.plot(t,x)

plt.figure()
sp = np.fft.fft(x)
freq = np.fft.fftfreq(t.shape[-1], dt)
plt.plot(freq, sp.real)
plt.xlim([0, 0.3])
plt.ylim(bottom=0)
# plt.vlines(omega/(2*np.pi), 0, 30, colors='k')
plt.gca().axvline(x=omega/(2*np.pi), color='k')
plt.gca().axvline(x=omega_2/(2*np.pi), color='k')