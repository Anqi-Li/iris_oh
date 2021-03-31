#%%
from operator import is_not
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
import glob
import matplotlib.pyplot as plt
from multiprocessing import Pool
from random import shuffle
import warnings
warnings.filterwarnings("ignore")

# %%
def gauss(x, a, x0, sigma):
    '''
    x: data
    a: amplitude
    x0: mean of data
    sigma: std
    '''
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def gauss_pdf_product(x1, x2, sigma1, sigma2):
    x = (x1 * sigma1**2 + x2 * sigma2**2)/(sigma1**2 + sigma2**2)
    sigma = np.sqrt((sigma1**2 * sigma2**2)/(sigma1**2 + sigma2**2))
    return x, sigma

def gauss_integral(amplitude, sigma, amplitude_error, sigma_error):
    integral_mean, integral_error = gauss_pdf_product(amplitude, sigma, amplitude_error, sigma_error)
    norm = np.sqrt(2*np.pi)
    return norm*integral_mean, norm*integral_error

def weighted_arithmetic_mean(x,y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y)) #this might become nan!
    return mean, sigma

def characterise_layer(da_ver_profile, a0=5e3, mean0=85e3, sigma0=5e3, ver_error_profile=None):
    y = da_ver_profile.dropna(dim='z')
    x = y.z
    if ver_error_profile is None:
        absolute_sigma = False
    else:
        absolute_sigma = True

    popt, pcov = curve_fit(gauss, x, y, 
                    p0=[a0, mean0, sigma0], 
                    sigma=ver_error_profile, absolute_sigma=absolute_sigma,
                    # bounds=([0, 70e3, 0], [1e5, 100e3, 40e3]) #some reasonable ranges for the airglow characteristics
                    )
    # residual = np.sqrt((y - gauss(x, *popt))**2).sum()
    r = y - gauss(x, *popt)
    chisq = sum((r / ver_error_profile) ** 2)/len(r)
    peak_intensity, peak_height, thickness_sigma = popt
    peak_intensity_error, peak_height_error, thickness_sigma_error = np.sqrt(np.diag(pcov))
    total_intensity, total_intensity_error = gauss_integral(peak_intensity, thickness_sigma, peak_intensity_error, thickness_sigma_error)
    cov_peak_intensity_peak_height, cov_peak_height_peak_sigma = np.diag(pcov, 1)
    cov_peak_intensity_peak_sigma = np.diag(pcov, 2)

    return (peak_intensity, peak_height, abs(thickness_sigma), 
        peak_intensity_error, peak_height_error, thickness_sigma_error,
        chisq, total_intensity, total_intensity_error, 
        cov_peak_intensity_peak_height, cov_peak_intensity_peak_sigma, cov_peak_height_peak_sigma)

def process_file(ver_file, save_file=False, agc_file_pattern=None):
    ds_ver = xr.open_dataset(ver_file)
    ds_ver.close()

    # ver_data = ds_ver.ver.where(ds_ver.A_diag>-1)
    # ver_error_data = ds_ver.error2_retrieval.where(ds_ver.A_diag>-1).pipe(np.sqrt).pipe(
    #     lambda x: xr.where(x==0, 1e-1, x)) #replace 0 std value to 0.1
    # ver_error_data = ds_ver.error2_retrieval.pipe(np.sqrt).where(ds_ver.A_diag>0.8, other=6e3)
    
    #option 1: whole VER and error profile
    # ver_data = ds_ver.ver
    # ver_error_data = ds_ver.error2_retrieval.pipe(np.sqrt).where(
    #     ds_ver.error2_retrieval!=0, other=1e-1)
    
    #option 2: modify the error profiles
    # ver_data = ds_ver.ver
    # ver_error_data = ds_ver.error2_retrieval.pipe(np.sqrt).where(
    #     ds_ver.error2_retrieval!=0, other=1e-1).where(
    #         ds_ver.A_peak>0.8, other=6e3)

    #option 3: shortern the VER and error profiles
    ver_data = ds_ver.ver.where(ds_ver.A_peak>0.8)
    ver_error_data = ds_ver.error2_retrieval.pipe(np.sqrt).where(
        ds_ver.error2_retrieval!=0, other=1e-1).where(
            ds_ver.A_peak>0.8)

    time_save = []
    zenith_intensity, zenith_intensity_error = [], []
    peak_intensity, peak_height, peak_sigma = [], [], []
    peak_intensity_error, peak_height_error, peak_sigma_error = [], [], []
    cov_peak_intensity_peak_height, cov_peak_intensity_peak_sigma, cov_peak_height_peak_sigma = [], [], []
    # residual = []
    chisq = []
    for i in range(len(ver_data.time)):
        # print('{}/{}'.format(i, ds_ver.orbit.item()))
        isel_args = dict(time = i)
        if len(ver_data.isel(**isel_args).dropna('z')) < 10:
            pass
        elif ver_data.isel(**isel_args).dropna('z').z.min() > 75e3:
            pass
        elif ver_data.isel(**isel_args).dropna('z').z.max() < 88e3:
            pass
        else:
            try:
                char = characterise_layer(ver_data.isel(**isel_args).dropna('z'),
                    ver_error_profile=ver_error_data.isel(**isel_args).dropna('z').values
                    )
                peak_intensity.append(char[0])
                peak_height.append(char[1])
                peak_sigma.append(char[2])
                peak_intensity_error.append(char[3])
                peak_height_error.append(char[4])
                peak_sigma_error.append(char[5])
                # residual.append(char[6])
                chisq.append(char[6])
                zenith_intensity.append(char[7])
                zenith_intensity_error.append(char[8])
                cov_peak_intensity_peak_height.append(char[9].item()) 
                cov_peak_intensity_peak_sigma.append(char[10].item())
                cov_peak_height_peak_sigma.append(char[11].item())
                time_save.append(ver_data.time.isel(**isel_args))
            except:
                pass
                # raise
    if len(time_save) > 0:
        time_save = xr.concat(time_save, dim='time')
        ds_gauss_para = xr.Dataset({
            'time': (['time'], time_save),
            'longitude': (['time'], ds_ver.longitude.sel(time=time_save)),
            'latitude': (['time'], ds_ver.latitude.sel(time=time_save)),
            'apparent_solar_time': (['time'], ds_ver.apparent_solar_time.sel(time=time_save)),
            'sza': (['time'], ds_ver.sza.sel(time=time_save)),
            'zenith_intensity': (['time'], zenith_intensity, dict(units='photons cm-2 s-1')),
            'zenith_intensity_error': (['time'], zenith_intensity_error, dict(units='photons cm-2 s-1')),
            'peak_intensity': (['time'], peak_intensity, dict(units=ds_ver.ver.units)),
            'peak_height': (['time'], peak_height, dict(units=ds_ver.z.units)),
            'peak_sigma': (['time'], peak_sigma, dict(units=ds_ver.z.units)),
            'peak_intensity_error': (['time'], peak_intensity_error, dict(units=ds_ver.ver.units)),
            'peak_height_error': (['time'], peak_height_error, dict(units=ds_ver.z.units)),
            'peak_sigma_error': (['time'], peak_sigma_error, dict(units=ds_ver.z.units)),
            'cov_peak_intensity_peak_height': (['time'], cov_peak_intensity_peak_height, dict(units=ds_ver.ver.units+ds_ver.z.units)),
            'cov_peak_intensity_peak_sigma': (['time'], cov_peak_intensity_peak_sigma, dict(units=ds_ver.ver.units+ds_ver.z.units)),
            'cov_peak_height_peak_sigma': (['time'], cov_peak_height_peak_sigma, dict(units=ds_ver.z.units+ds_ver.z.units)),
            # 'residual':(['time'], residual, dict(units=ds_ver.ver.units)),
            'chisq':(['time'], chisq),
            'orbit': ds_ver.orbit,
            'channel': ds_ver.channel,
        })
        if save_file:
            orbit_num = str(ds_ver.orbit.item()).zfill(6)
            print('saving agc file {}'.format(orbit_num))
            ds_gauss_para.to_netcdf(agc_file_pattern.format(orbit_num))
        return ds_gauss_para

#%%
if __name__ == '__main__':
    ver_path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/'
    ver_file_lst = glob.glob(ver_path + '*nc')
    gauss_path = '/home/anqil/Documents/osiris_database/iris_oh/'
    gauss_filename_pattern = 'gauss_character/A_filtered/gauss_{}.nc'

    # rough estimates of odin year-orbits
    orbit_year = xr.open_dataset('/home/anqil/Documents/osiris_database/odin_rough_orbit_year.nc')
    orbit_year.close()

    # for f in ver_file_lst:
    def fun(f):
        orbit_num = f[-9:-3]
        gauss_file_lst = glob.glob(gauss_path + gauss_filename_pattern.format('*'))
        
        if orbit_num in [k[-9:-3] for k in gauss_file_lst]:
            print('{} already done'.format(orbit_num))
            pass
        else:
            print('process orbit {}'.format(orbit_num))
            process_file(f, save_file=True, agc_file_pattern=gauss_path + gauss_filename_pattern)
            # process_file(f, save_file=False)
    
    def process_year(year):
        orbit_range = tuple(orbit_year.orbit.sel(year=[year, year+1]).values)
        print(orbit_range)
        yearly_ver_file_lst = [f for f in ver_file_lst 
            if int(f[-9:-3]) in range(
                *orbit_range)]
        shuffle(yearly_ver_file_lst)
        for f in yearly_ver_file_lst:
            fun(f)

    with Pool(processes=8) as p:
        p.map(process_year, range(2007, 2013))
# %%
