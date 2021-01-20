#%%
import numpy as np
import xarray as xr
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# from scipy.interpolate import interp1d
from oem_functions import linear_oem
from geometry_functions import pathl1d_iris
import glob

#%%
def invert_1d(orbit, ch, path, save_file=False, ver_file_pattern=None, im_lst=None):
    orbit_num = str(orbit).zfill(6)
    filename = 'ir_slc_{}_ch{}.nc'.format(orbit_num, ch)
    ir = xr.open_dataset(path+filename).sel(pixel=slice(21,128))
    l1 = ir.data.where(ir.data.notnull(), drop=True).where(ir.sza>90, drop=True)
    time = l1.time
    if len(time)<1:
        print('orbit {} does not have sufficient images that satisfy criterion'.format(orbit))
        return
    error = ir.error.sel(time=time)
    tan_alt = ir.altitude.sel(time=time)

    if im_lst == None:
        im_lst = range(0,len(time))

    tan_low = 60e3
    tan_up = 95e3
    pixel_map = ((tan_alt>tan_low)*(tan_alt<tan_up))
    if ch==1:
        peak_apprx = 5e3 #7.5e3  # approximation of the airglow peak, used in Sa
    elif ch==3:
        peak_apprx = 5e4 

    #%% 1D inversion
    z = np.arange(tan_low-5e3, tan_up+20e3, 1e3) # m
    z_top = z[-1] + (z[-1]-z[-2]) #m
    xa = np.ones(len(z)) * 0 
    sigma_a = np.ones_like(xa) * peak_apprx
    # sigma_a[np.where(np.logical_or(z<tan_low, z>tan_up))] = 1e-1
    sigma_a = np.ones_like(xa) * peak_apprx
    n = (z<tan_low).sum()
    sigma_a[np.where(z<tan_low)] = np.logspace(-1, np.log10(peak_apprx), n)
    n = (z>tan_up).sum()
    sigma_a[np.where(z>tan_up)] = np.logspace(np.log10(peak_apprx), -1, n)
    Sa = np.diag(sigma_a ** 2)

    mr = []
    error2_retrieval = []
    error2_smoothing = []
    ver = []
    limb_fit = []
    time_save = []
    A_diag = []
    A_peak = []
    A_peak_height = []
    for i in range(len(im_lst)):
        # print('{}/{}/{}'.format(i, len(im_lst), orbit))
        isel_args = dict(time=im_lst[i])
        h = tan_alt.isel(**isel_args).where(pixel_map.isel(**isel_args), drop=True)
        if len(h)<1:
            print('not enough pixels')
            continue
        K = pathl1d_iris(h.values, z, z_top)  *1e2 #m->cm
        y = l1.isel(**isel_args).reindex_like(h).values
        Se = np.diag(error.isel(**isel_args).reindex_like(h).values**2)
        x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
        ver.append(x)
        A_diag.append(A.diagonal())
        A_peak.append(A.max(axis=1)) #max of each row
        A_peak_height.append(z[A.argmax(axis=1)]) #z of the A_peak
        mr.append(A.sum(axis=1)) #sum of each row
        error2_retrieval.append(np.diag(Sm))
        error2_smoothing.append(np.diag(Ss))
        limb_fit.append(xr.DataArray(y-K.dot(x), coords=[('pixel', h.pixel)]
                        ).reindex(pixel=l1.pixel))
        time_save.append(time[im_lst[i]].values)

    if len(time_save) > 0:
        result_1d = xr.Dataset().update({
            'time': (['time'], time_save),
            'z': (['z',], z, {'units': 'm'}),
            'pixel': (['pixel',], l1.pixel),
            'ver': (['time','z'], ver, {'long name': 'VER', 'units': 'photons cm-3 s-1'}),
            'mr': (['time','z'], mr),
            'A_diag': (['time','z'], A_diag),
            'A_peak': (['time','z'], A_peak),
            'A_peak_height': (['time','z'], A_peak_height),
            'error2_retrieval': (['time','z'], error2_retrieval),
            'error2_smoothing': (['time','z'], error2_smoothing),
            'limb_fit': (['time','pixel'], limb_fit),
            'latitude': (['time',], ir.latitude.sel(time=time_save)),
            'longitude': (['time',], ir.longitude.sel(time=time_save)),
            'orbit': ir.orbit,
            'channel': ir.channel,
            })
        if save_file:
            if ver_file_pattern == None:
                result_1d.to_netcdf('~/Documents/osiris_database/iris_oh/iri_oh_ver_{}.nc'.format(orbit_num))
            else:
                result_1d.to_netcdf(ver_file_pattern.format(orbit_num))
        return result_1d
    # else:
        # print('orbit {} does not have sufficient images that satisfy criterion'.format(orbit))
        # return

#%%
if __name__ == '__main__':
    ch = 1
    path_limb = '/home/anqil/Documents/sshfs/oso_extra_storage/StrayLightCorrected/Channel{}/'.format(ch)
    orbit = 3713
    orbit_error = []
    # path_ver = '/home/anqil/Documents/osiris_database/iris_oh/'
    path_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/oh/'
    ver_filename_pattern = 'iri_oh_ver_{}.nc'
    ver_file_lst = glob.glob(path_ver + ver_filename_pattern.format('*'))
    while orbit < 90000:
        try:
            if path_ver+ver_filename_pattern.format(str(orbit).zfill(6)) in ver_file_lst:
                print('orbit {} already exist'.format(orbit))
                pass
            else:
                print('process orbit {}'.format(orbit))
                _ = invert_1d(orbit, ch, path_limb, save_file=True, 
                    ver_file_pattern=path_ver+ver_filename_pattern)
            orbit += 10
        except FileNotFoundError:
            orbit += 1
            print('invert the next orbit')
        except OSError:
            orbit_error.append(orbit)
            orbit += 1

            
            

# %%
