#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d

from osirisl1services.readlevel1 import open_level1_ir
from osirisl1services.services import Level1Services
import os
os.environ["ODINORBITDIR"] = '/home/anqil/Documents/osiris_database/odin-osiris.usask.ca/Level1/orbit'

#%%
channel = 1
orbit = 4700
ir = open_level1_ir(orbit, channel)
tan_alt = ir.l1.altitude
tan_lat = ir.l1.latitude
tan_lon = ir.l1.longitude
sc_look = ir.l1.look_ecef
sc_pos = ir.l1.position_ecef
l1 = ir.data

# ====drop all dates which have nan in l1
l1 = l1.dropna('mjd')
sc_look = sc_look.sel(mjd=l1.mjd)
sc_pos = sc_pos.sel(mjd=l1.mjd)
tan_lat = tan_lat.sel(mjd=l1.mjd)
tan_lon = tan_lon.sel(mjd=l1.mjd)
tan_alt = tan_alt.sel(mjd=l1.mjd)
mjd = l1.mjd.data
pixel = l1.pixel.data

# %%
im_lst = np.arange(850,1120,1)
pix_lst = np.arange(22, 128)
label_interval = 50
len(im_lst)

#%% interpolation 
# alts_interp = np.arange(10e3, 120e3, .25e3)
# data_interp = []
# for (data, alt) in zip(l1, tan_alt):
#     f = interp1d(alt, data, bounds_error=False)
#     data_interp.append(f(alts_interp))
# data_interp = xr.DataArray(data_interp, coords=[l1.mjd, alts_interp],
#                            dims=['mjd', 'altitude'])
# # plotting
# plt.figure(figsize=(15,6))
# data_interp.plot(x='mjd', y='altitude',
#                  norm=LogNorm(), vmin=1e9, vmax=1e13)
# # plt.title(str(num2date(ir.mjd[0],units))+' channel '+str(channel))
# plt.axvline(x=mjd[im_lst[0]], color='k', linewidth=5)
# plt.axvline(x=mjd[im_lst[-1]], color='k', linewidth=5)
# ax = plt.gca()
# ax.set_xticks(mjd[np.arange(0,len(mjd),300, dtype=int)])
# ax.set_xticklabels(np.arange(0,len(mjd),300))
# ax.set(xlabel='image index')
# plt.show()
# plt.figure()
# data_interp.isel(mjd=im_lst).plot(x='mjd', y='altitude', 
#                  norm=LogNorm(), 
#                  vmin=1e9, vmax=1e12)#, 
#                  #size=5, aspect=3)
# ax = plt.gca()
# ax.set(title='zoom in im_lst',
#       xlabel='image index')
# ax.set_xticks(mjd[im_lst[::label_interval]])
# ax.set_xticklabels(im_lst[::label_interval])
# plt.show()

#%% change coordinate
#====define the new base vectors
n_zenith = sc_pos.isel(mjd=0)
n_crosstrack = np.cross(sc_look.isel(mjd=0, pixel=60), n_zenith)
n_vel = np.cross(n_zenith, n_crosstrack)

#====tangent points in alpha, beta, rho coordinate
import pandas as pd
from geometry_functions import lla2ecef, cart2sphe, change_of_basis
tan_ecef = xr.concat(lla2ecef(tan_lat,tan_lon,tan_alt), 
                     pd.Index(['x','y','z'], name='xyz'))

tan_alpha = []
tan_beta = []
tan_rho = []
for i in im_lst:
    p_old = tan_ecef.isel(mjd=i, pixel=pix_lst)
    p_new = change_of_basis(n_crosstrack, n_vel, n_zenith, p_old)
    alpha, beta, rho = cart2sphe(p_new.sel(xyz='x'),
                                 p_new.sel(xyz='y'),
                                 p_new.sel(xyz='z'))
    tan_alpha.append(alpha)
    tan_beta.append(beta)
    tan_rho.append(rho)
tan_alpha = xr.DataArray(tan_alpha, 
                         coords=[mjd[im_lst], pixel[pix_lst]],
                         dims=['mjd', 'pixel'])
tan_beta = xr.DataArray(tan_beta, 
                        coords=[mjd[im_lst], pixel[pix_lst]],
                        dims=['mjd', 'pixel'])
tan_rho = xr.DataArray(tan_rho, 
                       coords=[mjd[im_lst], pixel[pix_lst]],
                       dims=['mjd', 'pixel'])

Re = 6371 + 100 #Earth radius in km

#%% plot tangent points in alpha-beta-rho space
plt.figure()
plt.plot(im_lst, tan_alpha.isel(pixel=60)*180/np.pi,label='cross')
plt.plot(im_lst, tan_beta.isel(pixel=60)*180/np.pi, label='along')

plt.plot(tan_lat.isel(pixel=60), '--', label='lat')
plt.plot(tan_lon.isel(pixel=60), '--', label='lon')
plt.legend()

#%% 1D inversion
from oem_functions import linear_oem
from geometry_functions import pathl1d_iris

z = np.arange(25e3, 175e3, 2e3) # m
z_top = z[-1] + 2e3
result_1d = np.zeros((len(im_lst), len(z)))
xa = np.ones(len(z)) * 0 # temp
Sa = np.diag(np.ones(len(z)))*1e10
Se = np.diag(np.ones(len(pixel))) * (2.5e10)**2# 1e10 #30 #temporary
Ave = []
for i in range(len(im_lst)):
    h = tan_alt.isel(mjd=im_lst[i]).data
    K = pathl1d_iris(h, z, z_top)    
    y = l1.isel(mjd=im_lst[i]).data    
#    Se = np.diag(error.data[i,:]**2)
    x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
    result_1d[i,:] = x
    Ave.append(A.sum(axis=1)) #sum over rows 

result_1d = xr.DataArray(result_1d, 
                         coords=(mjd[im_lst], z), 
                         dims=('mjd', 'z'))
result_1d.attrs['units'] = 'photons m-3 s-1 ?'
result_1d.attrs['long_name'] = '1d inversion VER'
Ave = np.array(Ave)
mr_threshold = 0.8
result_1d_mean = result_1d.where(Ave>mr_threshold).mean(dim='mjd')

# plot VER results
result_1d.plot(x='mjd', y='z',
#         norm=LogNorm(), 
         vmin=0, vmax=7.5e5, 
         size=3, aspect=3)
ax = plt.gca()
ax.set(title='1d retrieved VER',
      xlabel='tangent point along track distance from iris')
ax.set_xticks(mjd[im_lst[::label_interval]])
ax.set_xticklabels(np.round(tan_beta.sel(pixel=60).data[::label_interval]*Re))
plt.show()

plt.figure()
ax = plt.gca()
ax.plot(result_1d.T, z, '.')
result_1d_mean.plot(y='z', color='k',ls='-',
                    label='averaged profile with sum(A)>.{}'.format(mr_threshold))
ax.set_xscale('linear')
ax.set(#xlim=[1e4, 1e8],
       xlabel='volumn emission rate photons cm-3 s-1', 
       ylabel='altitdue grid',
       title='1d retrieval')
ax.legend()
plt.show()

# plot averaging kernel
plt.plot(Ave.T, z, '*')
plt.xlabel('Averaging kernel sum over rows')
plt.ylabel('altitude grid')
plt.title('Measurement response')
#plt.xlim([mr_threshold, 1.2])
plt.axvline(x=mr_threshold, ls=':', color='k')
#plt.text(mr_threshold, z[-1], 'threshold')
plt.show()
#%% Tomo grid
#====define atmosphere grid (the bin edges)
edges_alpha = np.linspace(tan_alpha.min()-0.01,
                          tan_alpha.max()+0.01, 2) #radian
edges_beta = np.arange(tan_beta.min()-0.1,
                         tan_beta.max()+0.15, 0.02) #radian (resolution 0.2 degree in Degenstein 2004)
edges_rho = np.append(z,z_top) # meter
edges = edges_alpha, edges_beta, edges_rho

#====grid points for plotting
grid_alpha = np.append(edges_alpha - np.gradient(edges_alpha)/2, 
                       edges_alpha[-1]+np.gradient(edges_alpha)[-1]/2)
grid_beta = np.append(edges_beta - np.gradient(edges_beta)/2, 
                       edges_beta[-1]+np.gradient(edges_beta)[-1]/2)
grid_rho = np.append(edges_rho - np.gradient(edges_rho)/2, 
                       edges_rho[-1]+np.gradient(edges_rho)[-1]/2)

#%% cal Jacobian
shape_tomo = (len(grid_alpha), len(grid_beta), len(grid_rho))
#====num of columns & rows of jacobian
col_len = len(grid_alpha) * len(grid_beta) * len(grid_rho)
row_len = l1.isel(mjd=im_lst, pixel=pix_lst).notnull().sum().item()

#====measure pathlength in each bin
from geometry_functions import los_points_fix_dl
from oem_functions import jacobian_row

dl = 3e3 #m fixed distance between all points, this unit will be the same as pathlength matrix
nop = 500 # choose number of points along the line
K_row_idx = []
K_col_idx = []
K_value = []
dll = dl * np.ones(nop) #temp
all_los_alpha, all_los_beta, all_los_rho = [], [], []
measurement_id = 0
for image in im_lst:
    #====generate points of los for all pixels in each image
    #====all points in cartesian coordinate relative to the space craft
    sc_look_new = change_of_basis(n_crosstrack, n_vel, n_zenith, 
                                  sc_look.isel(mjd=image).T)
    sc_pos_new = change_of_basis(n_crosstrack, n_vel, n_zenith, 
                                 sc_pos.isel(mjd=image).T)
    lx, ly, lz = los_points_fix_dl(sc_look_new, sc_pos_new, dl=dl, nop=nop)    
    #====convert xyz to alpha, beta, rho for all points
    los_alpha, los_beta, los_rho = cart2sphe(lx, ly, lz)
    all_los_alpha.append(los_alpha)
    all_los_beta.append(los_beta)
    all_los_rho.append(los_rho)
    
    #====build K
    for pix in l1.pixel.isel(pixel=pix_lst): 
        #print(image, pix.data, measurement_id)
        los = los_alpha.sel(pixel=pix), los_beta.sel(pixel=pix), los_rho.sel(pixel=pix)
        measurement_idx, grid_idx, pathlength = jacobian_row(dll, edges, los, measurement_id)
        K_row_idx.append(measurement_idx)
        K_col_idx.append(grid_idx)
        K_value.append(pathlength)
        measurement_id += 1
        
K_row_idx = np.concatenate(K_row_idx).astype('int')
K_col_idx = np.concatenate(K_col_idx).astype('int')
K_value = np.concatenate(K_value) *1e2 # meter --> cm

#==== create sparse matrix
from scipy.sparse import coo_matrix
K_coo = coo_matrix((K_value, (K_row_idx, K_col_idx)), shape = (row_len, col_len))

#%% Tomo inversion
from oem_functions import linear_oem_sp
import scipy.sparse as sp
y = l1.isel(mjd=im_lst, pixel=pix_lst).data.ravel()
y[y<0] = 0 #temp
#xa = np.ones(col_len) # temp
xa = interp1d(z, result_1d_mean.data, fill_value='extrapolate')(grid_rho)
xa = np.tile(xa, (len(grid_alpha),len(grid_beta),1)).ravel()
Sa = sp.diags([1], shape=(col_len, col_len)) * (2e4)**2 #temp
Se = sp.diags([1], shape=(measurement_id, measurement_id)) * (1e9)**2 #temp
x_hat, G = linear_oem_sp(K_coo, Se, Sa, y, xa)

shape_tomo = (len(grid_alpha), len(grid_beta), len(grid_rho))
result_tomo = x_hat.reshape(shape_tomo)
result_tomo = xr.DataArray(result_tomo, 
                           coords=(grid_alpha, grid_beta, grid_rho), 
                           dims=('alpha', 'beta', 'rho')) #temp

#====check residual
zoom = np.arange(1000)
plt.figure()
plt.plot(y[zoom], label='y')
plt.plot(K_coo.dot(x_hat)[zoom], label='K*x_hat')
plt.ylabel('signal')
plt.legend()
plt.show()


#====check residual
plt.figure()
plt.plot(K_coo.dot(x_hat)[zoom]-y[zoom])
plt.ylabel('residual')
plt.show()


#====contour plot 
plt.figure(figsize=(10,5))
ax = plt.gca()
plt.pcolor(grid_beta*Re, grid_rho*1e-3, result_tomo.isel(alpha=1).T)
ax.set(xlabel='distance along track in km',
       ylabel='altitude in km',
       title='slice along track')
plt.colorbar()
for i in range(0,len(im_lst),1):
    plt.axvline(x=tan_beta.sel(pixel=60)[i].data*Re, ymin=0.9)
plt.show()


#==== vertical profiles within the 3D matrix
plt.figure()
fig, ax = plt.subplots()
ax.set_xscale('linear')
ax.set(xlabel='VER',
      ylabel='altitude',
      title='tomography VER (all columns along track)')
for i in range(1,len(grid_alpha)-1):
    for j in range(1, len(grid_beta)-1):
        ax.plot(result_tomo[i,j,:], grid_rho*1e-3, '-')
plt.show()
# %%
