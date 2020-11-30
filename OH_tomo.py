#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d

#%%
ir = xr.open_dataset('ir_slc_003713_ch1.nc').sel(pixel=slice(21,128))

l1 = ir.data.where(ir.data.notnull(), drop=True).where(ir.sza>90, drop=True)
time = l1.time
error = ir.error.sel(time=time)
tan_alt = ir.altitude.sel(time=time)
tan_lat, *_ = xr.broadcast(ir.latitude.sel(time=time), tan_alt)
tan_lon, *_ = xr.broadcast(ir.longitude.sel(time=time), tan_alt)
sc_look = ir.look_ecef.sel(time=time)
sc_pos, *_ = xr.broadcast(ir.position_ecef.sel(time=time), sc_look)

# %% interpolation in altitude grid
# alts_interp = np.arange(20e3, 90e3, .5e3)
# data_interp = []
# for (data, alt) in zip(ir.data.where(ir.data.notnull(), drop=True).T, 
#                         ir.altitude.where(ir.data.notnull(), drop=True)):
#     f = interp1d(alt, data, bounds_error=False)
#     data_interp.append(f(alts_interp))
# data_interp = xr.DataArray(data_interp, coords=[ir.data.dropna('time', 'all').time, alts_interp],
#                            dims=[ir.time.name, 'z'])
# fig, ax = plt.subplots(2,1, sharex=True)
# data_interp.isel(time=slice(0,500)).plot(x=ir.time.name, norm=LogNorm(), vmin=1e10, vmax=5e11, ax=ax[0])
# ir.sza.plot(x=ir.time.name, ax=ax[1])

# %%
im_lst = range(0,100)
pixel_map = ((tan_alt>60e3)*(tan_alt<95e3))

#%% 1D inversion
%%time
from oem_functions import linear_oem
from geometry_functions import pathl1d_iris

z = np.arange(55e3, 95e3, 1e3) # m
z_top = z[-1] + 2e3
xa = np.ones(len(z)) * 0 # temp
Sa = np.diag(np.ones(len(z)))*(5e3)**2 #temp
# xa = result_1d_save.ver.mean(dim='time').values
# Sa = np.diag((0.2*xa)**2)
# xa_error2_save = result_1d_save.error2_retrieval
mr = []
error2_retrieval = []
ver = []
for i in range(len(im_lst)):
    isel_args = dict(time=im_lst[i])
    h = tan_alt.isel(**isel_args).where(pixel_map.isel(**isel_args), drop=True)
    K = pathl1d_iris(h.values, z, z_top)  *1e2 #m->cm
    y = l1.sel(pixel=h.pixel, time=h.time).values
    # Sa = np.diag(xa_error2_save.sel(time=h.time).values)
    Se = np.diag(error.sel(pixel=h.pixel, time=h.time).values**2)
    x, A, Ss, Sm = linear_oem(K, Se, Sa, y, xa)
    ver.append(x)
    mr.append(A.sum(axis=1)) #sum over rows 
    error2_retrieval.append(np.diag(Sm))

result_1d = xr.Dataset().update({
    'time': time[im_lst],
    'z': (['z',], z, {'units': 'm'}),
    'ver': (['time','z'], ver, {'long name': 'VER', 'units': 'photons cm-3 s-1'}),
    'mr': (['time','z'], mr),
    'error2_retrieval': (['time','z'], error2_retrieval),
    })

mr_threshold = 0.8
ver_1d_mean = result_1d.ver.where(result_1d.mr>mr_threshold).mean(dim='time')
error_1d_mean = result_1d.error2_retrieval.where(result_1d.mr>mr_threshold).mean('time')
#%%
# plot VER results
plt.figure()
result_1d.ver.plot.line(y='z', marker='.', ls='', add_legend=False)
ver_1d_mean.plot(y='z', color='k',ls='-',
                    label='averaged with MR>.{}'.format(mr_threshold))
(ver_1d_mean + np.sqrt(error_1d_mean)).plot(y='z', color='k', ls='--', label='averaged + error')
(ver_1d_mean - np.sqrt(error_1d_mean)).plot(y='z', color='k', ls='--', label='averaged - error')

plt.fill_betweenx(z, xa+np.sqrt(np.diag(Sa)), xa-np.sqrt(np.diag(Sa)), alpha=0.2, label='xa, Sa')
# plt.fill_betweenx(z, ver_1d_mean+np.sqrt(error_1d_mean), ver_1d_mean-np.sqrt(error_1d_mean), alpha=0.9, label='x_hat, Sm')

plt.gca().set(xlabel='VER / [photons cm-3 s-1]', 
    #    ylabel='Altitdue grid',
       title='1D retrieval of images {}'.format(im_lst))
plt.legend()
plt.show()

# AVK plot
A = xr.DataArray(A, coords=(z,z), dims=('row', 'col'), name='AVKs')
plt.figure()
A.plot.line(y='col', hue='row', add_legend=False)
A.sum('col').plot(y='row', color='k', label='MR')
# result_1d.mr.isel(time=-1).plot.line(y='z')
plt.gca().set(xlabel='',
            title='AVKs and MR')
plt.show()

# countour plot
fig, ax = plt.subplots(2,1,sharex=True, sharey=True)
plt_args = dict(x='time', y='z')
map_args = dict(cond=result_1d.mr>mr_threshold, drop=True)
result_1d.ver.where(**map_args).plot(ax=ax[0], vmin=0, vmax=7.5e3, **plt_args)
result_1d.error2_retrieval.where(**map_args).pipe(lambda x: 100*np.sqrt(x)/result_1d.ver
            ).rename('% error').plot(ax=ax[1], vmin=0, vmax=50, **plt_args)
plt.show()

#%%
# result_1d_save = result_1d.copy()

#%% change coordinate for Tomography
#====define the new base vectors
n_zenith = sc_pos.isel(time=0, pixel=60)
n_crosstrack = np.cross(sc_look.isel(time=0, pixel=60), n_zenith)
n_vel = np.cross(n_zenith, n_crosstrack)

#====tangent points in alpha, beta, rho coordinate
import pandas as pd
from geometry_functions import lla2ecef, cart2sphe, change_of_basis
tan_ecef = xr.concat(lla2ecef(tan_lat, tan_lon, tan_alt),  
                    pd.Index(['x','y','z'], name='xyz'))

tan_alpha, tan_beta, tan_rho = [], [], []
for im in im_lst:
    p_old = tan_ecef.isel(time=im)
    p_new = change_of_basis(n_crosstrack, n_vel, n_zenith, p_old)
    alpha, beta, rho = cart2sphe(p_new.sel(xyz='x'), p_new.sel(xyz='y'), p_new.sel(xyz='z'))
    tan_alpha.append(alpha)
    tan_beta.append(beta)
    tan_rho.append(rho)
kwargs = dict(coords=[time[im_lst], ir.pixel], dims=['time', 'pixel'])
tan_alpha, tan_beta, tan_rho = (xr.DataArray(tan_alpha, **kwargs),
                                xr.DataArray(tan_beta, **kwargs),
                                xr.DataArray(tan_rho, **kwargs))
Re = 6371 + 100 #Earth radius in km

#%% plot tangent points in alpha-beta-rho space
plt.figure()
plt.plot(im_lst, tan_alpha.isel(pixel=60)*180/np.pi,label='cross')
plt.plot(im_lst, tan_beta.isel(pixel=60)*180/np.pi, label='along')

plt.plot(tan_lat.isel(pixel=60), '--', label='lat')
plt.plot(tan_lon.isel(pixel=60), '--', label='lon')
plt.legend()


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
%%time
shape_tomo = (len(grid_alpha), len(grid_beta), len(grid_rho))
#====num of columns & rows of jacobian
col_len = len(grid_alpha) * len(grid_beta) * len(grid_rho)
row_len = pixel_map.isel(time=im_lst).sum().item()

#====measure pathlength in each bin
from geometry_functions import los_points_fix_dl
from oem_functions import jacobian_row

d_start = 1730e3
nop = 500 # choose number of points along the line
dl = 3e3 #m fixed distance between all points, this unit will be the same as pathlength matrix
dll = dl * np.ones(nop) #temp
K_row_idx, K_col_idx, K_value = [], [], []
all_los_alpha, all_los_beta, all_los_rho = [], [], []
measurement_id = 0
for im in im_lst:
    #====generate points of los for all pixels in each image
    #====all points in cartesian coordinate relative to the space craft as origin
    sc_look_new = change_of_basis(n_crosstrack, n_vel, n_zenith, sc_look.isel(time=im).T)
    sc_pos_new = change_of_basis(n_crosstrack, n_vel, n_zenith, sc_pos.isel(time=im, pixel=60).T)
    lx, ly, lz = los_points_fix_dl(sc_look_new, sc_pos_new, dl=dl, nop=nop)    
    #====convert xyz to alpha, beta, rho for all points
    los_alpha, los_beta, los_rho = cart2sphe(lx, ly, lz)
    all_los_alpha.append(los_alpha)
    all_los_beta.append(los_beta)
    all_los_rho.append(los_rho)
    
    #====build K
    for pix in l1.pixel.where(pixel_map.isel(time=im_lst).isel(time=im), drop=True): 
        # print(measurement_id)
        los = los_alpha.sel(pixel=pix), los_beta.sel(pixel=pix), los_rho.sel(pixel=pix)
        measurement_idx, grid_idx, pathlength = jacobian_row(dll, edges, los, measurement_id)
        K_row_idx.append(measurement_idx)
        K_col_idx.append(grid_idx)
        K_value.append(pathlength)
        measurement_id += 1

# sc_look_flat = sc_look.isel(time=im_lst).stack(meas_id=['time', 'pixel'])
# sc_pos_flat = sc_pos.isel(time=im_lst).stack(meas_id=['time', 'pixel'])
# for measurement_id in range(row_len):
#     print(measurement_id)
#     #====all points in cartesian coordinate relative to the space craft as origin
#     sc_look_new = change_of_basis(n_crosstrack, n_vel, n_zenith, sc_look_flat.isel(meas_id=measurement_id))
#     sc_pos_new = change_of_basis(n_crosstrack, n_vel, n_zenith, sc_pos_flat.isel(meas_id=measurement_id))
#     #====generate points of los and convert xyz (ecef) to alpha, beta, rho coordinates
#     los_xyz = []
#     for i in range(nop):
#         los_xyz.append(sc_pos_new + (i+d_start/dl)*sc_look_new*dl)
#     los_xyz = xr.concat(los_xyz, dim='nop').assign_coords(nop=np.arange(nop))
#     #====convert xyz to alpha, beta, rho for all points
#     los_abr = cart2sphe(*los_xyz.transpose('xyz', 'nop'))
#     #====row, column indeces and value for building Jacobian
#     measurement_idx, grid_idx, pathlength = jacobian_row(dll, edges, los_abr, measurement_id)
#     K_row_idx.append(measurement_idx)
#     K_col_idx.append(grid_idx)
#     K_value.append(pathlength)

K_row_idx = np.concatenate(K_row_idx).astype('int')
K_col_idx = np.concatenate(K_col_idx).astype('int')
K_value = np.concatenate(K_value) *1e2 # meter --> cm
#==== create sparse matrix
from scipy.sparse import coo_matrix
K_coo = coo_matrix((K_value, (K_row_idx, K_col_idx)), shape = (row_len, col_len))

#%% Tomo inversion
%%time
from oem_functions import linear_oem_sp
import scipy.sparse as sp
# y = np.ravel(l1.isel(time=im_lst))[np.where(np.ravel(pixel_map.isel(time=im_lst)))] # remove all nan
y = l1.isel(time=im_lst).stack(meas_id=['time', 'pixel']).where(
    pixel_map.isel(time=im_lst).stack(meas_id=['time', 'pixel']), drop=True).values
# y[y<0] = 0 #temp
# xa = np.ones(col_len) *0 # temp
xa = ver_1d_mean.fillna(0).interp(z=grid_rho, kwargs=dict(fill_value='extrapolate'))
xa = np.tile(xa, (len(grid_alpha),len(grid_beta),1)).ravel()
sigma2 = error_1d_mean.mean().values
Sa = sp.diags(sigma2 * np.ones(col_len))
# Sa = sp.diags([1], shape=(col_len, col_len)) * 2e6 #(5e3)**2 #temp 1e10?
Se = sp.diags(error.isel(time=im_lst).stack(meas_id=['time', 'pixel']).where(
    pixel_map.isel(time=im_lst).stack(meas_id=['time', 'pixel']), drop=True).values**2)
x_hat, G = linear_oem_sp(K_coo, Se, Sa, y, xa)
A = G.dot(K_coo)
mr = A.sum(axis=1)
Sm = G.dot(Se).dot(G.T) #retrieval noise

shape_tomo = (len(grid_alpha), len(grid_beta), len(grid_rho))
ver = x_hat.reshape(shape_tomo)
mr = np.array(mr).reshape(shape_tomo)
error2_retrieval = Sm.diagonal().reshape(shape_tomo)
result_tomo = xr.Dataset({
    'alpha': grid_alpha,
    'beta': grid_beta,
    'rho': grid_rho,
    'ver': (['alpha','beta', 'rho'], ver),
    'mr': (['alpha','beta', 'rho'], mr),
    'error2_retrieval': (['alpha','beta', 'rho'], error2_retrieval),
    })

#%% contour plot 
fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
result_tomo.ver.isel(alpha=1
    ).assign_coords(beta=result_tomo.beta*Re, rho=result_tomo.rho*1e-3
    ).plot(ax=ax[0], x='beta', vmin=0, vmax=7.5e3, cmap='viridis')

CS = result_tomo.mr.isel(alpha=1
    ).assign_coords(beta=result_tomo.beta*Re, rho=result_tomo.rho*1e-3
    ).plot.contour(ax=ax[0], x='beta', levels=[0, 0.8, 1, 1.5], cmap='gray_r')
ax[0].clabel(CS, CS.levels, inline=True, fmt='%r', fontsize=10)

for i in range(0,len(im_lst),1):
    ax[0].axvline(x = tan_beta.sel(pixel=60)[i].values * Re, ymax=0.1)

result_tomo.error2_retrieval.pipe(lambda x: 100*np.sqrt(x)/result_tomo.ver
    ).isel(alpha=1
    ).rename('% error'
    ).assign_coords(beta=result_tomo.beta*Re, rho=result_tomo.rho*1e-3
    ).plot(ax=ax[1], x='beta', vmin=-50, vmax=50, cmap='viridis')

ax[0].set(title='Middle slice along track',
        xlabel='',
        ylabel='z / km')
ax[1].set(title='',
        xlabel='Distance along track / km',
        ylabel='z / km')
plt.show()
#%% vertical profiles 
plt.figure()
result_tomo.ver.where(result_tomo.mr>mr_threshold).isel(alpha=1
    ).assign_coords(beta=result_tomo.beta*Re, rho=result_tomo.rho*1e-3
    ).plot.line(y='rho', marker='.', add_legend=False)

ver_tomo_mean = result_tomo.ver.where(result_tomo.mr>mr_threshold).mean(dim=['alpha', 'beta'])
ver_tomo_mean.assign_coords(rho=result_tomo.rho*1e-3
    ).plot(y='rho', color='k', label='averaged with mr>{}'.format(mr_threshold))
error_tomo_mean = result_tomo.error2_retrieval.where(result_tomo.mr>mr_threshold).mean(dim=['alpha', 'beta'])
(ver_tomo_mean+np.sqrt(error_tomo_mean)).assign_coords(rho=result_tomo.rho*1e-3
    ).plot(y='rho', color='k', ls='--', label='averaged + error')
(ver_tomo_mean-np.sqrt(error_tomo_mean)).assign_coords(rho=result_tomo.rho*1e-3
    ).plot(y='rho', color='k', ls='--', label='averaged - error')
     
plt.fill_betweenx(grid_rho*1e-3, xa[:len(grid_rho)]+np.sqrt(sigma2), xa[:len(grid_rho)]-np.sqrt(sigma2),
    label='xa, Sa', alpha=0.2)
plt.legend()
plt.gca().set(xlabel='VER / [photons cm-3 s-1]', 
       ylabel='z / km',
       title='Tomo retrieval of images {}'.format(im_lst))
plt.show()

#%%
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

# %%
