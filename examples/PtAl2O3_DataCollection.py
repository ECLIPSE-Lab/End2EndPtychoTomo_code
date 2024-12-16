import taichi

taichi.init(arch=taichi.gpu)

from scatterem.operators.basic import Propagator
from scatterem.operators.loss import amplitude_loss
from scatterem import ReconstructionOptions
from scatterem.models import MultiSlicePtychographyTomographyModel
from scatterem.util.fourd import ComplexProbe
from scatterem.util.base import advanced_raster_scan, init_gauss_blur_regularization, regularize_multilayers, \
    add_modes_cartesian_hermite, energy_to_sigma
import numpy as np
import torch as th
from scatterem.util.base import gaussian_blur_regularization, energy_to_wavelength as wavelength
from scatterem.operators.split import grid_sample_partial
import matplotlib.pyplot as plt
import h5py

device = 'cuda:0'

# %%
E = 200e3
potvol_ori = np.load('/home/shengbo/PycharmProjects/nano_particles/PtAl2O3_333nm_edge_RawVol_10FP.npy') * energy_to_sigma(E)
potvol_ori = np.float32(potvol_ori)


gauss_kernel_size = 5
gauss_sigma = 3
gauss_kernel = init_gauss_blur_regularization(gauss_kernel_size, gauss_sigma)

potvol = []
for poti in range(potvol_ori.shape[0]):
    volpre = th.as_tensor(potvol_ori[poti]).unsqueeze(0).unsqueeze(0)
    volpre = gaussian_blur_regularization(volpre, gauss_kernel).squeeze()
    vol_pad = np.zeros([644, 644, 644])
    vol_pad[:, :, :] = volpre[:, :, 1:645]
    vol_pad = np.array(vol_pad).squeeze()
    vol = np.float32(np.swapaxes(vol_pad, 1, 2))
    potvol.append(vol)

# %%
ang = np.array([-45, -41.5, -38, -34.5, -31, -28, -25, -22, -19, -15.5, -12, -8.5, -5, 0, 5, 8.5, 12, 15.5, 19, 22, 25,
                28, 31, 34.5, 38, 41.5, 45])
phi_deg = ang
psi_deg = np.zeros_like(phi_deg)
theta_deg = np.zeros_like(psi_deg)

n_angles = ang.shape[0]
phi = th.as_tensor(np.deg2rad(phi_deg), dtype=th.float32)
psi = th.as_tensor(np.deg2rad(psi_deg), dtype=th.float32)
theta = th.as_tensor(np.deg2rad(theta_deg), dtype=th.float32)

angles = th.zeros((3, n_angles))
translation = th.zeros((2, 1), dtype=th.float32)
translations = th.zeros((2, 1))

np.save(f'/home/shengbo/PycharmProjects/nano_particles/FP10_3nm_phi.npy', phi)
np.save(f'/home/shengbo/PycharmProjects/nano_particles/FP10_3nm_psi.npy', psi)
np.save(f'/home/shengbo/PycharmProjects/nano_particles/FP10_3nm_theta.npy', theta)

print(f'angles: {np.rad2deg(phi)}')
# %%
alpha = 25
defval = -60
_polar_parameterstarget = {'C10': defval}

dx = 0.05
E = 200e3
scan_physical = 0.3
scan_step = scan_physical/dx

M1 = 128

M3 = np.ceil(vol.shape[0]).astype(int)
M3z = np.ceil(vol.shape[1]).astype(int)

M3_step = np.int64(vol.shape[0]/scan_step)
M3z_step = np.int64(vol.shape[1]/scan_step)

padding = 8

V_shape1D = M1 + M3 + padding
V_shapez = M1 + M3z + padding
bin_factor = 780

V_shape = (V_shapez, V_shapez, V_shapez)
start_end = [[[0, V_shape[0]], [0, V_shape[2]]]] * n_angles

V_model_ori = np.zeros(V_shape)
volfit = np.int64((V_shapez - vol.shape)/2)
V_model_ori[volfit[0]:-volfit[0], volfit[1]:-volfit[1], volfit[2]:-volfit[2]] = vol

V_model_pot = np.exp(1j * V_model_ori)
V_target = th.tensor(V_model_pot).unsqueeze(0).to(th.complex64)
print('V_model shape', V_model_pot.shape)


probe_target0 = []
for i in range(n_angles):
    probe_target = ComplexProbe(gpts=np.array([M1, M1]), sampling=[0.2, 0.2], energy=E, semiangle_cutoff=alpha, rolloff=2.0,
                        vacuum_probe_intensity=None, parameters=_polar_parameterstarget, device="cpu").build()
    probe_target = np.fft.fftshift(probe_target._array.astype(np.complex64))
    probe_modes = add_modes_cartesian_hermite(probe_target[None, None], 3)
    psi_target = th.as_tensor(probe_modes)
    probe_target0.append([psi_target])


print('probe shape', probe_target0[0][0].shape)

# %%
positions = []
num_steps = np.array([M3_step, M3_step])
positions_float = advanced_raster_scan(ny=num_steps[0], nx=num_steps[1], dy=scan_step, dx=scan_step, theta=0)
positions_round = np.array(positions_float)
positions_round += volfit[0]-M1/2
print(positions_round.min(), positions_round.max() + M1)
positions_temp = th.tensor(positions_round).to(th.float32)
for i in range(n_angles):
    positions.append(positions_temp)

dr = [th.zeros_like(positions[0])] * n_angles
# %%
slice_thickness = bin_factor * dx
E = 200e3
lam = wavelength(E)
dxx = [dx, dx]
shape = (M1, M1)
propagator = Propagator(slice_thickness, lam, dxx, shape)
# %%
from numpy.fft import fft, ifft, fftshift

kernel_num = 297
psf_shape = 297
kernel_shape = [kernel_num, kernel_num, kernel_num]
psf = np.zeros([psf_shape, psf_shape, psf_shape])
psf[0, 0, 0] = 1

Wa = regularize_multilayers(kernel_shape, 90, alpha=1)
psf2 = np.fft.ifftn(np.fft.fftn(psf) * Wa).real
m0 = 146
m1 = 112
m2 = 146
psf3 = fftshift(psf2)[m2:-m2, m0:-m0, m1:-m1]
psfimg = fftshift(psf2[0])[m0:-m0, m1:-m1]
kernel = th.as_tensor(np.swapaxes(psf3, 1, 2)).unsqueeze(0)

gauss_kernel_size = 3
gauss_sigma = 1
gauss_kernel = init_gauss_blur_regularization(gauss_kernel_size, gauss_sigma)
# %%

tau1 = 1e-9
tau2 = 3e-9
braggpix = 240
butterorder = 5
error_norm = []
# %%
options = ReconstructionOptions()

model = MultiSlicePtychographyTomographyModel(V_target,
                                              probe_target0,
                                              positions,
                                              dr,
                                              error_norm,
                                              angles=angles,
                                              tau1=tau1,
                                              tau2=tau2,
                                              translations=translations,
                                              start_end_model=start_end,
                                              loss_function=amplitude_loss,
                                              propagator=propagator,
                                              braggpix=braggpix,
                                              butterorder=butterorder,
                                              dx=dx,
                                              bin_factor=bin_factor,
                                              intensity_scaling_factor=1,
                                              kernel=kernel,
                                              subpix_shift=False)

del V_target
del probe_target0
del positions

a_target_list = []
a_target_fp_list = []
dose_target = 1.7e4
for ai in range(ang.shape[0]):
    for pi in range(len(potvol)):

        phi_temp = th.as_tensor([phi[ai]]).to(device)
        psi_temp = th.as_tensor([psi[ai]]).to(device)
        theta_temp = th.as_tensor([theta[ai]]).to(device)
        translation_temp = th.tensor(translation.to(device))

        V_model_ori = np.float32(np.zeros(V_shape))
        V_model_ori[volfit[0]:-volfit[0], volfit[1]:-volfit[1], volfit[2]:-volfit[2]] = potvol[pi]
        vol_ori = th.as_tensor(V_model_ori, dtype=th.float32).unsqueeze(0).unsqueeze(0).to(device)
        v_rot = grid_sample_partial(vol_ori, phi_temp, theta_temp, psi_temp, translation_temp, start_end[0])
        v_rot_np = v_rot.cpu().numpy()
        # np.save(f'/home/shengbo/PycharmProjects/nano_particles/vol_rotation_{ang[0]}.npy', v_rot)
        V_model = np.exp(1j * v_rot_np)
        # V_target = th.tensor(V_model).unsqueeze(0).to(th.complex64).to(device)
        V_target = th.tensor(V_model).unsqueeze(0).to(th.complex64)

        V_target.requires_grad = True
        a_target = model(V_target, psi_target, positions_temp, dr, angle=angles[:, 0], translation=translations,
                     bin_factor=bin_factor, start_end=start_end[0], propagator=propagator, factor=1)
        aa64 = a_target.clone().detach().cpu().numpy()
        aa = np.float32(aa64)
        # dpshift = np.fft.fftshift(aa, (2, 3))
        del a_target
        del V_target
        del V_model_ori
        del vol_ori
        del v_rot
        del V_model
        del v_rot_np
        del aa64
        # aa = aa * 30
        dose_ori = np.sum((aa ** 2).mean(0)) / (scan_physical ** 2)
        aa = aa * np.sqrt(dose_target/dose_ori)
        aa = np.array(th.poisson(th.as_tensor(aa)))
        a_target_fp_list.append(aa)
        print('number of Frozen Phonon:', pi)
        th.cuda.empty_cache()

    a_target_fp_mean = np.float32(np.mean(a_target_fp_list, axis=0))
    a_target_list.append(a_target_fp_mean)
    a_target_fp_list = []
    print('Number of angle:', ai)
    np.save(f'/home/shengbo/PycharmProjects/nano_particles/PtAl2O3_DP_4nm_edge_angle_{ang[ai]}_test.npy', a_target_fp_mean)
# %%

dp = aa
fig, ax = plt.subplots()
imax = ax.imshow(np.mean(np.fft.fftshift(dp), axis=0))
plt.colorbar(imax)
plt.show()
# %%
# aa = a_target.clone().detach().cpu().numpy()
# np.save(f'/home/shengbo/PycharmProjects/nano_particles/test_{ang[0]}.npy', aa)
np.save(f'/home/shengbo/PycharmProjects/nano_particles/PtAl2O3_DP_3nm_edge_{n_angles}angles_noise_test.npy', a_target_list)


