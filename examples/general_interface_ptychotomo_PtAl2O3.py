import taichi

taichi.init(arch=taichi.gpu)

from scatterem.operators.basic import Propagator
from scatterem.operators.loss import amplitude_loss
from scatterem import ReconstructionInputs, ReconstructionOptions, reconstruct
from scatterem.models import MultiSlicePtychographyTomographyModel
from scatterem.kernels import update_object_2D
from scatterem.data import RasterScanningDiffractionDataset
from scatterem.util.fourd import ComplexProbe
from scatterem.util.base import advanced_raster_scan, init_gauss_blur_regularization, regularize_multilayers, \
    add_modes_cartesian_hermite
import numpy as np
import torch as th
from scatterem.util.base import energy_to_wavelength as wavelength
import matplotlib.pyplot as plt
import h5py

device = 'cuda:0'

# %% Load the diffraction patterns
error_norm = []
measurements_list_ori = []
out_path = '/home/shengbo/PycharmProjects/nano_particles/'
data_array = np.load('/home/shengbo/PycharmProjects/nano_particles/PtAl2O3_DP_3nm_edge_27angles_noise.npy')
raw_shape = [1, 644, 644, 644]
sind1 = 0
sind2 = 27
for di in range(data_array.shape[0]):
    if di >= sind1 and di < sind2:
        data_ori = data_array[di]
        measurements_list_ori.append(data_ori)
        error_norm.append(th.tensor(np.sum(data_ori)).to(th.float32).to(device))

measurements_list = []
for mi in measurements_list_ori:
    data = th.as_tensor(mi)
    measurements_list.append(data)
# %% Load the rotation angles
n_angles = len(measurements_list_ori)
angles = th.zeros((3, n_angles), device=device)
translations = [th.zeros((2, 1), device=device)] * n_angles

path_angles = '/home/shengbo/PycharmProjects/nano_particles/'
psi = np.load(path_angles + 'FP10_3nm_psi.npy')[sind1:sind2]
theta = np.load(path_angles + 'FP10_3nm_theta.npy')[sind1:sind2]
phi = np.load(path_angles + 'FP10_3nm_phi.npy')[sind1:sind2]

angles[0, :] = th.as_tensor(phi).to(device)

print(f'angles: {np.rad2deg(phi)}')
# %% Load the probe and create the initial guess of the volume
M1 = data.shape[-1]
M3 = np.sqrt(data.shape[0]).astype(int)
alpha = 25
defval = -60
_polar_parameterstarget = {'C10': defval}

collection_dx = 0.05
dx = 0.2
E = 200e3
scan_physical = 0.3
scan_step = scan_physical/dx
M3_step = np.ceil(scan_step * M3).astype(int)

padding = 1
volz_shape = np.int64(raw_shape[3]/(dx/collection_dx))
V_shape1D = M1 + M3_step + padding

bin_factor = 29

V_shape = (V_shape1D, V_shape1D, V_shape1D)
# V_shape = (V_shapez, V_shapez, V_shapez)
V_model = th.zeros(V_shape, dtype=th.complex64).unsqueeze(0).to(device)
print('V_model shape', V_model.shape)

probe_model = []
for i in range(len(measurements_list_ori)):
    probe_target = ComplexProbe(gpts=np.array([M1, M1]), sampling=[0.2, 0.2], energy=E, semiangle_cutoff=alpha, rolloff=2.0,
                        vacuum_probe_intensity=None, parameters=_polar_parameterstarget, device="cpu").build()
    probe_target = np.fft.fftshift(probe_target._array.astype(np.complex64))
    probe_modes = add_modes_cartesian_hermite(probe_target[None, None], 3)

    dp_int = np.sum((measurements_list_ori[i] ** 2).mean(0))
    probe_int = np.sum(np.abs(probe_modes) ** 2)
    probe_modes = probe_modes * np.sqrt(dp_int / probe_int)
    probe_temp = [th.as_tensor(probe_modes, device=device)]
    # psi_target = th.as_tensor(probe_target[None, None], device=device)
    probe_model.append(probe_temp)


print('probe shape', probe_model[0][0].shape)
# %% Define the positions
positions = []
num_steps = np.array([M3, M3])
positions_float = advanced_raster_scan(ny=num_steps[0], nx=num_steps[1], dy=scan_step, dx=scan_step, theta=0)
positions_round = np.array(positions_float)
positions_round += padding/2
print(positions_round.min(), positions_round.max() + M1)
positions_original = th.tensor(positions_round).to(th.float32).to(device)
scale_factor = V_shape1D/2
translations = translations * scale_factor


for i in range(len(measurements_list_ori)):
    positions_temp = positions_original.clone()
    difx = translations[1, i]
    dify = translations[0, i]
    positions_temp[0] -= difx
    positions_temp[1] -= dify
    positions.append(positions_temp)

dr = [th.zeros_like(positions[0])] * n_angles
start_end = [[[0, V_shape[0]], [0, V_shape[2]]]] * n_angles
# %% Define the multi-slice propagator
slice_thickness = bin_factor * dx
E = 200e3
lam = wavelength(E)
dxx = [dx, dx]
shape = (M1, M1)
propagator = Propagator(slice_thickness, lam, dxx, shape, device=device)
# %% Define the PSF kernel and Gaussian kernel
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
kernel = th.as_tensor(np.swapaxes(psf3, 1, 2)).unsqueeze(0).to(device)

gauss_kernel_size = 3
gauss_sigma = 1
gauss_kernel = init_gauss_blur_regularization(gauss_kernel_size, gauss_sigma)
# %%
num_ite = 500
f = 4
N_batches = f
batch_size = data.shape[0] // f

tau1 = 1e-7
tau2 = 3e-7
braggpix = 90
butterorder = 5
# %%
del measurements_list_ori
options = ReconstructionOptions()


model = MultiSlicePtychographyTomographyModel(V_model,
                                              probe_model,
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
# %%
datasets = []
for i, m in enumerate(measurements_list):
    datasets += [
        RasterScanningDiffractionDataset(m, probe_index=0, angles_index=i, translation_index=i, start_end_index=i)]

# %%
options = ReconstructionOptions()
options.optimize_probe = lambda it: it > -1
options.optimize_sample = lambda it: it > -1
options.num_iterations = num_ite
options.update_object_function = update_object_2D
options.probe_optimizer_parameters = {'lr': 6e-5, 'momentum': 0.4}
options.object_optimizer_parameters = {'lr': 4e-5, 'momentum': 0.5}
options.batch_size = batch_size

options.after_iteration_hooks = []
options.after_batch_hooks = []
options.after_dataset_hooks = []

inputs = ReconstructionInputs(
    model=model,
    datasets=datasets,
)
results = reconstruct(inputs, options)
# %%

t1 = results.V[0].real
# %%
error_norm_total = th.sum(th.stack(error_norm)).clone().cpu().numpy()
losses_np = np.array(results.losses / error_norm_total)
plt.plot(losses_np[:])
plt.show()
# %%
t2 = np.sum(t1, axis=0)
fig, ax = plt.subplots()
imax = ax.imshow(t2)
plt.colorbar(imax)
plt.show()

# %%
out_path = '/home/shengbo/PycharmProjects/nano_particles/'
prism_output_name = f'PtAl2O3_3nm_500ite_41deg.h5'
with h5py.File(out_path + prism_output_name, 'w') as f:
    dset = f.create_dataset('volume_data', data=t1)
    dset.attrs['description'] = 'This is 3D dataset'
    dset.attrs['unit'] = 'arbitrary unit'

print("HDF5 file created successfully.")

