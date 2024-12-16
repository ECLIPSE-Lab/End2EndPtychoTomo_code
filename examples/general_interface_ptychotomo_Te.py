import taichi

taichi.init(arch=taichi.gpu)

from scatterem.operators.basic import Propagator
from scatterem.operators.loss import amplitude_loss
from scatterem import ReconstructionInputs, ReconstructionOptions, reconstruct
from scatterem.models import MultiSlicePtychographyTomographyModel
from scatterem.kernels import update_object_2D
from scatterem.data import RasterScanningDiffractionDataset
from scatterem.util.base import regularize_multilayers
import numpy as np
import torch as th
from scatterem.util.base import energy_to_wavelength as wavelength
import matplotlib.pyplot as plt
from glob import glob
import zarr
import h5py
from numpy.fft import fft, ifft, fftshift


device = 'cuda:0'

# %% Define the path for data, alignment, probe and scan positions

# Download all raw data from https://zenodo.org/records/13060513 and put them in the data folder
# Download all MSP reconstructed results in results.tar.gz from https://zenodo.org/records/14499409 and put them in the results folder
# Download all alignment results from https://zenodo.org/records/14499409 and put them in this folder

base_path = '/home/shengbo/PycharmProjects/te_particle/'
raw_data_path = base_path + 'data/'
probe_position_path = base_path + 'results/'

path_angles = '/home/shengbo/FAUbox/Pelz Lab (Philipp Pelz)/shengbo/te_particle/'
# %% Load the 4D-STEM diffraction patterns and convert the intensity to modulus
M1 = 66
margin = 5
scan_shape = [220, 220]
V_shape1D = scan_shape[0] + M1 + margin * 2

take = [171, 173, 175, 177, 179, 183, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 214, 216, 218, 220, 222,
        224, 226, 228, 230, 232]

error_norm_list = []
measurements_list = []
metadata_list = []
data_files = sorted(glob(raw_data_path + '/*.zip'))
print(data_files)
for i, df in enumerate(data_files):
    # print(i, len(data_files))
    number = int(df.split('/')[-1].split('.')[0])
    if number not in [204] and number in take:
        store2 = zarr.open(df, mode='r')
        data = store2['/data'][:, :, :, :]
        ds = np.array(data.shape)
        meta_dict = store2['/meta'][0]
        print(number)
        da = np.sqrt(np.fft.fftshift(data, (2, 3)))
        error_norm_list.append(th.tensor(np.sum(da)).to(th.float32).to(device))
        measurements_list.append(da)
        metadata_list.append(meta_dict)

# %% Load the affine alignment results from the step 2

psi = np.load(path_angles + 'psi_astra.npy')
theta = np.load(path_angles + 'theta_astra.npy')
phi = np.load(path_angles + 'phi_astra.npy')
translation_pix = np.load(path_angles + 'translation_astra.npy').T
translation = 2 * translation_pix/V_shape1D

fig, ax = plt.subplots(1, 3)
ax[0].scatter(np.arange(len(psi)), psi)
ax[1].scatter(np.arange(len(psi)), phi)
ax[2].scatter(np.arange(len(psi)), theta)
plt.show()

# %% Calculate the real space voxel size

k_max = meta_dict['k_max'][0]
dx = 1 / 2 / k_max
dx
# %% Load the probe and positions

select_inds = {
    171: 3,
    173: 5,
    175: 5,
    177: 1,
    179: 5,
    183: 4,
    184: 2,
    186: 4,
    188: 4,
    190: 3,
    192: 6,
    194: 5,
    196: 3,
    198: 6,
    200: 5,
    202: 6,
    204: 5,
    214: 1,
    216: 6,
    218: 0,
    220: 0,
    222: 6,
    224: 6,
    226: 0,
    228: 3,
    230: 2,
    232: 3
}

probes_list = []
positions_list = []
dr_list = []
probes_files = sorted(glob(probe_position_path + '/*.h5'))
for i, df in enumerate(probes_files):
    number = int(df.split('/')[-1].split('.')[0])
    with h5py.File(df, 'r') as f:
        # probe_ori = f['probe'][select_inds[number]][:, :, :]
        probe = f['probe'][select_inds[number]][:, :, :]
        # probe_th = th.from_numpy(probe).unsqueeze(1)
        # probe = th.detach(probe_th).cpu().numpy
        # probe = np.expand_dims(probe_ori, axis=1)
        positions = f['best_positions'][select_inds[number]][:, :]
        positions_float = positions / dx + (scan_shape[0] / 2) + margin
        positions_round = np.round(positions_float).astype(np.int64)
        dr = positions_round - positions_float
        if probe.shape[1] == 66:
            probes_list.append(probe)
            pt = positions_round.T
            pa = pt[:, 0]
            pb = pt[:, 1]
            pc = np.zeros_like(pt)
            pc[:, 0] = pb
            pc[:, 1] = pa
            positions_list.append(pc)
            dr_list.append(th.tensor(dr.T).to(th.float32))
            print(number, probe.shape, positions.shape)

ind = 5
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(positions_list[ind][:, 0], positions_list[ind][:, 1], marker='x', s=1)
plt.show()
# %% Scale the initial guess of the probe to have the same intensity scale with diffraction patterns
ind = 0
probes_scaled_list = []

for ind in range(len(probes_list)):
    probes = probes_list[ind]
    probe_int = np.sum(np.abs(probes) ** 2)
    print(f'{ind} probe_int: {probe_int}')
    mean_int = np.sum((measurements_list[ind] ** 2).mean((0, 1)))
    probes_scaled = probes * np.sqrt(mean_int / probe_int)
    print(f'{ind} mean_int: {mean_int}')
    probes_scaled_int = np.sum(np.abs(probes_scaled) ** 2)
    print(f'{ind} probes_scaled_int: {probes_scaled_int}')
    probes_scaled_list.append(probes_scaled)
    print()

probes_list = th.as_tensor(probes_scaled_list)

# %% Check data, probes and positions are having the same number of set
len(measurements_list), len(probes_list), len(positions_list)

# %% Check the positions are properly padded
for posi in positions_list:
    print(posi.min(0), posi.max(0))

# %% Create the initial guess of the potential volume
V_shape = (V_shape1D, V_shape1D, V_shape1D)
V_model = th.zeros(V_shape, dtype=th.complex64).unsqueeze(0).to(device)

# %% Determine the number of slices in multi-slice calculation
slice_number = 8
bin_factor = V_shape1D // slice_number
slice_thickness = bin_factor * dx
print('slice_thickness', slice_thickness)
E = 80e3
lam = wavelength(E)
dx = [dx, dx]
shape = (M1, M1)

# %%
Nmodes = probes_list[0].shape[0]
B = 1
N = th.tensor([V_shape[0], V_shape[2]]).int()
M = th.tensor([M1, M1]).int()
K = positions_list[0].shape[0]
print('Nmodes:', Nmodes)
print('B: ', B)
print('N: ', N)
print('M: ', M)
print('K: ', K)

fig, ax = plt.subplots()
imax = ax.imshow(measurements_list[13].mean((0, 1)))
plt.colorbar(imax)
plt.show()
# %% Reshape the diffraction patterns
measurements_list2 = []
for mi in measurements_list:
    ms = mi.shape
    w = th.as_tensor(mi).view((ms[0] * ms[1], ms[2], ms[3]))
    measurements_list2.append(w)
measurements_list = measurements_list2

# %% Create the Point Spread Function to regularize the binning effect
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
fig, ax = plt.subplots()
imax = ax.imshow(psfimg)
plt.colorbar(imax)
plt.show()
kernel = th.as_tensor(np.swapaxes(psf3, 1, 2)).unsqueeze(0).to(device)
# %% Upload the variables to GPU
n_angles = len(phi)
angles = th.zeros((3, n_angles), device=device)
angles[0, :] = th.as_tensor(phi).to(device)
angles[1, :] = th.as_tensor(theta).to(device)
angles[2, :] = th.as_tensor(psi).to(device)
start_end = [[[0, V_shape[0]], [0, V_shape[2]]]] * n_angles
translations = [th.tensor([[ti[0]], [ti[1]]], device=device) for ti in translation]
positions = [th.as_tensor(posi).to(device) for posi in positions_list]
dr = dr_list
error_norm = error_norm_list
probe_model = [[th.as_tensor(probe_i).unsqueeze(1).to(device).requires_grad_(False)] for probe_i in probes_list]
propagator = Propagator(slice_thickness, lam, dx, shape, device=device)

error_norm_total = th.sum(th.stack(error_norm)).clone().cpu().numpy()
tau1 = 1e-9
tau2 = 3e-9
braggpix = 113
butterorder = 5
# %% Prepare the reconstruction model
num_ite = 300
f = 5
N_batches = f
batch_size = w.shape[0] // f

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
                                              dx=dx[0],
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

# options.upper_limit = 1.3

options.after_iteration_hooks = []
options.after_batch_hooks = []
options.after_dataset_hooks = []

inputs = ReconstructionInputs(
    model=model,
    datasets=datasets,
)
results = reconstruct(inputs, options)

t1 = results.V[0].real
np.save('/home/shengbo/Gd2O3_data/set3/joint_recon/Te_8slice_astra.npy', t1)
# np.save('/home/shengbo/Gd2O3_data/set3/joint_recon/test_fullres.npy', results)
losses_np = np.array(results.losses / error_norm_total)
plt.plot(losses_np[:])
plt.show()
