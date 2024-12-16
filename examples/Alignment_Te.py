import torch as th
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scatterem.solvers import (ray_transform, rotate_volume, LinearTomographyInputs, LinearTomographyOptions,
                               LinearTomographyResults, LinearTomographySolver, TomographyAlignmentInputs,
                               TomographyAlignmentOptions, TomographyAlignmentResults, TomographyAlignmentSolver)

def mosaic(data):
    n, w, h = data.shape
    diff = np.sqrt(n) - int(np.sqrt(n))
    s = np.sqrt(n)
    m = int(s)
    # print 'm', m
    if diff > 1e-6: m += 1
    mosaic = np.zeros((m * w, m * h)).astype(data.dtype)
    for i in range(m):
        for j in range(m):
            if (i * m + j) < n:
                mosaic[i * w:(i + 1) * w, j * h:(j + 1) * h] = data[i * m + j]
    return mosaic

def plotmosaic(img, title='Image', savePath=None, cmap='hot', show=True, dpi=150, vmax=None):
    fig, ax = plt.subplots(dpi=dpi)
    mos = mosaic(img)
    cax = ax.imshow(mos, interpolation='nearest', cmap=plt.cm.get_cmap(cmap), vmax=vmax)
    cbar = fig.colorbar(cax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.grid(False)
    plt.show()
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath + '.png', dpi=dpi)

device = th.device('cuda:0')
# %%

path_angles = '/home/shengbo/FAUbox/Pelz Lab (Philipp Pelz)/shengbo/te_particle/'
# path_angles = '/home/philipp/FAUbox/Pelz Lab/shengbo/te_particle/pytorch_radon/'
#
psi = np.load(path_angles + 'psi.npy')
theta = np.load(path_angles + 'theta.npy')
phi = np.load(path_angles + 'phi.npy')
translation_pix = np.load(path_angles + 'translation_pix.npy')
# translation_pix = 2 * translation_pix/296

psi = np.delete(psi, [16, 17, 18, 19, 20])
theta = np.delete(theta, [16, 17, 18, 19, 20])
phi = np.delete(phi, [16, 17, 18, 19, 20])
translation_pix = np.delete(translation_pix, [16, 17, 18, 19, 20], axis=1)

phi_target = th.as_tensor(psi, dtype=th.float32, device=device)
theta_target = th.as_tensor(phi, dtype=th.float32, device=device)
psi_target = th.as_tensor(theta, dtype=th.float32, device=device)
translations_target = th.as_tensor(translation_pix, dtype=th.float32, device=device)



base_path = Path('/home/shengbo/FAUbox/Pelz Lab (Philipp Pelz)/shengbo/te_particle/')

fn_stack = 'stack.npy'
fn_angles = 'angles.npy'

sino_target_ori = np.load(base_path / fn_stack).astype('float32')
sino_target_ori = th.as_tensor(np.delete(sino_target_ori, [16], axis=0))
h = 36
w = 36
sino_target = sino_target_ori[:, h:-h, w:-w]

ph_size = sino_target.shape[-1]
ph = th.zeros(ph_size, ph_size, ph_size)

theta_init = np.load(base_path / fn_angles).astype('float32')
theta_init = np.delete(theta_init, [16])

phi_init = np.zeros_like(theta_init)
psi_init = np.zeros_like(theta_init)
translations_init = np.zeros([2, theta_init.shape[0]])

theta_init = th.as_tensor(theta_init, dtype=th.float32, device=device) * th.pi / 180
phi_init = th.as_tensor(phi_init, dtype=th.float32, device=device) * th.pi / 180
psi_init = th.as_tensor(psi_init, dtype=th.float32, device=device) * th.pi / 180
translations_init = th.as_tensor(translations_init, dtype=th.float32, device=device)

print(f'psi target shape:', psi_target.shape)
print(f'theta target shape:', theta_target.shape)
print(f'phi target shape:', phi_target.shape)
print(f'translation target shape:', translations_target.shape)

print(f'psi initial shape:', psi_init.shape)
print(f'theta initial shape:', theta_init.shape)
print(f'phi initial shape:', phi_init.shape)
print(f'translation initial shape:', translations_init.shape)

plotmosaic(sino_target, "Sino target")
# %%
# fig, ax = plt.subplots()
# ax.scatter()
#%%
# Define callbacks

def plot_tomo_results(epoch, results):
    if epoch >= 299:
        volume = results.volume.squeeze().cpu().detach().numpy()
        losses = results.losses
        sinogram = results.sinogram.squeeze().cpu().detach().numpy()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].scatter(np.arange(len(losses)), np.log10(losses))
        slice = int(volume.shape[0] / 2)
        ax[1].imshow(volume[slice])
        ax[2].imshow(mosaic(sinogram))
        ax[0].set_title(f'Loss final: {losses[-1]}')
        plt.show()


epoch_to_show = 5
target_alignment = TomographyAlignmentResults(None, phi_target, theta_target, psi_target, translations_target, None)
def plot_intermediate_results(epoch, results: TomographyAlignmentResults):
    global epoch_to_show
    global target_alignment
    if epoch != 0 and epoch % epoch_to_show == 0:
        epoch_to_show *= 2
        number_of_angles = len(results.phi)
        fig, axs = plt.subplots(2, 4, figsize=(17, 10))

        losses = results.losses
        phi = results.phi.cpu().numpy().squeeze()
        theta = results.theta.cpu().numpy().squeeze()
        psi = results.psi.cpu().numpy().squeeze()
        translations = results.translations.cpu().numpy().squeeze()
        phi_init = target_alignment.phi.cpu().numpy().squeeze()
        theta_init = target_alignment.theta.cpu().numpy().squeeze()
        psi_init = target_alignment.psi.cpu().numpy().squeeze()
        translations_init = target_alignment.translations.cpu().numpy().squeeze()


        axs[0, 0].scatter(np.arange(number_of_angles), phi)
        axs[0, 0].scatter(np.arange(number_of_angles), phi_init)
        axs[0, 1].scatter(np.arange(number_of_angles), theta)
        axs[0, 1].scatter(np.arange(number_of_angles), theta_init)
        axs[0, 2].scatter(np.arange(number_of_angles), psi)
        axs[0, 2].scatter(np.arange(number_of_angles), psi_init)
        axs[0, 3].scatter(translations[0], translations[1])
        axs[0, 3].scatter(translations_init[0], translations_init[1])

        axs[1, 0].scatter(np.arange(len(losses)), np.log10(losses))
        axs[1, 1].scatter(np.arange(number_of_angles), theta - theta_init)
        axs[1, 2].scatter(np.arange(number_of_angles), translations[0])
        axs[1, 2].scatter(np.arange(number_of_angles), translations_init[0])
        axs[1, 3].scatter(np.arange(number_of_angles), translations[1])
        axs[1, 3].scatter(np.arange(number_of_angles), translations_init[1])

        axs[0, 0].set_title('phi')
        axs[0, 1].set_title('theta')
        axs[0, 2].set_title('psi')
        axs[0, 3].set_title('translations')
        axs[1, 0].set_title('losses')
        axs[1, 1].set_title('theta difference')
        axs[1, 2].set_title('translation x')
        axs[1, 3].set_title('translation z')

        fig.suptitle(f'Epoch {epoch}')
        plt.show()
#%%

#%%
# Test alignment
# Translations

volume = th.zeros(ph.shape, device=device, dtype=th.float32)
phi_init = th.zeros_like(phi_init)
psi_init = th.zeros_like(psi_init)
translations_init = th.zeros_like(translations_init)
alignment_inputs = TomographyAlignmentInputs(sino_target, volume, phi_init, theta_init,
                                             psi_init, translations_init)

options = LinearTomographyOptions()
options.device = device
options.num_iterations = 30
options.optimizer_params['lr'] = 0.05
options.reg_pars['regularisation_parameter'] = 5e-4
options.callback = None
options.edge_reg_pars['regularisation_parameter'] = 1


alignment_options = TomographyAlignmentOptions(options)
alignment_options.num_iterations = 100
alignment_options.optimizers_params = {'phi': {'lr': 1e-4},
                                       'theta': {'lr': 1e-4},
                                       'psi': {'lr': 1e-5},
                                       'translations': {'lr': 1e-2}}
alignment_options.to_fit = {'phi': False, 'theta': False, 'psi': False, 'translations': True}
alignment_options.callback = plot_intermediate_results
epoch_to_show = 5
alignment_options.angle_reg_options['regularisation_parameter'] = 1e-2

alignmentSolver = TomographyAlignmentSolver(alignment_inputs, alignment_options)
alignment_results = alignmentSolver.solve()
#%%
alignment_results.volume.max()
#%%
# Test alignment
# Rotation amd Translation
volume = th.zeros(ph.shape, device=device, dtype=th.float32)
phi_init = th.zeros_like(phi_init)
psi_init = th.zeros_like(psi_init)
translations_init = th.clone(alignment_results.translations.squeeze()).detach()
alignment_inputs = TomographyAlignmentInputs(sino_target, volume, phi_init, theta_init,
                                             psi_init, translations_init)

options = LinearTomographyOptions()
options.device = device
options.num_iterations = 100
options.optimizer_params['lr'] = 0.05
options.reg_pars['regularisation_parameter'] = 5e-4
options.callback = None
options.edge_reg_pars['regularisation_parameter'] = 1


alignment_options = TomographyAlignmentOptions(options)
alignment_options.num_iterations = 500
alignment_options.optimizers_params = {'phi': {'lr': 1e-4},
                                       'theta': {'lr': 1e-4},
                                       'psi': {'lr': 1e-4},
                                       'translations': {'lr': 1e-2}}
alignment_options.to_fit = {'phi': True, 'theta': True, 'psi': True, 'translations': True}
alignment_options.callback = plot_intermediate_results
epoch_to_show = 5
alignment_options.angle_reg_options['regularisation_parameter'] = 1e-5

alignmentSolver = TomographyAlignmentSolver(alignment_inputs, alignment_options)
alignment_results = alignmentSolver.solve()
#%%
np.save('test_phantom_alignment.npy', alignment_results.volume.squeeze().cpu().detach().numpy())
#%%
alignment_results.translations[1] - target_alignment.translations[1]
#%%
sino_result = ray_transform(alignment_results.volume,
                            alignment_results.phi,
                            alignment_results.theta,
                            alignment_results.psi,
                            alignment_results.translations)
sino_result = sino_result.squeeze().cpu().detach().numpy()

plotmosaic(sino_result, "sino_result")

from torch.nn.functional import mse_loss

loss_test = mse_loss(th.as_tensor(sino_result), th.as_tensor(sino_target))
np.log10(loss_test)

angle_reg_pars = alignmentSolver.angles_reg_init(alignment_options.angle_reg_options, alignment_results.phi)
loss_angles = alignmentSolver.angles_regularization_diff(alignment_results.phi,
                                                         alignment_results.theta,
                                                         alignment_results.psi,
                                                         angle_reg_pars,
                                                         alignment_options.angle_reg_options)

print(np.log10(loss_test + loss_angles.cpu().detach()))
#%%