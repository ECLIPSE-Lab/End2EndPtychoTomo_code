# plot RDF with cluster of unit cell
import scipy.io
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from pathlib import Path
from rdfpy import rdf
from scipy.ndimage import gaussian_filter1d
import ase.io

dx = 0.396

# %% Define the FSC function, from 'teamtomo', https://github.com/teamtomo/libtilt/blob/main/src/libtilt/fsc/fsc.py
import einops
import torch

from scatterem.util.fftfreq_grid import fftfreq_grid
from scatterem.util.fft_utils import rfft_shape

def fsc(
    a: torch.Tensor,
    b: torch.Tensor,
    rfft_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Fourier ring/shell correlation between two square/cubic images."""
    # input handling
    image_shape = a.shape
    dft_shape = rfft_shape(image_shape)
    if a.ndim not in (2, 3):
        raise ValueError('images must be 2D or 3D.')
    elif a.shape != b.shape:
        raise ValueError('images must be the same shape.')
    elif rfft_mask is not None and rfft_mask.shape != dft_shape:
        raise ValueError('valid rfft indices must have same shape as rfft.')

    # linearise data and fftfreq of each component
    a, b = torch.fft.rfftn(a), torch.fft.rfftn(b)
    frequency_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=True,
        fftshift=False,
        norm=True,
        device=a.device,
    )
    if rfft_mask is not None:
        a, b, frequencies = (arr[rfft_mask] for arr in [a, b, frequency_grid])
    else:
        a, b, frequencies = (torch.flatten(arg) for arg in [a, b, frequency_grid])

    # define frequency bins
    bin_centers = torch.fft.rfftfreq(image_shape[0])
    df = 1 / image_shape[0]

    # define split points in data as midpoint between bin centers
    bin_centers = torch.cat([bin_centers, torch.as_tensor([0.5 + df])])
    bin_centers = bin_centers.unfold(dimension=0, size=2, step=1)  # (n_shells, 2)
    split_points = einops.reduce(bin_centers, 'shells high_low -> shells', reduction='mean')

    # find indices of all components in each shell
    sorted_frequencies, sort_idx = torch.sort(frequencies, descending=False)
    split_idx = torch.searchsorted(sorted_frequencies, split_points)
    shell_idx = torch.tensor_split(sort_idx, split_idx)[:-1]

    # calculate normalised cross correlation in each shell
    fsc = [
        _normalised_cc_complex_1d(a[idx], b[idx])
        for idx in
        shell_idx
    ]
    fscth = [idx.shape for idx in shell_idx]
    fsc_real = torch.real(torch.tensor(fsc))
    fscth_real = torch.real(torch.tensor(fscth))

    onebit_thres = (0.5 + 2.4142/th.sqrt(fscth_real))/(1.5 + 1.4142/th.sqrt(fscth_real))
    return fsc_real, onebit_thres


def _normalised_cc_complex_1d(a: torch.Tensor, b: torch.Tensor):
    correlation = torch.dot(a, torch.conj(b))
    return correlation / (torch.linalg.norm(a) * torch.linalg.norm(b))

# %% plot FSC with one-bit criteria
datapath = Path('/home/shengbo/PycharmProjects/zrte2_particle/hyperparameters')
vol1 = np.load(str(datapath) + '/downsample_d2.npy')
vol2 = np.load(str(datapath) + '/downsample_d2_other.npy')
vol3 = np.load(str(datapath) + '/downsample_d8.npy')
vol4 = np.load(str(datapath) + '/downsample_d8_other.npy')
vol5 = np.load(str(datapath) + '/downsample_d12.npy')
vol6 = np.load(str(datapath) + '/downsample_d12_other.npy')
vol7 = np.load(str(datapath) + '/downsample_d20.npy')
vol8 = np.load(str(datapath) + '/downsample_d20_other.npy')
vol9 = np.load(str(datapath) + '/downsample_d21.npy')
vol10 = np.load(str(datapath) + '/downsample_d21_other.npy')
vol1 = th.as_tensor(vol1)
vol2 = th.as_tensor(vol2)
vol3 = th.as_tensor(vol3)
vol4 = th.as_tensor(vol4)
vol5 = th.as_tensor(vol5)
vol6 = th.as_tensor(vol6)
vol7 = th.as_tensor(vol7)
vol8 = th.as_tensor(vol8)
vol9 = th.as_tensor(vol9)
vol10 = th.as_tensor(vol10)
fsc_result1, hbth = fsc(vol1, vol2)
fsc_result2, hbth = fsc(vol3, vol4)
fsc_result3, hbth = fsc(vol5, vol6)
fsc_result4, hbth = fsc(vol7, vol8)
fsc_result5, hbth = fsc(vol9, vol10)
# %%
wavevec = np.arange(0, 149, 1)/(dx*148)

plt.plot(wavevec, fsc_result1, 'b', label='FSC subsample by 2')
# plt.plot(wavevec, fsc_result2, 'r', label='FSC subsample by 8')
plt.plot(wavevec, fsc_result3, 'g', label='FSC subsample by 12')
plt.plot(wavevec, fsc_result4, 'r', label='FSC subsample by 20')
# plt.plot(wavevec, fsc_result5, 'k', label='FSC subsample by 21')
plt.plot(wavevec, hbth, '--', label='one-bit threshold')
plt.legend()
plt.ylabel('FSC')
plt.xlabel('Spatial frequency, $Å^{-1}$')
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.show()


# %% Define the Radially Averaged Power Spectrum
import numpy as np
import torch as th
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

def volume_to_polar(vol, center=None):
    if center is None:
        center = (vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2)

    # Create polar coordinates grid
    theta = np.linspace(0, 2*np.pi, max(vol.shape), endpoint=False)
    r = np.arange(0, max(vol.shape[0]//2, vol.shape[1]//2, vol.shape[2]//2), 1)

    # Convert polar coordinates to Cartesian coordinates
    x = center[0] + np.outer(r, np.cos(theta))
    y = center[2] + np.outer(r, np.sin(theta))
    z = center[1] + np.outer(r, np.sin(theta))

    # Interpolate values from Cartesian to polar coordinates
    polar_volume = map_coordinates(vol, [x, y, z], order=1, mode='constant', cval=0)

    # Reshape the result to polar coordinates
    polar_volume = polar_volume.reshape((len(r), len(theta)))

    return polar_volume

def radial_integration(polar_volume):
    k = polar_volume.shape[1] - 1
    return np.trapz(polar_volume, axis=1) / k


# %% plot RAPSD for reconstructed volumes with different hyperparameters
ampthres = 0.003
datapath = Path('/home/shengbo/PycharmProjects/zrte2_particle/hyperparameters')
# vol1 = np.load(str(datapath) + '/8slice_butter2.npy')
# vol2 = np.load(str(datapath) + '/8slice_butter8.npy')
# vol3 = np.load(str(datapath) + '/8slice_butterlast90.npy')
# vol4 = np.load(str(datapath) + '/8slice_tau0_real.npy')
# vol5 = np.load(str(datapath) + '/8slice_tau1e7_real.npy')
# vol6 = np.load(str(datapath) + '/8slice_L160.npy')

vol1 = np.load(str(datapath) + '/1slice_kernel40.npy')
vol2 = np.load(str(datapath) + '/1slice_kernel120.npy')
vol3 = np.load(str(datapath) + '/4slice_kernel40.npy')
vol4 = np.load(str(datapath) + '/4slice_kernel120.npy')
vol5 = np.load(str(datapath) + '/8slice_kernel40.npy')
vol6 = np.load(str(datapath) + '/8slice_kernel120.npy')

vol1[vol1 < ampthres] = 0
vol2[vol2 < ampthres] = 0
vol3[vol3 < ampthres] = 0
vol4[vol4 < ampthres] = 0
vol5[vol5 < ampthres] = 0
vol6[vol6 < ampthres] = 0

ftvol1 = np.abs(np.fft.fftshift(np.fft.fftn(vol1))) ** 2
polar_vol1 = volume_to_polar(ftvol1)
rapsd1 = radial_integration(polar_vol1)

ftvol2 = np.abs(np.fft.fftshift(np.fft.fftn(vol2))) ** 2
polar_vol2 = volume_to_polar(ftvol2)
rapsd2 = radial_integration(polar_vol2)

ftvol3 = np.abs(np.fft.fftshift(np.fft.fftn(vol3))) ** 2
polar_vol3 = volume_to_polar(ftvol3)
rapsd3 = radial_integration(polar_vol3)

ftvol4 = np.abs(np.fft.fftshift(np.fft.fftn(vol4))) ** 2
polar_vol4 = volume_to_polar(ftvol4)
rapsd4 = radial_integration(polar_vol4)

ftvol5 = np.abs(np.fft.fftshift(np.fft.fftn(vol5))) ** 2
polar_vol5 = volume_to_polar(ftvol5)
rapsd5 = radial_integration(polar_vol5)

ftvol6 = np.abs(np.fft.fftshift(np.fft.fftn(vol6))) ** 2
polar_vol6 = volume_to_polar(ftvol6)
rapsd6 = radial_integration(polar_vol6)

halfvsize = rapsd6.shape[0]
wavevec = np.arange(0, halfvsize, 1)/(dx*halfvsize)


# %%
plt.plot(wavevec, rapsd1, 'k', label='1 slice PSF relax')
plt.plot(wavevec, rapsd2, '--', c='k', label='1 slice PSF strict')
plt.plot(wavevec, rapsd3, 'r', label='4 slice PSF relax')
plt.plot(wavevec, rapsd4, '--', c='r', label='4 slice PSF strict')
plt.plot(wavevec, rapsd5, 'b', label='8 slice PSF relax')
plt.plot(wavevec, rapsd6, '--', c='b', label='8 slice PSF strict')
plt.legend()
plt.yscale('log', base=10)
plt.ylabel('Radially Integrated Power Spectrum')
plt.xlabel('Spatial frequency, $Å^{-1}$')
plt.grid(color='green', linestyle='--', linewidth=0.5)
regions = [0.6, 1.20, 1.25, 1.6]
for rr in regions:
    plt.axvline(rr)
plt.show()
