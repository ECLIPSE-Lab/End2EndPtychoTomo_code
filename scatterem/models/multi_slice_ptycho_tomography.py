# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/models.multi_slice_ptycho_tomography.ipynb.

# %% auto 0
__all__ = ['MultiSlicePtychographyTomographyModel']

# %% ../../nbs/models.multi_slice_ptycho_tomography.ipynb 1
from . import BaseModel
from ..operators.composite import exitwave_measurement, grid_sample_partial, bin_z
from ..operators.split import BatchCropVolume, SubpixShift
from ..kernels import overlap_real_volume, overlap_intensity_no_subpix, overlap_intensity
from ..core import dtype_complex, dtype_real
from ..util.plot import plot, plotAbsAngle
import torch as th 
from ..operators.basic import Bin
from kornia.filters import filter3d


class MultiSlicePtychographyTomographyModel(BaseModel):
    def __init__(self,
                 object,
                 probe,
                 positions,
                 dr,
                 error_norm,
                 tau1,
                 tau2,
                 angles,
                 translations,
                 start_end_model,
                 loss_function,
                 propagator,
                 braggpix,
                 butterorder,
                 dx,
                 bin_factor,
                 intensity_scaling_factor,
                 kernel = None,
                 subpix_shift = False):
        self.subpix_shift = subpix_shift
        MP = th.tensor(probe[0][0].shape[1:])
        if subpix_shift:
            self.shift = SubpixShift(MP[0].item(), MP[1].item(), probe[0][0].device)
        else:
            self.shift = None
        self.object = object
        self.probe = probe
        self.positions = positions
        self.dr = dr
        self.error_norm = error_norm
        self.tau1 = tau1
        self.tau2 = tau2
        self.angles = angles
        self.translations = translations
        self.start_end_model = start_end_model
        self.loss_function = loss_function
        self.propagator = propagator
        self.braggpix = braggpix
        self.butterorder = butterorder
        self.dx = dx
        self.bin_factor = bin_factor
        self.intensity_scaling_factor = intensity_scaling_factor
        NVol, NY, NZ, NX = self.object.shape
        # TODO make general
        Nnodes, kk, MY, MX = probe[0][0].shape
        # TODO make general
        K, _ = positions[0].shape
        self.object_norm = th.zeros((NY, NZ // self.bin_factor, NX), device=self.object.device, dtype=dtype_real,
                                    requires_grad=False)
        self.probe_norm = th.zeros((Nnodes, kk, MY, MX), device=self.object.device, dtype=dtype_real, requires_grad=False)
        self.NZ = NZ

        def cropping_backward_hook(module, grad_input, grad_output):
            grad_object, grad_wave, grad_r = grad_input
            alpha = 0.9
            denom = th.sqrt(1e-16 + ((1 - alpha) * self.object_norm) ** 2 + (alpha * th.max(self.object_norm)) ** 2)
            # plot(denom[:,4,:].cpu().numpy(), ' denom')
            new_grad_object_patches = grad_object / denom
            return new_grad_object_patches, grad_wave, grad_r
        
        def binning_backward_hook(module, grad_input, grad_output):
            a, = grad_input
            new_grad_input = filter3d(a.unsqueeze(0).unsqueeze(0), kernel, border_type='constant')
            new_grad_input = new_grad_input[0][0]
            # vol = volk
            # volk = th.abs(th.fft.ifftn(th.fft.fftn(vol) * kernel))
            new_grad_input = new_grad_input * th.sqrt(th.sum(th.abs(a)**2)/th.sum(th.abs(new_grad_input)**2))
            # grad_sum = th.sum(grad_output)
            return new_grad_input,

        self.batch_crop_volume = BatchCropVolume()
        self.batch_crop_volume.register_full_backward_hook(cropping_backward_hook)
        self.bin_z = Bin(bin_factor)
        self.bin_z.register_full_backward_hook(binning_backward_hook)
        
    def shrink_nonnegative(self, object, tau1, tau2):
        x = object[0]
        x_complex = th.view_as_real(x)
        x_real = x_complex[:,:,:,0]
        x_imag = x_complex[:,:,:,1]
        # # # # x_real[th.sign(x_real) < 0] += tau1
        # # # # x_real[th.sign(x_real) > 0] -= tau1
        # # # # x_imag[th.sign(x_imag) < 0] += tau2
        # # # # x_imag[th.sign(x_imag) > 0] -= tau2
        x_real = x_real - tau1
        x_imag = x_imag - tau2
        x_stack = th.stack([x_real, x_imag], 3)
        x_new = th.view_as_complex(x_stack)
        # tau1_tensor = th.ones_like(model) * tau1
        # tau2_tensor = th.ones_like(model) * tau2
        # model = model - tau1_tensor - 1j*tau2_tensor
        return x_new.unsqueeze(0)
        
    def scale_probe_gradient(self, probe):
        # alpha = 0.9
        # denom = th.sqrt(1e-16 + ((1 - alpha) * self.probe_norm) ** 2 + (alpha * th.max(self.probe_norm)) ** 2)
        # # print('denom ', denom.max().item(), denom.min().item())
        # denom = th.prod(denom, dim=0, keepdim=True)
        # probe.grad /= denom
        pass
    
    def scale_object_gradient(self, object):
        # alpha = 0.9
        # denom = th.sqrt(1e-16 + ((1 - alpha) * self.object_norm) ** 2 + (alpha * th.max(self.object_norm)) ** 2)
        # object.grad /= denom
        # self.object_norm.fill_(0)
        pass
    
    def multislice_exitwave_multimode(self, object_patches, waves, pos, propagator):
        """
        Implements the multislice algorithm - no anti-aliasing masks
        :param object_patches:             NZ_bin x K x M1 x M2         complex
        :param object_patch_normalization: NZ_bin x K x M1 x M2         real
        :param waves:                      Nmodes x K x M1 x M2     complex
        :param propagator: f: (K x M1 x M2, Nmodes      x K x M1 x M2) -> Nmodes      x K x M1 x M2
        :return: exitwaves:                Nmodes x K x M1 x M2
        """
        for i in range(len(object_patches)):
            waves = object_patches[i].unsqueeze(0) * waves
            if object_patches.requires_grad:
                tmp = th.zeros_like(self.object_norm[:, i, :])
                if self.subpix_shift:
                    overlap_intensity(pos.data.to(th.int64), th.view_as_real(waves.data), tmp.data)
                else:
                    patch_norm = th.sum(th.abs(waves.clone().detach()) ** 2, 0)
                    overlap_real_volume(pos.data.to(th.int64), patch_norm.data, tmp.data)
                self.object_norm[:, i, :] = tmp
            waves = propagator(waves)
        return waves

    def object_model(self, V, probe, pos, angles, translation, bin_factor, start_end):
        V_rot_partial_real = grid_sample_partial(V.real.unsqueeze(0), angles[[0]], angles[[1]], angles[[2]],
                                                 translation,
                                                 start_end)
        V_rot_partial_imag = grid_sample_partial(V.imag.unsqueeze(0), angles[[0]], angles[[1]], angles[[2]],
                                                 translation,
                                                 start_end)
        V_rot_partial_real = self.bin_z(V_rot_partial_real)
        V_rot_partial_imag = self.bin_z(V_rot_partial_imag)

        V_rot_partial = V_rot_partial_real + 1j * V_rot_partial_imag
        T_rot_partial = th.exp(1j * V_rot_partial)
        # NX x NZ x NY -> NZ x K x M1 x M2
        T_rot_patches = self.batch_crop_volume(T_rot_partial, probe, pos)
        return T_rot_patches

    # object_patches, positions, probe, propagator, factor
    def measurement_model(self, object_patches, pos, dr, probe, propagator, factor):
        """

        :param object_patches: (NZ, K, MY, MX) complex
        :param object_patch_normalization: (NZ, K, MY, MX) real
        :param pos: (K, 2) real
        :param probe: (Nmodes, K, MY, MX) complex
        :param propagator: Callable
        :param factor: float
        :return: (K, MY, MX) real
        """
        # Nmodes x K x MY x MX
        if probe.requires_grad:
            # sum over positions
            self.probe_norm = th.sum(th.abs(object_patches.detach()) ** 2, 1, keepdim=True)
        if self.subpix_shift:
            probe = self.shift(probe, dr)
        else:
            # expand probe to K positions to keep dimension same as subpix_shifted probe
            ps = probe.shape            
            probe = probe.expand((ps[0], pos.shape[0], ps[2], ps[3]))
        phi = self.multislice_exitwave_multimode(object_patches, probe, pos, propagator)
        measurements = exitwave_measurement(phi)
        # K x MY x MX
        return measurements

    def __call__(self, object, probe, positions, dr, angle=None, translation=None, bin_factor=None, start_end=None,
                 propagator=None, factor=1):
        # NZ x K x MY x MX
        Nvol, NY, NZ, NX = object.shape
        _, _, MY, MX = probe.shape
        K, _ = positions.shape
        object_patches = self.object_model(object, probe, positions, angle, translation, bin_factor, start_end)
        measurements = self.measurement_model(object_patches, positions, dr, probe, propagator, factor)
        return measurements
