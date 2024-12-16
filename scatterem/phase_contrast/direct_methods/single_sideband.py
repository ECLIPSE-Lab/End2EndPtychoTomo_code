# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/phase_contrast.direct_methods.single_sideband.ipynb.

# %% auto 0
__all__ = ['aperture2', 'chi3', 'disk_overlap_function', 'single_sideband_reconstruction', 'batched_single_sideband_kernel',
           'single_sideband_kernel']

# %% ../../../nbs/phase_contrast.direct_methods.single_sideband.ipynb 2
import taichi as ti
import taichi.math as tm


@ti.func
def aperture2(qx: ti.float32, qy: ti.float32, wavelength: ti.float32, alpha_max: ti.float32):
    qx2 = qx ** 2
    qy2 = qy ** 2
    q = ti.sqrt(qx2 + qy2)
    ktheta = tm.asin(q * wavelength)
    return ktheta < alpha_max


@ti.func
def chi3(qy: ti.float32,
         qx: ti.float32,
         wavelength: ti.float32,
         C: ti.types.ndarray(ndim=1)):
    """
    Zernike polynomials in the cartesian coordinate system
    :param qx:
    :param qy:
    :param wavelength: wavelength in Angstrom
    :param C:   (12 ,)
    :return:
    """

    u = qx * wavelength
    v = qy * wavelength
    u2 = u ** 2
    u3 = u ** 3
    u4 = u ** 4
    # u5 = u ** 5

    v2 = v ** 2
    v3 = v ** 3
    v4 = v ** 4
    # v5 = v ** 5

    chi = 0.0

    # r-2 = x-2 +y-2.
    chi += 1 / 2 * C[0] * (u2 + v2)  # r^2
    # r-2 cos(2*phi) = x"2 -y-2.
    # r-2 sin(2*phi) = 2*x*y.
    chi += 1 / 2 * (C[1] * (u2 - v2) + 2 * C[2] * u * v)  # r^2 cos(2 phi) + r^2 sin(2 phi)
    # r-3 cos(3*phi) = x-3 -3*x*y'2. r"3 sin(3*phi) = 3*y*x-2 -y-3.
    chi += 1 / 3 * (C[5] * (u3 - 3 * u * v2) + C[6] * (3 * u2 * v - v3))  # r^3 cos(3phi) + r^3 sin(3 phi)
    # r-3 cos(phi) = x-3 +x*y-2.
    # r-3 sin(phi) = y*x-2 +y-3.
    chi += 1 / 3 * (C[3] * (u3 + u * v2) + C[4] * (v3 + u2 * v))  # r^3 cos(phi) + r^3 sin(phi)
    # r-4 = x-4 +2*x-2*y-2 +y-4.
    chi += 1 / 4 * C[7] * (u4 + v4 + 2 * u2 * v2)  # r^4
    # r-4 cos(4*phi) = x-4 -6*x-2*y-2 +y-4.
    chi += 1 / 4 * C[10] * (u4 - 6 * u2 * v2 + v4)  # r^4 cos(4 phi)
    # r-4 sin(4*phi) = 4*x-3*y -4*x*y-3.
    chi += 1 / 4 * C[11] * (4 * u3 * v - 4 * u * v3)  # r^4 sin(4 phi)
    # r-4 cos(2*phi) = x-4 -y-4.
    chi += 1 / 4 * C[8] * (u4 - v4)
    # r-4 sin(2*phi) = 2*x-3*y +2*x*y-3.
    chi += 1 / 4 * C[9] * (2 * u3 * v + 2 * u * v3)
    # r-5 cos(phi) = x-5 +2*x-3*y-2 +x*y-4.
    # r-5 sin(phi) = y*x"4 +2*x-2*y-3 +y-5.
    # r-5 cos(3*phi) = x-5 -2*x-3*y-2 -3*x*y-4.
    # r-5 sin(3*phi) = 3*y*x-4 +2*x-2*y-3 -y-5.
    # r-5 cos(5*phi) = x-5 -10*x-3*y-2 +5*x*y-4.
    # r-5 sin(5*phi) = 5*y*x-4 -10*x-2*y-3 +y-5.

    chi *= 2 * 3.141592653589793 / wavelength

    return chi


@ti.kernel
def disk_overlap_function(Γ: ti.types.ndarray(ndim=4),
                          Qx_all: ti.types.ndarray(ndim=1),
                          Qy_all: ti.types.ndarray(ndim=1),
                          Kx_all: ti.types.ndarray(ndim=1),
                          Ky_all: ti.types.ndarray(ndim=1),
                          aberrations: ti.types.ndarray(ndim=1),
                          theta_rot: ti.float32,
                          alpha: ti.float32,
                          wavelength: ti.float32):
    ti.loop_config(parallelize=8, block_dim=256)
    J, IKY, IKX, ww = Γ.shape
    for j, iky, ikx in ti.ndrange(J, IKY, IKX):
        Qx = Qx_all[j]
        Qy = Qy_all[j]
        Kx = Kx_all[ikx]
        Ky = Ky_all[iky]

        Qx_rot = Qx * tm.cos(theta_rot) - Qy * tm.sin(theta_rot)
        Qy_rot = Qx * tm.sin(theta_rot) + Qy * tm.cos(theta_rot)

        Qx = Qx_rot
        Qy = Qy_rot

        chi = chi3(Ky, Kx, wavelength, aberrations)
        apert = ti.math.vec2(aperture2(Ky, Kx, wavelength, alpha), 0)
        expichi = tm.cexp(ti.math.vec2(1, -chi))
        A = tm.cmul(apert, expichi)

        chi = chi3(Ky + Qy, Kx + Qx, wavelength, aberrations)
        apert = ti.math.vec2(aperture2(Ky + Qy, Kx + Qx, wavelength, alpha), 0)
        expichi = tm.cexp(ti.math.vec2(1, -chi))
        Ap = tm.cmul(apert, expichi)

        chi = chi3(Ky - Qy, Kx - Qx, wavelength, aberrations)
        apert = ti.math.vec2(aperture2(Ky - Qy, Kx - Qx, wavelength, alpha), 0)
        expichi = tm.cexp(ti.math.vec2(1, -chi))
        Am = tm.cmul(apert, expichi)

        gamma_complex = tm.cmul(tm.cconj(A), Am) - tm.cmul(A, tm.cconj(Ap))

        Γ[j, iky, ikx, 0] = gamma_complex[0]
        Γ[j, iky, ikx, 1] = gamma_complex[1]

import torch as th
def single_sideband_reconstruction(G,
                           Qx_all,
                           Qy_all,
                           Kx_all,
                           Ky_all,
                           aberrations,
                           theta_rot,
                           alpha,
                           object_bright_field,
                           object_ssb,
                           eps,
                           wavelength):
    device = G.device
    nx, ny, _, _ = G.shape
    object_bright_field = th.zeros((ny, nx), dtype=G.dtype, device=device)
    object_ssb = th.zeros((ny, nx), dtype=G.dtype, device=device)
    eps = 1e-3
    single_sideband_kernel(
        th.view_as_real(G.contiguous()),
        Qx1d.to(device),
        Qy1d.to(device),
        Kx.to(device),
        Ky.to(device),
        aberrations.to(device),
        best_angle,
        meta.alpha_rad,
        th.view_as_real(object_bright_field),
        th.view_as_real(object_ssb),
        eps,
        meta.wavelength,
    )
    object_bright_field = th.fft.ifft2(object_bright_field, norm='ortho')
    object_ssb = th.fft.ifft2(object_ssb, norm='ortho')
    angle_obj_ssb = np.angle(object_ssb.cpu().numpy())
    from skimage.restoration import unwrap_phase
    from skimage import exposure
    angle_obj_ssb = unwrap_phase(angle_obj_ssb)
    
    return object_bright_field, object_ssb

def batched_single_sideband_kernel(G: th.tensor,
                                   Qx_all: th.tensor,
                                   Qy_all: th.tensor,
                                   Kx_all: th.tensor,
                                   Ky_all: th.tensor,
                                   aberrations: th.tensor,
                                   # dot_product_bf: th.tensor,
                                   # dot_produt_ssb: th.tensor,
                                   theta_rot: float,
                                   alpha: float,
                                   object_bright_field: th.tensor,
                                   object_ssb: th.tensor,
                                   eps: float,
                                   wavelength: float):
    i = 0
    for aberrations_i, object_bright_field_i, object_ssb_i in zip(aberrations, object_bright_field, object_ssb):
        # dot_product_bf_i = th.zeros((1,), device = G.device)
        # dot_produt_ssb_i = th.zeros((1,), device = G.device)
        single_sideband_kernel(
            th.view_as_real(G),
            Qx_all,
            Qy_all,
            Kx_all,
            Ky_all,
            aberrations_i,
            # dot_product_bf_i,
            # dot_produt_ssb_i,
            theta_rot,
            alpha,
            th.view_as_real(object_bright_field_i),
            th.view_as_real(object_ssb_i),
            eps,
            wavelength
        )
        # dot_product_bf[i] = dot_product_bf_i
        # dot_produt_ssb[i] = dot_produt_ssb_i
        # i += 1


@ti.kernel
def single_sideband_kernel(G: ti.types.ndarray(ndim=5),
                           Qx_all: ti.types.ndarray(ndim=1),
                           Qy_all: ti.types.ndarray(ndim=1),
                           Kx_all: ti.types.ndarray(ndim=1),
                           Ky_all: ti.types.ndarray(ndim=1),
                           aberrations: ti.types.ndarray(ndim=1),
                           # dot_product_bf: ti.types.ndarray(ndim=1),
                           # dot_produt_ssb: ti.types.ndarray(ndim=1),
                           theta_rot: ti.float32,
                           alpha: ti.float32,
                           object_bright_field: ti.types.ndarray(ndim=3),
                           object_ssb: ti.types.ndarray(ndim=3),
                           eps: ti.float32,
                           wavelength: ti.float32):
    IQY, IQX, IKY, IKX, cx = G.shape
    ti.loop_config(parallelize=8, block_dim=256)
    for iqy, iqx, iky, ikx in ti.ndrange(IQY, IQX, IKY, IKX):

        Qx = Qx_all[iqx]
        Qy = Qy_all[iqy]
        Kx = Kx_all[ikx]
        Ky = Ky_all[iky]

        Qx_rot = Qx * tm.cos(theta_rot) - Qy * tm.sin(theta_rot)
        Qy_rot = Qx * tm.sin(theta_rot) + Qy * tm.cos(theta_rot)

        Qx = Qx_rot
        Qy = Qy_rot

        chi = chi3(Ky, Kx, wavelength, aberrations)
        apert = ti.math.vec2(aperture2(Ky, Kx, wavelength, alpha), 0)
        expichi = tm.cexp(ti.math.vec2(1, -chi))
        A = tm.cmul(apert, expichi)

        chi = chi3(Ky + Qy, Kx + Qx, wavelength, aberrations)
        apert = ti.math.vec2(aperture2(Ky + Qy, Kx + Qx, wavelength, alpha), 0)
        expichi = tm.cexp(ti.math.vec2(1, -chi))
        Ap = tm.cmul(apert, expichi)

        chi = chi3(Ky - Qy, Kx - Qx, wavelength, aberrations)
        apert = ti.math.vec2(aperture2(Ky - Qy, Kx - Qx, wavelength, alpha), 0)
        expichi = tm.cexp(ti.math.vec2(1, -chi))
        Am = tm.cmul(apert, expichi)

        gamma_complex = tm.cmul(tm.cconj(A), Am) - tm.cmul(A, tm.cconj(Ap))

        Kplus = tm.sqrt((Kx + Qx) ** 2 + (Ky + Qy) ** 2)
        Kminus = tm.sqrt((Kx - Qx) ** 2 + (Ky - Qy) ** 2)
        K = tm.sqrt(Kx ** 2 + Ky ** 2)
        bright_field = K < alpha / wavelength
        double_overlap1 = (Kplus < alpha / wavelength) * bright_field * (Kminus > alpha / wavelength)

        Γ_abs = ti.abs(gamma_complex)
        take = Γ_abs[0] > eps and bright_field
        if take:
            g = ti.math.vec2(G[iqy, iqx, iky, ikx, 0], G[iqy, iqx, iky, ikx, 1])
            val = tm.cmul(g, tm.cconj(gamma_complex))
            ti.atomic_add(object_bright_field[iqy, iqx, 0], val[0])
            ti.atomic_add(object_bright_field[iqy, iqx, 1], val[1])
            # ti.atomic_add(dot_product_bf[0], ti.abs(val)[0])
        if double_overlap1:
            g = ti.math.vec2(G[iqy, iqx, iky, ikx, 0], G[iqy, iqx, iky, ikx, 1])
            val = tm.cmul(g, tm.cconj(gamma_complex))
            ti.atomic_add(object_ssb[iqy, iqx, 0], val[0])
            ti.atomic_add(object_ssb[iqy, iqx, 1], val[1])
            # ti.atomic_add(dot_produt_ssb[0], ti.abs(val)[0])
        if iqx == 0 and iqy == 0:
            g = ti.math.vec2(G[iqy, iqx, iky, ikx, 0], G[iqy, iqx, iky, ikx, 1])
            val = ti.abs(g)
            ti.atomic_add(object_bright_field[iqy, iqx, 0], val[0])
            ti.atomic_add(object_ssb[iqy, iqx, 0], val[0])

