import numpy as np
import matplotlib.pyplot as plt
import abtem
import ase.io
from pathlib import Path
import torch as th

def affine_matrix_3D_ZYX(phi, theta, psi):
    c1 = th.cos(phi)
    s1 = th.sin(phi)
    c2 = th.cos(theta)
    s2 = th.sin(theta)
    c3 = th.cos(psi)
    s3 = th.sin(psi)
    zeros = th.zeros_like(phi)
    line1 = th.stack([c1*c2, c1*s2*s3 - c3*s1, s1*s3 + c1*c3*s2, zeros], 1)
    line2 = th.stack([c2*s1, c1*c3 + s1*s2*s3, c3*s1*s2 - c1*s3, zeros], 1)
    line3 = th.stack([  -s2,            c2*s3,            c2*c3, zeros], 1)
    R = th.stack([line1, line2, line3], 1)
    return R

abtem.config.set({"device": "gpu", "fft": "fftw"})
# %% Build the super unit cell
basepath = Path('./')
smrcokunit = ase.io.read(str(basepath) + '/bc3006sup1.cif')
ptunit = ase.io.read(str(basepath) + '/Pt.cif')

sizex = 4
sizey = 4
sizez = 4
smrcok = smrcokunit * (sizex, sizey, sizez)

sizept = 4
pt = ptunit * (sizept, sizept, sizept)

pt.positions[:, 0] += (smrcok.positions[:, 0].max() / 2) - (pt.positions[:, 0].max() / 2)
pt.positions[:, 1] += (smrcok.positions[:, 1].max() / 2) - (pt.positions[:, 1].max() / 2)
pt.positions[:, 2] += (smrcok.positions[:, 2].max() / 2) - (pt.positions[:, 2].max() / 2)

nanoparticle_ori = smrcok + pt

print('Pt positions min', pt.positions.min(0))
print('Pt positions max', pt.positions.max(0))

print('positions x min:', nanoparticle_ori.positions[:, 0].min(), 'max:', nanoparticle_ori.positions[:, 0].max())
print('positions y min:', nanoparticle_ori.positions[:, 1].min(), 'max:', nanoparticle_ori.positions[:, 1].max())
print('positions z min:', nanoparticle_ori.positions[:, 2].min(), 'max:', nanoparticle_ori.positions[:, 2].max())
# %% Delete the overlapping Al2O3 atoms
smrcokpos = smrcok.get_positions()
print(smrcok.positions.shape[0])
ptpos = pt.get_positions()

al2o2mask = np.zeros(smrcokpos.shape[0], dtype=bool)
for ali in range(smrcokpos.shape[0]):
    if ali % 1000 == 0:
        print(ali)
    for pti in range(ptpos.shape[0]):
        altemp = smrcokpos[ali, :]
        pttemp = ptpos[pti, :]
        d = np.sqrt((pttemp[0] - altemp[0]) ** 2 + (pttemp[1] - altemp[1]) ** 2 + (pttemp[2] - altemp[2]) ** 2)
        if d < 2:
            al2o2mask[ali] = True
#
del smrcok[np.where(al2o2mask)]
nanoparticle_ori = smrcok + pt

# %% Show the particle

abtem.show_atoms(
    nanoparticle_ori,
    plane="xy",  # show a view perpendicular to the 'xy' plane
    scale=0.5,  # scale atoms to 0.5 of their covalent radii; default is 0.75
    legend=True,  # show a legend with the atomic symbols
);

plt.show()
# %% Check the position boundry
print('positions x min:', nanoparticle_ori.positions[:, 0].min(), 'max:', nanoparticle_ori.positions[:, 0].max())
print('positions y min:', nanoparticle_ori.positions[:, 1].min(), 'max:', nanoparticle_ori.positions[:, 1].max())
print('positions z min:', nanoparticle_ori.positions[:, 2].min(), 'max:', nanoparticle_ori.positions[:, 2].max())
# %%
# ang = np.arange(-30, 33, 6)
ang = np.array([0])
phi_deg = ang
# phi_deg = ang + np.random.randn(1) - 1
# psi_deg = np.random.randn(ang.shape[0])
# theta_deg = np.random.randn(ang.shape[0]) * 0.5 - 0.5
# ang = ang_int + np.random.rand(1)

phi3 = th.as_tensor(np.deg2rad(phi_deg))
# psi3 = th.as_tensor(np.deg2rad(psi_deg))
# theta3 = th.as_tensor(np.deg2rad(theta_deg))
psi3 = th.zeros_like(phi3)
theta3 = th.zeros_like(phi3)

R = affine_matrix_3D_ZYX(phi3, theta3, psi3)

# np.save(f'/home/shengbo/PycharmProjects/nano_particles/test3_phi2.npy', phi3)
# np.save(f'/home/shengbo/PycharmProjects/nano_particles/test3_psi2.npy', psi3)
# np.save(f'/home/shengbo/PycharmProjects/nano_particles/test3_theta2.npy', theta3)
out_path = '/home/shengbo/PycharmProjects/nano_particles/'

for i in range(phi3.shape[0]):
    nanoparticle = nanoparticle_ori.copy()
    nanoparticle.center(vacuum=0.3)
    # x1 = nanoparticle.positions[:, 0].min()
    # x2 = nanoparticle.positions[:, 0].max()
    # y1 = nanoparticle.positions[:, 1].min()
    # y2 = nanoparticle.positions[:, 1].max()
    # z1 = nanoparticle.positions[:, 2].min()
    # z2 = nanoparticle.positions[:, 2].max()
    #
    # # x1 = 0
    # # x2 = 15
    # # y1 = 0
    # # y2 = 15
    # # z1 = 0
    # # z2 = 15
    #
    # xmid = (x1 + x2) / 2
    # ymid = (y1 + y2) / 2
    # zmid = (z1 + z2) / 2
    #
    # nanoparticle.positions[:, [0, 2]] = nanoparticle.positions[:, [2, 0]]
    # nanoparticle.positions = np.matmul(nanoparticle.positions, R[i, :, 0:3])
    # nanoparticle.positions[:, [2, 0]] = nanoparticle.positions[:, [0, 2]]
    #
    # xmid2 = (nanoparticle.positions[:, 0].min() + nanoparticle.positions[:, 0].max()) / 2
    # ymid2 = (nanoparticle.positions[:, 1].min() + nanoparticle.positions[:, 1].max()) / 2
    # zmid2 = (nanoparticle.positions[:, 2].min() + nanoparticle.positions[:, 2].max()) / 2
    # xdif = xmid - xmid2
    # ydif = ymid - ymid2
    # zdif = zmid - zmid2
    # nanoparticle.positions[:, 0] = nanoparticle.positions[:, 0] + xdif
    # nanoparticle.positions[:, 1] = nanoparticle.positions[:, 1] + ydif
    # nanoparticle.positions[:, 2] = nanoparticle.positions[:, 2] + zdif
    #
    # # cor3 = ase.Atoms("C", positions=[(x1, y1, z1)])
    # # cor4 = ase.Atoms("C", positions=[(x2, y2, z2)])
    # # nanoparticle = nanoparticle + cor3 + cor4
    #
    # maskx1 = nanoparticle.positions[:, 0] < x1
    # del nanoparticle[np.where(maskx1)[0]]
    # maskx2 = nanoparticle.positions[:, 0] > x2
    # del nanoparticle[np.where(maskx2)[0]]
    # masky1 = nanoparticle.positions[:, 1] < y1
    # del nanoparticle[np.where(masky1)[0]]
    # masky2 = nanoparticle.positions[:, 1] > y2
    # del nanoparticle[np.where(masky2)[0]]
    # maskz1 = nanoparticle.positions[:, 2] < z1
    # del nanoparticle[np.where(maskz1)[0]]
    # maskz2 = nanoparticle.positions[:, 2] > z2
    # del nanoparticle[np.where(maskz2)[0]]

    atom_pos = nanoparticle.get_positions()
    xyz_name = f'PtAl2O3_3nm_Rotation{ang[i]}degree.xyz'
    print(xyz_name)
    with open(out_path + xyz_name, 'w') as f:
        # f.writelines(f"PtAl2O3 nanoparticle coordinates, euler_angles = {np.rad2deg({ang[i]})}\n")
        f.writelines(f"PtAl2O3 nanoparticle coordinates\n")
        f.writelines(f"{nanoparticle.cell[0][0]:3.6f}\t{nanoparticle.cell[1][1]:3.6f}\t{nanoparticle.cell[2][2]:3.6f}\n")
        # f.writelines(f"{atom_pos[:, 0].max():3.6f}\t{atom_pos[:, 1].max():3.6f}\t{atom_pos[:, 2].max():3.6f}\n")
        for at in nanoparticle:
            f.writelines(f"{at.number}\t{at.position[0]:3.6f}\t{at.position[1]:3.6f}\t{at.position[2]:3.6f}\t1.0\t0.08\n")
        f.writelines(f"-1")
