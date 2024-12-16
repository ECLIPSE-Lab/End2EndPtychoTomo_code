from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import numpy as np

angles = np.array([0])

# %%

import pyprismatic as pypr

m = pypr.Metadata()

m.realspacePixelSizeX = 0.05
m.realspacePixelSizeY = 0.05
m.interpolationFactorX = 1
m.interpolationFactorY = 1

m.potBound = 1
m.numFP = 1
m.sliceThickness = 0.05
m.numSlices = 0
m.zStart = 0

m.tileX = 1
m.tileY = 1
m.tileZ = 1

m.E0 = 200
m.alphaBeamMax = 25

m.numGPUs = 1
m.numStreamsPerGPU = 4

m.numThreads = 16
m.batchSizeTargetCPU = 256
m.batchSizeTargetGPU = 256
m.earlyCPUStopCount = 10

m.probeStepX = 10
m.probeStepY = 10

m.probeDefocus = -30
m.C3 = 0
m.probeSemiangle = 25
m.integrationAngleMin = 30
m.integrationAngleMax = 200

m.detectorAngleStep = 2
m.probeXtilt = 0
m.probeYtilt = 0
m.scanWindowXMin = 0.51
m.scanWindowXMax = 0.52
m.scanWindowYMin = 0.51
m.scanWindowYMax = 0.52

m.algorithm = 'prism'
m.includeThermalEffects = True
m.alsoDoCPUWork = False
m.save2DOutput = True
m.save3DOutput = False
m.save4DOutput = True
m.saveDPC_CoM = False
m.savePotentialSlices = True
# m.savePotentialSlices = False
m.nyquistSampling = False
m.saveProbe = True
m.maxFileSize = 35*10**9
# m.potential3D = True

m.transferMode = 'auto'

# %%
import time
from tqdm import trange

out_path = '/home/shengbo/PycharmProjects/nano_particles/'

rsind = 0
for i in trange(angles.shape[0]):
    ang = angles[i]
    prism_output_name = f'PtAl2O3_3nm_{m.numFP}FP.h5'
    axis_string = 'YXZ'

    m.filenameAtoms = f'/home/shengbo/PycharmProjects/nano_particles/NanoParticle_3nm_{ang}degree.xyz'
    m.filenameOutput = out_path + prism_output_name

    s = time.time()
    m.go()
    print(f'{i:03d} angle: {ang:2.2f} PRISM took {time.time() - s}')

# %%
import h5py
import matplotlib.pyplot as plt
f = h5py.File(out_path + prism_output_name, 'r')
dset = f['4DSTEM_simulation']
fulldata = dset['data']
ptychodata = fulldata['datacubes']['CBED_array_depth0000']['data'][:, :, :, :]
adfdata = fulldata['realslices']['annular_detector_depth0000']['data'][:, :]
# probe = fulldata['diffractionslices']['probe']['data'][:, :]

# %%
pot = []
for pi in range(m.numFP):
    potslice = fulldata['realslices'][f'ppotential_fp000{pi}']['data'][:, :, :]
    pot.append(potslice)

np.save(f'/home/shengbo/PycharmProjects/nano_particles/PtAl2O3_3nm_Oxygen_RawVol_{m.numFP}FP.npy', pot)
print(len(pot))
# %%
fig, ax = plt.subplots()
imax = ax.imshow(ptychodata[1][1])
plt.colorbar(imax)
plt.show()

fig, ax = plt.subplots()
imax = ax.imshow(adfdata[:, :])
plt.colorbar(imax)
plt.show()