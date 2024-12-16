# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/data.ipynb.

# %% auto 0
__all__ = ['RasterScanningDiffractionDataset']

# %% ../nbs/data.ipynb 2
from torch.utils.data import Dataset, DataLoader
import torch as th

class RasterScanningDiffractionDataset(Dataset):
    def __init__(self, measurements: th.tensor, probe_index: int, angles_index: int, translation_index: int,
                 start_end_index: int):
        # if len(measurements.shape) < 4:
        #     raise RuntimeError('shape must be 4-dim')
        self.data = measurements
        # ms = measurements.shape
        # self.data_3d = measurements.view(ms[0] * ms[1], ms[2], ms[3])
        self.probe_index = probe_index
        self.angles_index = angles_index
        self.translation_index = translation_index
        self.start_end_index = start_end_index

    def __len__(self):
        return self.data.shape[0]  

    def __getitem__(self, item):
        """
        Expects batched indices in item, so a List
        Expects a 7-tuple as output (batch_index, probe_index, angles_index, r_indices, translation_index, start_end_index, amplitudes_target)
        :param item:
        :return:
        """
        r_indices = item
        return (item[0],
                self.probe_index,
                self.angles_index,
                r_indices,
                self.translation_index,
                self.start_end_index,
                self.data[item])
