# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/io.ipynb.

# %% auto 0
__all__ = ['read']

# %% ../nbs/io.ipynb 2
from .util.io import h5read
def read(filename):
    ending = filename.split('.')[-1]
    if ending == 'h5' or ending == 'hdf5':
        d = h5read(filename)
        data = d['data']
    return data
