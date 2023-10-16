from .durham import DurhamDataset
from .spatial_dataset import SpatialDataset


def load(name, data_dir, window_size):
    if name == "durham":
        return DurhamDataset(data_dir, window_size)
    else:
        raise ValueError("Dataset {} not supported".format(name))
