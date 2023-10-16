import os
import numpy as np
from scipy.signal import convolve2d

import rasterio as rio

import multiprocessing as mp


class SpatialDataset:
    def __init__(
        self,
        dataset_path,
        nlcd_ws=20,
        ndvi_ws=10,
        albedo_ws=10,
        ndvi_ls=5,
        albedo_ls=5,
    ):
        # Get the NDVI, Land Cover, and Temperature Data
        with rio.open(os.path.join(dataset_path, "durham_nlcd.tif")) as src:
            nlcd = src.read(1)

        with rio.open(os.path.join(dataset_path, "durham_temp.tif")) as src:
            temp = src.read(1)

        with rio.open(os.path.join(dataset_path, "durham_ndvi.tif")) as src:
            ndvi = src.read(1)
        ndvi = np.flipud(ndvi)

        with rio.open(os.path.join(dataset_path, "durham_albedo.tif")) as src:
            albedo = src.read(1)
        # For some reason, this is upside down...let's fix that.
        albedo = np.flipud(albedo)

        assert ndvi.shape == albedo.shape
        assert nlcd.shape == ndvi.shape
        assert temp.shape == ndvi.shape

        self.coords = np.array(np.where(temp != 0)).T

        self.temp = temp
        self.nlcd = self._calc_nlcd_percentage(nlcd, nlcd_ws)
        self.ndvi = ndvi
        self.albedo = albedo

        assert ndvi.shape == albedo.shape
        assert nlcd.shape == ndvi.shape

        # Get weights
        self.ndvi_weight = self._calc_weight_matrix(ndvi_ws, ndvi_ls)
        self.albedo_weight = self._calc_weight_matrix(albedo_ws, albedo_ls)

        # Get spatial terms
        self.ndvi_spatial = self._calc_spatial_term(self.ndvi, self.ndvi_weight)
        self.albedo_spatial = self._calc_spatial_term(self.albedo, self.albedo_weight)

        # # Normalize all of the features
        # self.temp = self._normalize(self.temp)
        # self.ndvi = self._normalize(self.ndvi)
        # self.albedo = self._normalize(self.albedo)
        # self.ndvi_spatial = self._normalize(self.ndvi_spatial)
        # self.albedo_spatial = self._normalize(self.albedo_spatial)

    def _normalize(self, data):
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def _calc_spatial_term(self, data, weight):
        return convolve2d(data, weight, mode="same")

    def _calc_weight_matrix(self, window_size, length_scale):
        dist_matrix = np.sqrt(
            np.arange(-window_size, window_size + 1)[np.newaxis, :] ** 2
            + np.arange(-window_size, window_size + 1)[:, np.newaxis] ** 2
        )
        weight_matrix = np.exp(-dist_matrix / length_scale)
        weight_matrix[window_size, window_size] = 0
        weight_matrix /= weight_matrix.sum()
        return weight_matrix

    def _create_one_hot_encoding(self, nlcd):
        # Creating the spatially aggregated one-hot encoding
        cat = len(np.unique(nlcd))
        # Initialize the one-hot encoding vector as zeros.
        one_hot = np.zeros((nlcd.shape[0], nlcd.shape[1], cat))

        # Set the values of the one-hot encoding based on the land use data.
        for i, segment in enumerate(np.unique(nlcd)):
            one_hot[:, :, i] = (nlcd == segment).astype(int)

        return one_hot

    def _task(self, args):
        weight = convolve2d(args[0], args[1], mode="same")
        # weight = (weight - np.mean(weight)) / (np.std(weight) + 1e-8)
        return weight

    def _calc_nlcd_percentage(self, nlcd, window_size):
        one_hot = self._create_one_hot_encoding(nlcd)

        weight_matrix = np.ones((window_size * 2 + 1, window_size * 2 + 1))
        weight_matrix /= weight_matrix.sum()

        # Define the iterable
        iterable = [(one_hot[:, :, i], weight_matrix) for i in range(one_hot.shape[2])]

        nlcd_weight = np.zeros_like(one_hot)
        with mp.Pool(8) as pool:
            for i, result in enumerate(pool.map(self._task, iterable)):
                nlcd_weight[:, :, i] = result

        return nlcd_weight

    def get_data(self):
        return self.temp, self.nlcd, self.ndvi, self.albedo
