import os
import numpy as np
import rasterio as rio


class DurhamDataset():
    
    def __init__(self, dataset_path, window_size):
        
        # Get the NDVI, Land Cover, and Temperature Data
        with rio.open(os.path.join(dataset_path, 'durham_nlcd.tif')) as src:
            nlcd = src.read(1)
            
        self.nlcd = self._create_one_hot_encoding(nlcd)

        with rio.open(os.path.join(dataset_path, 'durham_temp.tif')) as src:
            self.temp = src.read(1)

        with rio.open(os.path.join(dataset_path, 'durham_ndvi.tif')) as src:
            ndvi = src.read(1)

        self.ndvi = np.flipud(ndvi)

        with rio.open(os.path.join(dataset_path, 'durham_albedo.tif')) as src:
            albedo = src.read(1)
        # For some reason, this is upside down...let's fix that.
        self.albedo = np.flipud(albedo)

        self.window_size = window_size
        self.coords = self.get_coords()

    def _create_one_hot_encoding(self, land_use):
        # Get the number of categories based on the number of unique values
        # in the land use data.
        cat = len(np.unique(land_use))
        # Initialize the one-hot encoding vector as zeros.
        one_hot = np.zeros((land_use.shape[0], land_use.shape[1], cat))

        # Set the values of the one-hot encoding based on the land use data.
        for i, segment in enumerate(np.unique(land_use)):
            one_hot[:, :, i] = (land_use == segment).astype(int)

        return one_hot
    
    
    def get_coords(self):
        coords = np.array(np.where(self.temp != 0)).T
        return coords
    
    def __getitem__(self, idx):
        i, j = self.coords[idx]

        nlcd = self.nlcd[
            i - self.window_size : i + self.window_size + 1,
            j - self.window_size : j + self.window_size + 1,
            :
        ]
        
        ndvi = self.ndvi[
            i - self.window_size : i + self.window_size + 1,
            j - self.window_size : j + self.window_size + 1,
        ]
        albedo = self.albedo[
            i - self.window_size : i + self.window_size + 1,
            j - self.window_size : j + self.window_size + 1,
        ]
        
        temp = self.temp[i, j]
        
        return i, j, nlcd, ndvi, albedo, temp
        