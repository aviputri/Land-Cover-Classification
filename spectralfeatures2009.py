# Import GDAL, NumPy, and matplotlib
from osgeo import gdal, gdal_array
import numpy as np
import scipy

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

#import the spectral features and save as Numpy Array
ndvi_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2009/Spectral features/ndvi_09.tif', gdal.GA_ReadOnly)
ndvi_ar = ndvi_ds.GetRasterBand(1).ReadAsArray()
ndvi = ndvi_ar.ravel()

ndwi_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2009/Spectral features/ndwi_09.tif', gdal.GA_ReadOnly)
ndwi_ar = ndwi_ds.GetRasterBand(1).ReadAsArray()
ndwi = ndwi_ar.ravel()

mndwi1_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2009/Spectral features/mndwi1_09.tif', gdal.GA_ReadOnly)
mndwi1_ar = mndwi1_ds.GetRasterBand(1).ReadAsArray()
mndwi1 = mndwi1_ar.ravel()

mndwi2_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2009/Spectral features/mndwi2_09.tif', gdal.GA_ReadOnly)
mndwi2_ar = mndwi2_ds.GetRasterBand(1).ReadAsArray()
mndwi2 = mndwi2_ar.ravel()

ndbi_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2009/Spectral features/ndbi_09.tif', gdal.GA_ReadOnly)
ndbi_ar = ndbi_ds.GetRasterBand(1).ReadAsArray()
ndbi = ndbi_ar.ravel()

mndbi_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2009/Spectral features/mndbi_09.tif', gdal.GA_ReadOnly)
mndbi_ar = mndbi_ds.GetRasterBand(1).ReadAsArray()
mndbi = mndbi_ar.ravel()

sf09_ar = np.array([ndvi,ndwi,mndwi1,mndwi2,ndbi,mndbi]) #2D array
sf09 = sf09_ar.transpose() # transposed for RF input

#saving the resulting array into .npy
a = "/Volumes/ga87rif/Study Project/satelite images/2009/sf09.npy"
np.save(a, sf09)

#load
a = "/Volumes/ga87rif/Study Project/satelite images/2009/sf09.npy"
sf09 = np.load(a)

#load reference data
b = "/Volumes/ga87rif/Study Project/Ground Truth Data/data_09.npy"
data09 = np.load(b)