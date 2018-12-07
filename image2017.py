# Import GDAL, NumPy, and matplotlib
from osgeo import gdal, gdal_array
import numpy as np
import scipy

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

b2_17_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2017/Sentinel 2/clip_b2_r.tif', gdal.GA_ReadOnly)
b2_17 = b2_17_ds.GetRasterBand(1).ReadAsArray()
b2_17 = b2_17[20:9042,20:9187]
b2_17_r = b2_17.ravel()

b3_17_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2017/Sentinel 2/clip_b3_r.tif', gdal.GA_ReadOnly)
b3_17 = b3_17_ds.GetRasterBand(1).ReadAsArray()
b3_17 = b3_17[20:9042,20:9187]
b3_17_r = b3_17.ravel()

b4_17_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2017/Sentinel 2/clip_b4_r.tif', gdal.GA_ReadOnly)
b4_17 = b4_17_ds.GetRasterBand(1).ReadAsArray()
b4_17 = b4_17[20:9042,20:9187]
b4_17_r = b4_17.ravel()

b8_17_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2017/Sentinel 2/clip_b8_r.tif', gdal.GA_ReadOnly)
b8_17 = b8_17_ds.GetRasterBand(1).ReadAsArray()
b8_17 = b8_17[20:9042,20:9187]
b8_17_r = b8_17.ravel()

b11_17_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2017/Sentinel 2/clip_b11_r.tif', gdal.GA_ReadOnly)
b11_17 = b11_17_ds.GetRasterBand(1).ReadAsArray()
b11_17 = b11_17[20:9042,20:9187]
b11_17_r = b11_17.ravel()

b12_17_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2017/Sentinel 2/clip_b12_r.tif', gdal.GA_ReadOnly)
b12_17 = b12_17_ds.GetRasterBand(1).ReadAsArray()
b12_17 = b12_17[20:9042,20:9187]
b12_17_r = b12_17.ravel()

ndvi_17_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2017/Sentinel 2/ndvi_r.tif', gdal.GA_ReadOnly)
ndvi_17 = ndvi_17_ds.GetRasterBand(1).ReadAsArray()
ndvi_17 = ndvi_17[20:9042,20:9187]
ndvi_17_r = ndvi_17.ravel()

#stack these bands
data_17 = np.array([b2_17_r,b3_17_r,b4_17_r,b8_17_r,b11_17_r,b12_17_r, ndvi_17_r])
predict_img_17 = data_17.transpose()

#saving the resulting array into .npy
a = "/Volumes/ga87rif/Study Project/satelite images/2017/img_17.npy"
np.save(a, predict_img_17)

#load
a = "/Volumes/ga87rif/Study Project/satelite images/2017/img_17.npy"
predict_img_17 = np.load(a)
