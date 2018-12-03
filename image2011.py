# Import GDAL, NumPy, and matplotlib
from osgeo import gdal, gdal_array
import numpy as np
import scipy

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

#_____________IMAGE 2011_____________

b1_11_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2011/corrected/clip_B1_r.tif', gdal.GA_ReadOnly)
b1_11 = b1_11_ds.GetRasterBand(1).ReadAsArray()
b1_11_r = b1_11.ravel()

b2_11_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2011/corrected/clip_B2_r.tif', gdal.GA_ReadOnly)
b2_11 = b2_11_ds.GetRasterBand(1).ReadAsArray()
b2_11_r = b2_11.ravel()

b3_11_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2011/corrected/clip_B3_r.tif', gdal.GA_ReadOnly)
b3_11 = b3_11_ds.GetRasterBand(1).ReadAsArray()
b3_11_r = b3_11.ravel()

b4_11_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2011/corrected/clip_B4_r.tif', gdal.GA_ReadOnly)
b4_11 = b4_11_ds.GetRasterBand(1).ReadAsArray()
b4_11_r = b4_11.ravel()

b5_11_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2011/corrected/clip_B5_r.tif', gdal.GA_ReadOnly)
b5_11 = b5_11_ds.GetRasterBand(1).ReadAsArray()
b5_11_r = b5_11.ravel()

b7_11_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2011/corrected/clip_B7_r.tif', gdal.GA_ReadOnly)
b7_11 = b7_11_ds.GetRasterBand(1).ReadAsArray()
b7_11_r = b7_11.ravel()

ndvi_11_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2011/corrected/ndvi_r.tif', gdal.GA_ReadOnly)
ndvi_11 = ndvi_11_ds.GetRasterBand(1).ReadAsArray()
ndvi_11_r = ndvi_11.ravel()

#stack these bands
data_11 = np.array([b1_11_r,b2_11_r,b3_11_r,b4_11_r,b5_11_r,b7_11_r,ndvi_11_r])
predict_img_11 = data_11.transpose()

#saving the resulting array into .npy
a = "/Volumes/ga87rif/Study Project/satelite images/2011/img_11.npy"
np.save(a, predict_img_11)

#load
a = "/Volumes/ga87rif/Study Project/satelite images/2011/img_11.npy"
predict_img_11 = np.load(a)

gt_ds = gdal.Open('/Volumes/ga87rif/Study Project/Ground Truth Data/clip_data2011.tif', gdal.GA_ReadOnly)
gt = gt_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
gt_c = gt[0:3010,:]
data_11 = gt_c.ravel()

#saving the resulting array into .npy
y = "/Volumes/ga87rif/Study Project/Ground Truth Data/data_11.npy"
np.save(y, ref_data_11)

#load
y = "/Volumes/ga87rif/Study Project/Ground Truth Data/data_11.npy"
ref_data_11 = np.load(y)