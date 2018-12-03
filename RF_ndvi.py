from osgeo import gdal, gdal_array
import numpy as np
import scipy

gdal.UseExceptions()
gdal.AllRegister()

ndvi_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/clip_ndvi_r.tif', gdal.GA_ReadOnly)
ndvi = ndvi_ds.GetRasterBand(1).ReadAsArray()
ndvi_r = ndvi.ravel()

ndvi_reshape = ndvi_r.reshape(-1,1)

roi_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/clip_data2009.tif', gdal.GA_ReadOnly)
roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
#ref = roi[0:3010,:]
train_roi = roi.ravel()

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, oob_score=True)

rf = rf.fit(ndvi_reshape, train_roi)


ndvi_09_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2009/corrected/true_NDVI.tif', gdal.GA_ReadOnly)
ndvi_09 = ndvi_09_ds.GetRasterBand(1).ReadAsArray()
ndvi_09_r = ndvi_09.ravel()

ndvi_09_re = ndvi_09_r.reshape(-1,1)

predict2009 = rf.predict(ndvi_09_re)

#----------------------------------------------------------------------------------------
# calculating the accuracy (2009) --> OVERALL ACCURACY
#----------------------------------------------------------------------------------------
#open the ground truth data

gt_ds = gdal.Open('/Volumes/ga87rif/Study Project/Ground Truth Data/clip_data2009.tif', gdal.GA_ReadOnly)
gt = gt_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
gt_c = gt[0:3010,:]
gt_data = gt_c.ravel()

#calculate 
from sklearn.metrics import accuracy_score

#accuracy_score(y_true, y_pred)
accuracy_score(gt_data, predict2009)

#saving the resulting array into .npy
file2009 = "/Users/aviputripertiwi/Documents/TU Munchen/Study Project/result_ndvi09_100.npy"
np.save(file2009, predict2009)

#load
file2009 = "/Users/aviputripertiwi/Documents/TU Munchen/Study Project/result_ndvi09_100.npy"
predict2009 = np.load(file2009)