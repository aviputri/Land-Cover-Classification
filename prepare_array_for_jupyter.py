# Import GDAL, NumPy, and matplotlib
from osgeo import gdal, gdal_array
import numpy as np
import scipy

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

#_____________Training 2009_____________
b1_09_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/2009/clip_clip_B1_r.tif', gdal.GA_ReadOnly)
b1_09 = b1_09_ds.GetRasterBand(1).ReadAsArray()
b1_09 = b1_09[0:2292,0:2090]
b1_09_r = b1_09.ravel()

b2_09_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/2009/clip_clip_B2_r.tif', gdal.GA_ReadOnly)
b2_09 = b2_09_ds.GetRasterBand(1).ReadAsArray()
b2_09 = b2_09[0:2292,0:2090]
b2_09_r = b2_09.ravel()

b3_09_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/2009/clip_clip_B3_r.tif', gdal.GA_ReadOnly)
b3_09 = b3_09_ds.GetRasterBand(1).ReadAsArray()
b3_09 = b3_09[0:2292,0:2090]
b3_09_r = b3_09.ravel()

b4_09_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/2009/clip_clip_B4_r.tif', gdal.GA_ReadOnly)
b4_09 = b4_09_ds.GetRasterBand(1).ReadAsArray()
b4_09 = b4_09[0:2292,0:2090]
b4_09_r = b4_09.ravel()

b5_09_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/2009/clip_clip_B5_r.tif', gdal.GA_ReadOnly)
b5_09 = b5_09_ds.GetRasterBand(1).ReadAsArray()
b5_09 = b5_09[0:2292,0:2090]
b5_09_r = b5_09.ravel()

b7_09_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/2009/clip_clip_B7_r.tif', gdal.GA_ReadOnly)
b7_09 = b7_09_ds.GetRasterBand(1).ReadAsArray()
b7_09 = b7_09[0:2292,0:2090]
b7_09_r = b7_09.ravel()

ndvi_09_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/2009/clip_true_NDVI.tif', gdal.GA_ReadOnly)
ndvi_09 = ndvi_09_ds.GetRasterBand(1).ReadAsArray()
ndvi_09 = ndvi_09[0:2292,0:2090]
ndvi_09_r = ndvi_09.ravel()

#stack these bands
data_09 = np.array([b1_09_r,b2_09_r,b3_09_r,b4_09_r,b5_09_r,b7_09_r,ndvi_09_r])
train_img_09 = data_09.transpose()

#saving the resulting array into .npy
file2009 = "/Volumes/ga87rif/Study Project/training datasets/train_img_09.npy"
np.save(file2009, train_img_09)

#load
file2009 = "/Volumes/ga87rif/Study Project/training datasets/train_img_09.npy"
train_img_09 = np.load(file2009)



gt_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/2009/clip_clip_data2009.tif', gdal.GA_ReadOnly)
gt = gt_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
gt = gt[0:2292,0:2090]
ref_data_09 = gt.ravel()

#saving the resulting array into .npy
data2009 = "/Volumes/ga87rif/Study Project/training datasets/ref_data_09.npy"
np.save(data2009, ref_data_09)

#load
data2009 = "/Volumes/ga87rif/Study Project/training datasets/ref_data_09.npy"
ref_data_09 = np.load(data2009)


#_____________IMAGE 2009_____________

b1_09_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2009/corrected/clip_B1_r.tif', gdal.GA_ReadOnly)
b1_09 = b1_09_ds.GetRasterBand(1).ReadAsArray()
b1_09_r = b1_09.ravel()

b2_09_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2009/corrected/clip_B2_r.tif', gdal.GA_ReadOnly)
b2_09 = b2_09_ds.GetRasterBand(1).ReadAsArray()
b2_09_r = b2_09.ravel()

b3_09_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2009/corrected/clip_B3_r.tif', gdal.GA_ReadOnly)
b3_09 = b3_09_ds.GetRasterBand(1).ReadAsArray()
b3_09_r = b3_09.ravel()

b4_09_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2009/corrected/clip_B4_r.tif', gdal.GA_ReadOnly)
b4_09 = b4_09_ds.GetRasterBand(1).ReadAsArray()
b4_09_r = b4_09.ravel()

b5_09_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2009/corrected/clip_B5_r.tif', gdal.GA_ReadOnly)
b5_09 = b5_09_ds.GetRasterBand(1).ReadAsArray()
b5_09_r = b5_09.ravel()

b7_09_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2009/corrected/clip_B7_r.tif', gdal.GA_ReadOnly)
b7_09 = b7_09_ds.GetRasterBand(1).ReadAsArray()
b7_09_r = b7_09.ravel()

ndvi_09_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2009/corrected/true_NDVI.tif', gdal.GA_ReadOnly)
ndvi_09 = ndvi_09_ds.GetRasterBand(1).ReadAsArray()
ndvi_09_r = ndvi_09.ravel()

#stack these bands
data_09 = np.array([b1_09_r,b2_09_r,b3_09_r,b4_09_r,b5_09_r,b7_09_r,ndvi_09_r])
train_img_09 = data_09.transpose()

#saving the resulting array into .npy
file2009 = "/Volumes/ga87rif/Study Project/satelite images/2009/img_09.npy"
np.save(file2009, train_img_09)

#load
file2009 = "/Volumes/ga87rif/Study Project/satelite images/2009/img_09.npy"
predict_img_09 = np.load(file2009)



gt_ds = gdal.Open('/Volumes/ga87rif/Study Project/Ground Truth Data/clip_data2009.tif', gdal.GA_ReadOnly)
gt = gt_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
gt_c = gt[0:3010,:]
ref_data_09 = gt_c.ravel()

#saving the resulting array into .npy
data2009 = "/Volumes/ga87rif/Study Project/Ground Truth Data/data_09.npy"
np.save(data2009, ref_data_09)

#load
data2009 = "/Volumes/ga87rif/Study Project/Ground Truth Data/data_09.npy"
data_09 = np.load(data2009)

#________________iseng nyoba di laptop
from sklearn.ensemble import RandomForestClassifier 

rf = RandomForestClassifier(n_estimators=50, oob_score=True)
rf = rf.fit(train_img_09, ref_data_09)

# save the model to disk
import pandas
from sklearn.externals import joblib

filename = "/Users/aviputripertiwi/Documents/TU Munchen/Study Project/rf_model_50_09.sav"
joblib.dump(rf, filename)


#______________iseng predict di laptop

file2009 = "/Volumes/ga87rif/Study Project/satelite images/2009/img_09.npy"
predict_img_09 = np.load(file2009)

predict2009 = rf.predict(predict_img_09)

#saving the resulting array into .npy
a = "/Volumes/ga87rif/Study Project/Result/predict_09_50trees_cropped.npy"
np.save(a, predict2009)

#load
a = "/Volumes/ga87rif/Study Project/Result/predict_09_50trees_cropped.npy"
predict2009 = np.load(a)

#cek akurasi
#open the ground truth data

data2009 = "/Volumes/ga87rif/Study Project/Ground Truth Data/data_09.npy"
data_09 = np.load(data2009)

from sklearn.metrics import accuracy_score
#accuracy_score(y_true, y_pred)
accuracy_score(data_09, predict2009)#calculate 



#confusion matrix
from sklearn.metrics import confusion_matrix

#confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
confusion_matrix(data_09, predict2009, labels=[1,2,3,4,5,6,7,8])

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