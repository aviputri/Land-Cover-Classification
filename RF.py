# Import GDAL, NumPy, and matplotlib
from osgeo import gdal, gdal_array
import numpy as np
import scipy
#import matplotlib.pyplot as plt
#%matplotlib inline

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()


#----------------------------------------------------------------------------------------
# PREPARING THE TRAINING DATASETS
#----------------------------------------------------------------------------------------

# Read in our image and ROI image (2009)

# the stacked image has an error, so I'm commenting this one out

#img_ds = gdal.Open('/Volumes/ga87rif/Study Project/satelite images/2009/stack.tif', gdal.GA_ReadOnly)
#img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount), gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
#for b in range(img.shape[2]):
#    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

# I import the satellite bands one by one instead

b1_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/clip_B1_r.tif', gdal.GA_ReadOnly)
b1 = b1_ds.GetRasterBand(1).ReadAsArray()
b1_r = b1.ravel()

b2_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/clip_B2_r.tif', gdal.GA_ReadOnly)
b2 = b2_ds.GetRasterBand(1).ReadAsArray()
b2_r = b2.ravel()

b3_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/clip_B3_r.tif', gdal.GA_ReadOnly)
b3 = b3_ds.GetRasterBand(1).ReadAsArray()
b3_r = b3.ravel()

b4_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/clip_B4_r.tif', gdal.GA_ReadOnly)
b4 = b4_ds.GetRasterBand(1).ReadAsArray()
b4_r = b4.ravel()

b5_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/clip_B5_r.tif', gdal.GA_ReadOnly)
b5 = b5_ds.GetRasterBand(1).ReadAsArray()
b5_r = b5.ravel()

b7_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/clip_B7_r.tif', gdal.GA_ReadOnly)
b7 = b7_ds.GetRasterBand(1).ReadAsArray()
b7_r = b7.ravel()

ndvi_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/clip_ndvi_r.tif', gdal.GA_ReadOnly)
ndvi = ndvi_ds.GetRasterBand(1).ReadAsArray()
ndvi_r = ndvi.ravel()

#stack these bands
data = np.array([b1_r,b2_r,b3_r,b4_r,b5_r,b7_r,ndvi_r])
#transpose to match per collumn with the training data
train_img = data.transpose()

#read the reference data

roi_ds = gdal.Open('/Volumes/ga87rif/Study Project/training datasets/clip_data2009.tif', gdal.GA_ReadOnly)
roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
#ref = roi[0:3010,:]
train_roi = roi.ravel()


#----------------------------------------------------------------------------------------
# Find how many non-zero entries we have -- i.e. how many training data samples?
#n_samples = roi.sum()
#print('We have {n} samples'.format(n=n_samples))
 
# What are our classification labels?
#labels = np.unique(roi[roi > 0])
#print('The training data include {n} classes: {classes}'.format(n=labels.size, classes=labels))


# We will need a "X" matrix containing our features, and a "y" array containing our labels
#     These will have n_samples rows
#     In other languages we would need to allocate these and them loop to fill them, but NumPy can be faster
 
#X = img #[roi > 0, :]
#y = roi #[roi > 0]

#----------------------------------------------------------------------------------------
# I DON'T USE FMASK SO I WILL DITCH THIS PART THO
#----------------------------------------------------------------------------------------
# Mask out clouds, cloud shadows, and snow using Fmask
#clear = X[:, 7] <= 1
 
#X = X[clear, :7]  # we can ditch the Fmask band now
#y = y[clear]

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier # make sure to install numpe 1.11.2 otherwise this won't work
 
# Initialize our model with 500 trees
# Object rf dibuat dan diinisialisasi 
rf = RandomForestClassifier(n_estimators=100, oob_score=True)

#with n_estimators=10:
#UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable oob estimates.
#RuntimeWarning: invalid value encountered in true_divide
#predictions[k].sum(axis=1)[:, np.newaxis])

#n_estimators=50 works --> 13.39 GB
#n_estimators=100 works --> 26.57 GB
 
# Fit our model to training data
#(Object rf di-train dgn setdata image X dan label kelas y)
rf = rf.fit(train_img, train_roi)


#----------------------------------------------------------------------------------------
# SAVING THE RF MODEL
#----------------------------------------------------------------------------------------
# save the model to disk
import pandas
from sklearn.externals import joblib

filename = "/Users/aviputripertiwi/Documents/TU Munchen/Study Project/rf_model_100.sav"
joblib.dump(rf, filename)


#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

# Read in new image (new data)

#img_ds = gdal.Open('/Users/aviputripertiwi/Documents/TU Munchen/Study Project/Images/LS7_09_stack.tif', gdal.GA_ReadOnly)
#img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount), gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
 

# Reshape

#new_shape = (img.shape[0] * img.shape[1], img.shape[2] - 1)
#img_as_array = img[:, :, :7].reshape(new_shape)
 
# Predict the classes (labels) of the new data. 
# Hasilnya tersimpan di array class_prediction
#class_prediction = rf.predict(img_as_array)
#class_prediction = class_prediction.reshape(img[:, :, 0].shape)

#----------------------------------------------------------------------------------------
# PREDICT 2009
#----------------------------------------------------------------------------------------
# Import GDAL, NumPy, and matplotlib
from osgeo import gdal, gdal_array
import numpy as np
import scipy
#import matplotlib.pyplot as plt
#%matplotlib inline

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

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
#transpose to match per collumn with the training data
predict_img_09 = data_09.transpose()

#predit 2009 image
predict2009 = rf.predict(predict_img_09)

#saving the resulting array into .npy
file2009 = "/Users/aviputripertiwi/Documents/TU Munchen/Study Project/result_array_2009_50.npy"
np.save(file2009, predict2009)

#load
file2009 = "/Users/aviputripertiwi/Documents/TU Munchen/Study Project/result_array_2009_50.npy"
predict2009 = np.load(file2009)

#----------------------------------------------------------------------------------------
# LOADING THE RF MODEL
#----------------------------------------------------------------------------------------
import pandas
from sklearn.externals import joblib

filename = "/Users/aviputripertiwi/Documents/TU Munchen/Study Project/rf_model_100.sav"
loaded_model = joblib.load(filename)

predict2009 = loaded_model.predict(predict_img_09)


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

#----------------------------------------------------------------------------------------
# PREDICT 2011
#----------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------
# PREDICT 2017
#----------------------------------------------------------------------------------------

