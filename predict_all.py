# Import GDAL, NumPy, and matplotlib
from osgeo import gdal, gdal_array
import numpy as np
import scipy
from sklearn.ensemble import RandomForestClassifier 
import pandas
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()


file2009 = "/Volumes/ga87rif/Study Project/satelite images/2009/img_09.npy"
predict_img_09 = np.load(file2009)

data2009 = "/Volumes/ga87rif/Study Project/Ground Truth Data/data_09.npy"
data_09 = np.load(data2009)

rf = RandomForestClassifier(n_estimators=50, oob_score=True)
rf = rf.fit(predict_img_09, data_09)

filename = "/Users/aviputripertiwi/Documents/TU Munchen/Study Project/rf_model_50_09_full.sav"
joblib.dump(rf, filename)

predict2009 = rf.predict(predict_img_09)

#saving the resulting array into .npy
a = "/Volumes/ga87rif/Study Project/Result/predict_09_50trees_full.npy"
np.save(a, predict2009)

#load
a = "/Volumes/ga87rif/Study Project/Result/predict_09_50trees_full.npy"
predict2009 = np.load(a)


from sklearn.metrics import accuracy_score, confusion_matrix

#accuracy_score(y_true, y_pred)
accuracy_score(data_09, predict2009)
#calculate 
#confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
confusion_matrix(data_09, predict2009, labels=[1,2,3,4,5,6,7,8])

a = "/Volumes/ga87rif/Study Project/satelite images/2011/img_11.npy"
predict_img_11 = np.load(a)

predict2011 = rf.predict(predict_img_11)

#saving the resulting array into .npy
a = "/Volumes/ga87rif/Study Project/Result/predict_11_50trees_from09full.npy"
np.save(a, predict2011)

#load
a = "/Volumes/ga87rif/Study Project/Result/predict_11_50trees_from09full.npy"
predict2011 = np.load(a)

#accyracy
y = "/Volumes/ga87rif/Study Project/Ground Truth Data/data_11.npy"
ref_data_11 = np.load(y)

#accuracy_score(y_true, y_pred)
accuracy_score(ref_data_11, predict2011) 
#0.4249168348444885

#calculate 
#confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
confusion_matrix(ref_data_11, predict2011, labels=[1,2,3,4,5,6,7,8])

#-------------------------------------
a = "/Volumes/ga87rif/Study Project/satelite images/2017/img_17.npy"
predict_img_17 = np.load(a)

predict2017 = rf.predict(predict_img_17)

#saving the resulting array into .npy
a = "/Volumes/ga87rif/Study Project/Result/predict_17_50trees_from09full.npy"
np.save(a, predict2017)

#load
a = "/Volumes/ga87rif/Study Project/Result/predict_17_50trees_from09full.npy"
predict2017 = np.load(a)

#===========PREDIT FROM 2017 TRAINING=============
a = "/Volumes/ga87rif/Study Project/satelite images/2011/img_11.npy"
predict_img_11 = np.load(a)

y = "/Volumes/ga87rif/Study Project/Ground Truth Data/data_11.npy"
ref_data_11 = np.load(y)

rf2 = RandomForestClassifier(n_estimators=50, oob_score=True)
rf2 = rf2.fit(predict_img_11, ref_data_11)

filename = "/Users/aviputripertiwi/Documents/TU Munchen/Study Project/rf_model_50_11_full.sav"
joblib.dump(rf2, filename)

predict2011 = rf2.predict(predict_img_11)

#saving the resulting array into .npy
x = "/Volumes/ga87rif/Study Project/Result/predict_11_50trees_from11full.npy"
np.save(x, predict2011)

#load
x = "/Volumes/ga87rif/Study Project/Result/predict_11_50trees_from11full.npy"
predict2011 = np.load(x)

#----
#load
a = "/Volumes/ga87rif/Study Project/satelite images/2017/img_17.npy"
predict_img_17 = np.load(a)

filename = "/Users/aviputripertiwi/Documents/TU Munchen/Study Project/rf_model_50_11_full.sav"
rf2 = joblib.load(filename)

predict2011 = rf2.predict(predict_img_17)

