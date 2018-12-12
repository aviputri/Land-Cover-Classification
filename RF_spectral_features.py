import numpy as np
from sklearn.ensemble import RandomForestClassifier 
import pandas
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

#training 2009
rf = RandomForestClassifier(n_estimators=50, oob_score=True)
rf = rf.fit(sf09, data09)

#save
filename = "/Users/aviputripertiwi/Documents/TU Munchen/Study Project/rf_sf.sav"
joblib.dump(rf, filename)

#predict 2009
result09 = rf.predict(sf09)

#saving the resulting array into .npy
a = "/Volumes/ga87rif/Study Project/Result/predict_09_from_SF.npy"
np.save(a, result09)

#accuracy_score(y_true, y_pred)
accuracy_score(data09, result09)
#calculate 
#confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
confusion_matrix(data09, result09, labels=[1,2,3,4,5,6,7,8])

#predict 2011
result11 = rf.predict(sf11)

#saving the resulting array into .npy
a = "/Volumes/ga87rif/Study Project/Result/predict_11_from_SF.npy"
np.save(a, result11)

#load 2011 reference data
y = "/Volumes/ga87rif/Study Project/Ground Truth Data/data_11.npy"
data11 = np.load(y)

#accuracy_score(y_true, y_pred)
accuracy_score(data11, result11)
#calculate 
#confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
confusion_matrix(data11, result11, labels=[1,2,3,4,5,6,7,8])