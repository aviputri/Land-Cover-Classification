#----------------------------------------------------------------------------------------
# TRAIN 2009
#----------------------------------------------------------------------------------------

import numpy as np
import scipy

#load
file2009 = "/home/aviputripertiwi/Study Project/Imagery/train_img_09.npy"
train_img_09 = np.load(file2009)

#--------------------------------make the reference data--------------------------------
#----------------------------------------------------------------------------------------
#load
data2009 = "/home/aviputripertiwi/Study Project/Reference/ref_data_09.npy"
ref_data_09 = np.load(data2009)

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier 

rf50 = RandomForestClassifier(n_estimators=50, oob_score=True)

#rf50 = rf50.fit(IMAGE, DATA)
rf50 = rf50.fit(train_img_09, ref_data_09)


#----------------------------------------------------------------------------------------
# SAVING THE RF MODEL
#----------------------------------------------------------------------------------------
# save the model to disk
import pandas
from sklearn.externals import joblib

filename = "/home/aviputripertiwi/Study Project/RF/rf_model_50.sav"
joblib.dump(rf50, filename)

