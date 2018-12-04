import numpy as np
import pandas
from sklearn.metrics import accuracy_score, confusion_matrix

#import results
a = "/Volumes/ga87rif/Study Project/Result/predict_11_50trees_from09full.npy"
predict2011 = np.load(a)

#import ground truth data
data2011 = "/Volumes/ga87rif/Study Project/Ground Truth Data/data_11.npy"
data_11 = np.load(data2011)

#accuracy_score(y_true, y_pred)
b = accuracy_score(data_11, predict2011)
#calculate 
#confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
c = confusion_matrix(data_11, predict2011, labels=[1,2,3,4,5,6,7,8])

#print the result
print('The accuracy for the 2009 classification is: ', b)
print(c)