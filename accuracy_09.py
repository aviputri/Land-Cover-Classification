import numpy as np
import pandas
from sklearn.metrics import accuracy_score, confusion_matrix

#import results
a = "/Volumes/ga87rif/Study Project/Result/predict_09_50trees_full.npy"
predict2009 = np.load(a)

#import ground truth data
data2009 = "/Volumes/ga87rif/Study Project/Ground Truth Data/data_09.npy"
data_09 = np.load(data2009)

#accuracy_score(y_true, y_pred)
b = accuracy_score(data_09, predict2009)
#calculate 
#confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
c = confusion_matrix(data_09, predict2009, labels=[1,2,3,4,5,6,7,8])

#print the result
print('The accuracy for the 2009 classification is: ', b)
print(c)