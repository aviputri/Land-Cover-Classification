from sklearn.ensemble import RandomForestClassifier

X = [[1,3,4,7,9],[10,11,17,26,30],[40,45,20,14,2]]

Y = [[1,1,1,1,1],[1,2,2,3,4],[5,5,2,2,1]]

rf = rf.fit(X, Y)

F = [[3,4,5,6,7],[10,10,15,17,18],[42,43,25,26,1]]

predict = rf.predict(F)