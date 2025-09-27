import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from project import mymodel
from project.models.trainModles import trainModels
from project.mymodel import predict

X_load = np.load('project/dataFiles/X_train.npy')
y_load = np.load('project/dataFiles/Y_train.npy')
#X_train, y_train = train_test_split(X_load, y_load)
X_train, X_test, y_train, y_test = train_test_split(X_load, y_load, test_size=0.2, random_state=42)
print(X_load.shape)
print(y_load.shape)
trainModels(X_train, y_train)
#bestmodel(X_test, y_test)
#print( mymodel.predict(X_train))
# Avaliaçao Ridge
print("\n[RIDGE]")
print("MAE :", mean_absolute_error(mymodel.predict(X_test), y_test))
print("Mean Squared Error:", mean_squared_error(mymodel.predict(X_test), y_test))
print("R²  :", r2_score(mymodel.predict(X_test), y_test))
