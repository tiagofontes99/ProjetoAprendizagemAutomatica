import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from project.models.trainModles import trainModels
from project.mymodel import predict


X_load = np.load('project/dataFiles/X_train.npy')
y_load = np.load('project/dataFiles/Y_train.npy')
X_train, X_test, y_train, y_test = train_test_split(X_load, y_load, test_size=0.2, random_state=42)
trainModels(X_train, y_train)
predict(X_test)