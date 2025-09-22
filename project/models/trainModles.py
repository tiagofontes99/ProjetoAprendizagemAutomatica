from project.models.Lasso import lassoTrain
from project.models.Polynomial import polyTrain
from project.models.Ridge import ridgeTrain


def trainModels(X_train,y_train):
    lassoTrain(X_train, y_train)
    ridgeTrain(X_train, y_train)
    #polyTrain(X_train,y_train)