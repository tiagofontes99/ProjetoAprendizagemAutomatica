
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from project.pickle.pickleFun import loadFromPickle, saveToPickle

poly= PolynomialFeatures(degree=4)


def polyTrain(X_train, y_train):
    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=4, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])
    pipeline.fit(X_train, y_train)




    #Save model to pickle
    saveToPickle(pipeline, "polyTrained")

def polyPred(X_test):
    model = loadFromPickle("polyTrained")
    y_pred = model.predict(X_test)
    return y_pred

