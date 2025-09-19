import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from project.pickle.pickleFun import loadFromPickle, saveToPickle


def polyTrain(X_train, y_train):
    # Polinomial regression

    pipe_poly = Pipeline([
        ("poly", PolynomialFeatures(degree=1)),
        ("scaler", StandardScaler()),
        ("model", Ridge(random_state=42))
    ])

    param_poly = {
        "model__alpha": np.logspace(-3, 3, 15)
    }

    gs_poly = GridSearchCV(pipe_poly, param_poly, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs_poly.fit(X_train, y_train)
    saveToPickle(gs_poly, "polyTrained")

def polyPred(X_test):
    model = loadFromPickle("polyTrained")
    y_pred = model.predict(X_test)
    return y_pred

