

import numpy as np
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from project.pickle.pickleFun import loadFromPickle, saveToPickle

poly= PolynomialFeatures(degree=4)
def polyTrain1(X_train, y_train):
    pipe_lasso = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(max_iter=10000, random_state=42)),

    ])

    param_lasso = {
        "model__alpha": np.logspace(-3, 3, 1000)
    }

    # gridsearch
    # no scorring pode se usar outros valores como r2 que e o erro quadradp, igual ao metodo usaro apesar de nao
    # ser melhor que o neg_root
    gs_lasso = GridSearchCV(pipe_lasso, param_lasso, scoring="neg_mean_squared_error", n_jobs=-1)

    gs_poly = GridSearchCV(pipe_poly,param_poly,scoring="r2",n_jobs=-1)
    gs_poly.fit(X_train, y_train)
    saveToPickle(gs_poly, "polyTrained")

def polyPred1(X_test):
    model = loadFromPickle("polyTrained")
    y_pred = model.predict(poly.fit_transform(X_test))
    return y_pred
