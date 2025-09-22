import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from project.pickle.pickleFun import saveToPickle, loadFromPickle


def ridgeTrain(X_train, y_train):
    # Regressao usando o metodo Ridge

    # grid de hiperparâmetros
    param_ridge = {
        "model__alpha": np.logspace(-3, 3, 1000)
    }

    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=4, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])
    pipeline.fit(X_train, y_train)

    # gridsearch
    #gs_ridge = GridSearchCV(pipeline, param_ridge, scoring="neg_root_mean_squared_error", n_jobs=-1)
    #gs_ridge.fit(X_train, y_train)
    saveToPickle(pipeline, "ridgeTrained")

def ridgePred(X_test):
    model = loadFromPickle("ridgeTrained")
    y_pred = model.predict(X_test)
    return y_pred
