import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from project.pickle.pickleFun import saveToPickle, loadFromPickle


def ridgeTrain(X_train, y_train):
    # Regressao usando o metodo Ridge
    pipe_ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(random_state=42))
    ])

    # grid de hiperparâmetros
    param_ridge = {
        "model__alpha": np.logspace(-3, 3, 15)
    }

    # gridsearch
    gs_ridge = GridSearchCV(pipe_ridge, param_ridge, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs_ridge.fit(X_train, y_train)
    saveToPickle(gs_ridge, "ridgeTrained")

def ridgePred(X_test):
    model = loadFromPickle("ridgeTrained")
    y_pred = model.predict(X_test)
    return y_pred
