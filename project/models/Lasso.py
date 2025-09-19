import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from project.pickle.pickleFun import saveToPickle, loadFromPickle


def lassoTrain(X_train, y_train):


    pipe_lasso = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(max_iter=10000, random_state=42))
    ])

    # grid de hiperparâmetros
    param_lasso = {
        "model__alpha": np.logspace(-3, 2, 15)
    }

    # gridsearch
    gs_lasso = GridSearchCV(pipe_lasso, param_lasso, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs_lasso.fit(X_train, y_train)
    saveToPickle(gs_lasso , "lassoTrained")

def lassoPred(X_test):
    model =loadFromPickle("lassoTrained")
    y_pred=model.predict(X_test)
    return y_pred