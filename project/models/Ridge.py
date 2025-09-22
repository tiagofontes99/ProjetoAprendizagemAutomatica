import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from project.pickle.pickleFun import saveToPickle, loadFromPickle


def ridgeTrain(X_train, y_train):
    # Regressao usando o metodo Ridge
    param_ridge = {
        "poly__degree": [1, 2, 3, 4, 5, 6 , 7],
        "ridge__alpha": np.logspace(-3, 3, 20)  # 0.001 a 1000
    }

    pipeline_ridge = Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(random_state=42, max_iter=100000))
    ])
    # grid de hiperparâmetros



    cv_ridge = KFold(n_splits=5, shuffle=True, random_state=42)

    gs_ridge = GridSearchCV(
        estimator=pipeline_ridge,
        param_grid=param_ridge,
        scoring="neg_mean_squared_error",
        cv=cv_ridge,
        n_jobs=-1,
        refit=True,  # refit com os melhores hiperparâmetros
        verbose=0
    )
    gs_ridge.fit(X_train, y_train)
    print("Melhores parâmetros:", gs_ridge.best_params_)
    print("Melhor MSE (CV):", -gs_ridge.best_score_)
    saveToPickle(gs_ridge, "ridgeTrained")

def ridgePred(X_test):
    model = loadFromPickle("ridgeTrained")
    y_pred = model.predict(X_test)
    return y_pred
