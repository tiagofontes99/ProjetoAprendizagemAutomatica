import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from project.pickle.pickleFun import saveToPickle, loadFromPickle


# procura paramentros entre 10 elevado a -3 ou seja 0.001 ate 10 eevado a 3 = 1000 e faz 100 iterações
 #entre esses valorea a procurar os melhores
# grid de hiperparâmetros
#r2  coeficiente de determinação.
#neg_mean_squared_error" negativo do MSE.
#neg_root_mean_squared_error" negativo do RMSE.
#neg_mean_absolute_error" negativo do MAE.

def lassoTrain(X_train, y_train):
    param_lasso = {
        "model__alpha": np.logspace(-3, 3, 1000)
    }

    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=4, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lasso", Lasso(random_state=42, max_iter=10000))
    ])

    param_grid = {
        "poly__degree": [1, 2, 3, 4, 5 , 6 ,7],
        "lasso__alpha": np.logspace(-2, 4, 20)
    }


    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    gs_lasso = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    gs_lasso.fit(X_train, y_train)
    print("Melhores parâmetros:", gs_lasso.best_params_)
    print("Melhor MSE (CV):", -gs_lasso.best_score_)
    saveToPickle(gs_lasso , "lassoTrained")

def lassoPred(X_test):
    model =loadFromPickle("lassoTrained")
    y_pred=model.predict(X_test)
    return y_pred