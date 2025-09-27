import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from project.pickle.pickleFun import saveToPickle, loadFromPickle




def lassoTrain(X_train, y_train):

    pipeline = Pipeline([
        ("poly", PolynomialFeatures( include_bias=False)),
        ("scaler", StandardScaler()),
        ("lasso", Lasso(random_state=42, max_iter=100000))
    ])

    param_grid = {
        "poly__degree": [1, 2, 3, 4, 5 , 6 ,7],
        "lasso__alpha": np.logspace(-2, 4, 20)
    }


    cv_lasso= KFold(n_splits=5, shuffle=True, random_state=42)

    gs_lasso = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv_lasso,
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