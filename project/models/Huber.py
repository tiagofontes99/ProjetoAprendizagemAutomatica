from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression, HuberRegressor, SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler

from project.pickle.pickleFun import loadFromPickle, saveToPickle



def huberTrain(X_train, y_train):
    pipe = Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scale", StandardScaler()),
        ("pca", PCA(svd_solver="auto", whiten=False)),
        ("huber", HuberRegressor(max_iter=500000))
    ])

    param_grid = {
        "poly__degree": [1,2,3,4,5,6,7,8],
        "pca__n_components": [0.95, 0.99],
        "huber__alpha": [1e-3, 1e-2, 1e-1],
        "huber__epsilon": [1.35, 1.5, 1.8],
        "huber__tol": [1e-3, 1e-4],
    }

    gs_poly = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    gs_poly.fit(X_train, y_train)


    #Save model to pickle
    saveToPickle(gs_poly, "huberTrained")

def huberPred(X_test):
    model = loadFromPickle("huberTrained")
    y_pred = model.predict(X_test)
    return y_pred

