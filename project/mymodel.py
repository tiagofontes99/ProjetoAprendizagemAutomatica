from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from project.models.Lasso import lassoPred
from project.models.Polynomial import polyPred
from project.models.Ridge import ridgePred

def bestmodel(X_test, y_test):
    gs_lasso = lassoPred(X_test)
    gs_ridge = ridgePred(X_test)
    gs_poly = polyPred(X_test)
    print("\n[LASSO]")
    print("MAE :", mean_absolute_error(X_test, y_test))
    print("Mean Squared Error:", mean_squared_error(X_test, y_test))
    print("R²  :", r2_score(X_test, y_test))

    # Avaliaçao Ridge
    print("\n[RIDGE]")
    print("MAE :", mean_absolute_error(X_test, y_test))
    print("Mean Squared Error:", mean_squared_error(X_test, y_test))
    print("R²  :", r2_score(X_test, y_test))

    # Avaliaçao Poly
    print("\n[Poly]")
    print("MAE :", mean_absolute_error(X_test, y_test))
    print("Mean Squared Error:", mean_squared_error(X_test, y_test))
    print("R²  :", r2_score(X_test, y_test))

def predict(X_test):
    return lassoPred(X_test)