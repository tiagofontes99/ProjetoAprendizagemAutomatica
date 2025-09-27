import joblib


#def bestmodel(X_test, y_test):
    #gs_lasso = lassoPred(X_test)
    #gs_ridge = ridgePred(X_test)
    #poly = polyPred(X_test)
    #print("\n[LASSO]")
    #print("MAE :", mean_absolute_error(gs_lasso, y_test))
    #print("Mean Squared Error:", mean_squared_error(gs_lasso, y_test))
    #print("R²  :", r2_score(gs_lasso, y_test))

    # Avaliaçao Ridge
    #print("\n[RIDGE]")
    #print("MAE :", mean_absolute_error(gs_ridge, y_test))
    #print("Mean Squared Error:", mean_squared_error(gs_ridge, y_test))
    #print("R²  :", r2_score(gs_ridge, y_test))

     #Avaliaçao Poly
    #print("\n[Poly]")
    #print("MAE :", mean_absolute_error(poly, y_test))
    #print("Mean Squared Error:", mean_squared_error(poly, y_test))
    #print("R²  :", r2_score(poly, y_test))
    #return


def predict(X_test):
    loaded_model = joblib.load("ridgeTrained.pkl")
    Y_pred = loaded_model.predict(X_test)
    print("loaded model shape = ",Y_pred.shape)
    return Y_pred
