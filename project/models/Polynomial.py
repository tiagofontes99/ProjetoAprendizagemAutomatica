
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from project.pickle.pickleFun import loadFromPickle, saveToPickle

poly= PolynomialFeatures(degree=4)
def polyTrain(X_train, y_train):
    poly = PolynomialFeatures(degree=4)
    polynomialx = poly.fit_transform(X_train)

    poly.fit(polynomialx, y_train)
    linearRegression = LinearRegression()
    linearRegression.fit(polynomialx, y_train)
    saveToPickle(linearRegression, "polyTrained")

def polyPred(X_test):
    model = loadFromPickle("polyTrained")
    y_pred = model.predict(poly.fit_transform(X_test))
    return y_pred

