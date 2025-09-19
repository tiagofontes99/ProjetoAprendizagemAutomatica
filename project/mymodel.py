from project.models.Lasso import lassoPred
from project.models.Polynomial import polyPred
from project.models.Ridge import ridgePred


def predict(X_test):
    lassopred = lassoPred(X_test)
    ridgepred = ridgePred(X_test)
    polypred = polyPred(X_test)
    return