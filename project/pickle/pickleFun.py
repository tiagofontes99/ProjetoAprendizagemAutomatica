import joblib
import numpy as np

def saveToPickle(model , regressionName):
#################SAVE TO PICKE FILE ############
    X = np.array([[i] for i in range(-5, 6)]) # Inputs: -5 ... 5
    y = 1 + 2*X[:, 0] # Outputs: 1 + 2x
    joblib.dump(model, regressionName)


def loadFromPickle(regressionName):
    loaded_model = joblib.load(regressionName)
    return loaded_model



