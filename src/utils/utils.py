import numpy as np
from sklearn import linear_model

def sec(phi):
    """Computes the value of the sec() function for a given angle.
    
    Args:
        phi: angle

    Returns:
        the sec value of the input angle phi 
    """
    
    func = 1/np.cos(phi)
    
    return func

def compute_gradient_and_bias(path, distance_bias=0):
    """Computes the gradient and bias of the linear regression model for the calibration data.

    Args:
        path: path to the calibration data file.
        distance_bias: bias to add to the distance values.
    
    Returns:
        gradient: the gradient of the linear regression model.
        bias: the bias of the linear regression model.
    """
    calibration_data = np.loadtxt(path, delimiter=',')
    distances_cm = calibration_data[:,0] + distance_bias #added Distance of camera pinhole and the IR detector and Distance of the wall and the wooden list
    heights_px = calibration_data[:,1]

    regr = linear_model.LinearRegression()
    X = (1/heights_px).reshape(-1, 1)
    regr.fit(X, distances_cm)
    distances_cm_pred = regr.predict(X)

    gradient = regr.coef_
    bias = regr.intercept_

    return gradient, bias