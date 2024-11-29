import numpy as np

from ..utils.utils import sec, compute_gradient_and_bias
from .qr_code import QrCode


class Localization:
    """Class for destimating the position of the robot"""

    def __init__(self, h0:int, path:str, qr_codes: list[QrCode], distance_bias:int=0):
        """Constructor for the localization class.
        
        Args:
            h0: true height of the QR code.
            path: path to the camera calibration data.
            qr_codes: list of QR codes and their information.
            distance_bias: bias to add to the distance values for the calibration data.
        """
        self.h0 = h0
        self.path = path
        self.qr_codes = qr_codes
        self.camera_gradient, self.camera_bias = compute_gradient_and_bias(self.path, distance_bias)
        self.f = self.camera_gradient/h0

    def g(h0, f, sx, px, sy, py, psi):
        """
        What the function does
        :param x: x = [px, py, psi]
        :return: measurement model g
        """
        
        h = (h0*f)/np.sqrt((sx-px)**2+(sy-py)**2)
        
        cx = f*np.tan(np.arctan((sy-py)/(sx-px))-psi)
        
        g_function = np.array([[h],[cx]])
        
        return g_function

    def G(h0, sx, px, sy, py, psi):
        """Computes the Jacobian matrix of the function g.
        
        Args:
            h0: 
            sx: 
            px: 
            sy: 
            py: 
            psi: 

        Returns:
            jacobian: the Jacobian matrix of the function g. 
        """
        jacobian = np.zeros((2,3))
        
        # df1 / dpx
        jacobian[0][0] = (4 * h0 * (sx - px)) / (((sx - px) ** 2 + (sy - py) ** 2) ** (3 / 2))
        
        # df1 / dpy
        jacobian[0][1] = (4 * h0 * (sy - py)) / (((sx - px) ** 2 + (sy - py) ** 2) ** (3 / 2))
        
        # df1 / dpsi
        jacobian[0][2] = 0
        
        # df2 / dpx
        jacobian[1][0] = (np.tan(psi)) / ((sx - px) + (sy - py) * np.tan(psi)) + (sy - py - (sx - px) * np.tan(psi)) / ((sx - px + (sy - py) * np.tan(psi))**2)   
        
        # df2 / dpy
        jacobian[1][1] = - 1 / (sx - px + (sy - py) * np.tan(psi)) - ((sy - py - (sx - px) * np.tan(psi)) * np.tan(psi)) / ((sx - px + (sy - py) * np.tan(psi))**2)
        
        # df2 / dpsi
        jacobian[1][2] = (- (sx - px) * (sec(psi)) ** 2) / (sx - px + (sy - py) * np.tan(psi)) - ((sy - py - (sx - px) * np.tan(psi)) * (sy - py) * (sec(psi)) ** 2) / (sx - px + (sy - py) * np.tan(psi)) ** 2
        
        return jacobian

