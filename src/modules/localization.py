import numpy as np
from numpy.linalg import det

from utils.utils import sec, compute_gradient_and_bias
from .qr_code import Estimation
from .qr_code import QrCode


class Localization:
    """Class for estimating the position of the robot"""

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

        # Computing the vairance matrices
        avg_variance_cx = np.round(np.average([qr_code.var_cx for qr_code in qr_codes]), decimals=3) 
        avg_variance_height = np.round(np.average([qr_code.var_height for qr_code in qr_codes]), decimals=3)
        self.R = np.array([[avg_variance_height, 0], [0, avg_variance_cx]])
        self.R_inv = np.array([[np.round(1/avg_variance_height, decimals=3), 0], [0, np.round(1/avg_variance_cx, decimals=3)]])

    def g(self, x, sx, sy):
        """Measurement model 
        
        Args:
            x: camera measurements [px, py, psi]
            sx: x coordinate of the QR code.
            sy: y coordinate of the QR code.

        Returns:
            g_function: measurement model g
        """
        px, py, psi = x
        
        h = float((self.h0*self.f)/np.sqrt((sx-px)**2+(sy-py)**2))

        res = np.arctan2((sy-py), (sx-px))
        if res > np.pi:
            res = res - 2*np.pi
        elif res < -np.pi:
            res = res + 2*np.pi
        
        cx = float(self.f*np.tan(res-psi))
        
        return np.array([h,cx])
        
    def G(self, x, sx, sy):
        """Computes the Jacobian matrix of the function g.
        
        Args:
            x: camera measurements [px, py, psi]
            sx: x coordinate of the QR code.
            sy: y coordinate of the QR code.

        Returns:
            jacobian: the Jacobian matrix of the function g. 
        """
        px, py, psi = x


        res = np.arctan2((sy-py), (sx-px))
        if res > np.pi:
            res = res - 2*np.pi
        elif res < -np.pi:
            res = res + 2*np.pi

        jacobian = np.zeros((2,3))
        
        # df1 / dpx
        jacobian[0][0] = (-px+sx)*self.f*self.h0/(((-px+sx)**2+(-py+sy)**2)**(3/2))
        
        # df1 / dpy
        jacobian[0][1] = (-py+sy)*self.f*self.h0/(((-px+sx)**2+(-py+sy)**2)**(3/2))
        
        # df1 / dpsi
        jacobian[0][2] = 0
        
        # df2 / dpx
        jacobian[1][0] = (self.f * (-py + sy) * (1 + np.tan(psi - res)**2)) / ((sx - px)**2  * (1 + (sy-py)**2/(sx-px)**2))
        
        # df2 / dpy
        jacobian[1][1] = (-self.f * (1 + np.tan(psi - res)**2)) / ((sx - px)**2  * (1 + (sy-py)**2/(sx-px)**2)) 
        
        # df2 / dpsi
        jacobian[1][2] = (-self.f * (1 + np.tan(psi - res)**2))
    
        return jacobian
    
    """ Cost function for Weighted Least Squares

    Args:
        x: parameters/state/estimate
        y: measurement(s)
        g: nonlinear measurement models
        params: the rest of the parameters

    Returns:
        the cost function for Weighted Least Squares
    """
    def Jwls(self, x, y, sx, sy):
        e = (y - self.g(x, sx, sy)) @ self.R_inv
        return np.sum(e**2)
    
    def localize(self, x_0=[0, 0, 0], N_gamma_grid=1000, gamma_max=50, iteration_end=100, epsilon=1e-8):
        """Routine for Gauss Netwon with Line Search"""
        # for qr in self.qr_codes:
        #     y = [[measurement.height, measurement.cx] for measurement in qr.measurements]
        #     x_GN_LS = np.zeros((iteration_end,3)) #the path of the estimation
        #     x_GN_LS[0,:] = x_0
        #     gamma = np.arange(1,N_gamma_grid+1)*gamma_max/N_gamma_grid

        #     for j in range(iteration_end-1):
        #         x_now = x_GN_LS[j,:]
        #         gj = self.g(x_now, qr.sx, qr.sy)
        #         Gj = self.G(x_now, qr.sx, qr.sy)
        #         # if (det(Gj.T @ self.R_inv @ Gj) == 0):
        #         #     print(f"qr code {qr.id}, {j} th iteration found a singular matrix")
        #         #     print(Gj.T @ self.R_inv @ Gj)
        #         #     continue
        #         A = Gj.T @ self.R_inv @ Gj + epsilon * np.eye(Gj.shape[1])
        #         L = np.linalg.cholesky(A)
        #         Delta_x = np.linalg.solve(L,Gj.T@ self.R_inv @ np.sum(y - gj , axis=0) / len(y))
        #         # Delta_x = np.linalg.solve((Gj.T@self.R_inv@Gj)*len(y),Gj.T@self.R_inv@np.sum(y-gj,axis=0)/len(y))
        #         J_min = self.Jwls(x_now, y, qr.sx, qr.sy)
        #         gamma_min = 0

        #         for k in range(N_gamma_grid):
        #             J_prop = self.Jwls(x_now+gamma[k]*Delta_x, y, qr.sx, qr.sy)
        #             if J_prop < J_min:
        #                 J_min = J_prop
        #                 gamma_min = gamma[k]

        #         x_GN_LS[j+1,:] = x_now + gamma_min * Delta_x

        #     qr.update_estimation([Estimation(x[0], x[1], x[2]) for x in x_GN_LS])
        for qr in self.qr_codes:
            y = [[measurement.height, measurement.cx] for measurement in qr.measurements]
            x_LM = np.zeros((iteration_end,3)) #the path of the estimation
            x_LM[0,:] = x_0
            I = np.eye(3)
            lambda_LM = 0.01 # set this 0 and infinity (e.g. 1e10) to see the results.
            nu = 10
            for j in range(iteration_end-1):
                x_now = x_LM[j,:]
                gj = self.g(x_now, qr.sx, qr.sy)
                Gj = self.G(x_now, qr.sx, qr.sy)
                dx = np.linalg.solve((len(y)*Gj.T@self.R_inv@Gj + lambda_LM*I),Gj.T@self.R_inv@np.sum(y-gj,axis=0))    
                if self.Jwls( x_now + dx,y,qr.sx,qr.sy) < self.Jwls( x_now,y,qr.sx,qr.sy):
                    x_LM[j+1,:] = x_now + dx
                    lambda_LM = lambda_LM / nu
                else:
                    x_LM[j+1,:] = x_now
                    lambda_LM = lambda_LM * nu
            qr.update_estimation([Estimation(x[0], x[1], x[2]) for x in x_LM])