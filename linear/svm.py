import numpy as np
from .linear_classification import LinearClassification

class SVM(LinearClassification):

    @staticmethod
    def _hinge_loss_func(y_pred, y_true):
        return np.mean(np.maximum(0, 1 - y_true * y_pred))
    @staticmethod
    def _hinge_loss_func_derivative(y_pred, y_true, X):
        dL_dM = np.where(y_true * y_pred < 1, -1, 0)
        dM_dw = X * y_true[:, None] 
        return (dL_dM[:, None] * dM_dw).sum(axis=0) / X.shape[0]
    
    def _ridge_func(self, w):
        result = 1/(2 * self.C) * (w ** 2).sum()
        return result
    def _ridge_func_derivative(self, w):
        result =  1/self.C * w
        result[-1] = 0 # do not regularize bias
        return result
    #===========================================================================
    def __init__(self, kernel='linear', const=1):
        # future kernels: poly, rbf, tanh
        if kernel == 'linear':
            self.loss_func = self._hinge_loss_func
            self.loss_func_derivative = self._hinge_loss_func_derivative

            self.regularizator_func = self._ridge_func
            self.regularizator_func_derivative = self._ridge_func_derivative

            self.C = const
        elif kernel == 'poly':
            # TEMPLATE
            raise NotImplementedError("poly kernel not implemented yet")
        elif kernel == 'rbf':
            # TEMPLATE
            raise NotImplementedError("rbf kernel not implemented yet")
        elif kernel == 'tanh':
            # TEMPLATE
            raise NotImplementedError("tanh kernel not implemented yet")
    #===========================================================================