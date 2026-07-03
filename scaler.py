import numpy as np


class Scaler:
    
    @staticmethod
    def _min_max_scaling(x):
        mi = np.min(x)
        ma = np.max(x)
        return (x - mi) / (ma - mi if ma - mi != 0 else 1e-8)
    
    @staticmethod
    def _z_score_normalization(x):
        return (x - np.mean(x)) / (np.std(x) if np.std(x) != 0 else 1e-8)
    
    @staticmethod
    def _normalization(x):
        return x / (np.linalg.norm(x) if np.linalg.norm(x) != 0 else 1e-8)
    
    @staticmethod
    def _robust_scaling(x):
        return (x - np.median(x)) / ( (np.percentile(x, 75) - np.percentile(x, 25))
                                     if (np.percentile(x, 75) - np.percentile(x, 25)) != 0
                                     else 1e-8)

    def __init__(self, type='min_max'):

        if type == 'min_max':
            self.skrt = self._min_max_scaling
        elif type == 'z_norm':
            self.skrt = self._z_score_normalization
        elif type == 'norm':
            self.skrt = self._normalization
        elif type == 'robust':
            self.skrt = self._robust_scaling

    def fit_predict(self, X):
        _scaled_matrix = np.empty(X.shape)
        for feature_vector_num in range(X.shape[1]):
            _scaled_matrix[:, feature_vector_num] = self.skrt(X[:, feature_vector_num])
        return _scaled_matrix 