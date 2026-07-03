import numpy as np


class Scaler:
    
    def _min_max_scaling(self, X, use_fit_stats, axis):
        if use_fit_stats:
            minimum, maximum, denominator = self.params['minimum'], self.params['maximum'], self.params['denominator']
        else:
            minimum = np.min(X, axis, keepdims=True)
            maximum = np.max(X, axis, keepdims=True)
            denominator = (maximum - minimum)
            denominator = np.where(denominator == 0, 1, denominator)
            self.params = {'minimum': minimum, 'maximum': maximum, 'denominator': denominator}
        
        return (X - minimum) / denominator
    
    def _z_norm_normalization(self, X, use_fit_stats, axis):
        if use_fit_stats:
            mean, std = self.params['mean'], self.params['std']
        else:
            mean = np.mean(X, axis, keepdims=True)
            std = np.std(X, axis, keepdims=True)
            std = np.where(std == 0, 1, std)
            self.params = {'mean': mean, 'std': std}
        
        return (X - mean) / std
    
    def _normalization(self, X, use_fit_stats, axis):
        if use_fit_stats:
            norm = self.params['norm']
        else:
            norm = np.linalg.norm(X, axis, keepdims=True) 
            norm = np.where(norm == 0, 1, norm)
            self.params = {'norm': norm}
        
        return X / norm
    
    def _robust_scaling(self, X, use_fit_stats, axis):
        if use_fit_stats:
            median = self.params['median']
            iqr = self.params['iqr']
        else:
            median = np.median(X, axis, keepdims=True)
            percentile25 = np.percentile(X, 25, axis, keepdims=True)
            percentile75 = np.percentile(X, 75, axis, keepdims=True)
            iqr = percentile75 - percentile25
            iqr = np.where(iqr == 0, 1, iqr)

            self.params = {'median': median, 'percentile25': percentile25, 'percentile75': percentile75, 'iqr': iqr}

        return (X - median) / iqr

    def _get_method(self, func_name):
        functions = {
            'min_max': self._min_max_scaling,
            'z_norm': self._z_norm_normalization,
            'norm': self._normalization,
            'robust': self._robust_scaling
        }
        return functions[func_name]

    def __init__(self):
        pass

    def fit(self, X, type='min_max', axis=0):

        self.method = self._get_method(type)
        self.method(X, False, axis)

        return self

    def transform(self, X, axis=0, use_fit_stats=True):
        return self.method(X, use_fit_stats, axis)

    def fit_transform(self, X, type='min_max', axis=0, use_fit_stats=True):
        self.fit(X, type, axis)
        return self.transform(X, type, axis, use_fit_stats)