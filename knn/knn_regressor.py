import numpy as np 


class KNNRegression:

    @staticmethod
    def kernel_rectangular(x):
        return 1/2 * (x <= 1)
    @staticmethod
    def kernel_triangular(x):
        return (1 - np.abs(x)) * (x <= 1)
    @staticmethod
    def kernel_quartic(x):
        return 15/16 * ( (1 - x**2)**2 ) * (x <= 1)
    @staticmethod
    def kernel_epanechnikov(x):
        return 3/4 * (1 - x**2) * (x <= 1)
    @staticmethod
    def kernel_gaussian(x):
        return 1/np.sqrt(2 * np.pi) * np.exp( -(x ** 2)/2 )
    
    _euclid = lambda a, b: np.sqrt( ( (a - b) ** 2 ).sum(axis=1) )
    _manhattan = lambda a, b: np.abs( (a - b) ).sum(axis=1)

    @staticmethod
    def _minkowski_maker(p):
        def _minkowski(a, b):
            return ((np.abs( (a - b) ) ** p).sum(axis=1)) ** (1 / p)
        return _minkowski

    _cosine = lambda a, b: 1 - ( (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) )

    _jacquard = lambda a, b: np.array([1 - len(np.intersect1d(a[0], row)) / len(np.union1d(a[0], row)) for row in b])

    def __init__(self, neighbours=5, metric='euclid', p=None, weight_type='uniform', q=None, h=1, regression_type='classic'):
        self.K = neighbours
        self.metric_name = metric
        self.weight_type = weight_type  # 'uniform', 'rank_linear', 'rank_exp', 'kernel_rectangular', 
                                        # 'kernel_triangular', 'kernel_quartic', 'kernel_epanechnikov',
                                        # 'kernel_gaussian'
        self.q = q
        self.h = h
        self.rg = regression_type # 'classic' or 'nadaraya_watson'

        if metric == 'euclid':
            self.metric = KNNRegression._euclid
        elif metric == 'manhattan':
            self.metric = KNNRegression._manhattan
        elif metric == 'minkowski':
            self.metric = KNNRegression._minkowski_maker(p)
        elif metric == 'cosine':
            self.metric = KNNRegression._cosine
        elif metric == 'jacquard':
            self.metric = KNNRegression._jacquard
    
    def fit(self, X, y):
        self.X_data = X
        self.y_data = y

        return self
    
    def predict(self, X, h=1):

        _ans = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            dist = self.metric(X[i, None], self.X_data)
            K_actual = min(self.K, len(self.X_data))
            idx_k = np.argsort(dist)[:K_actual]

            kneighbours = dist[idx_k]
            
            numerator = np.zeros(K_actual)
            denominator = np.zeros(K_actual)
            for id in range(K_actual):
                yt = self.y_data[idx_k][id]
                dist_val = kneighbours[id]

                if self.weight_type == 'uniform':
                    vote = 1
                elif self.weight_type == 'rank_linear':
                    vote = (self.K + 1 - (id + 1)) / self.K # id + 1, rank = 1, ...
                elif self.weight_type == 'rank_exp':
                    vote = self.q ** (id + 1) # id + 1, rank = 1, ...
                else:

                    if self.weight_type == 'kernel_rectangular':
                        vote = self.kernel_rectangular(dist_val / self.h)
                    elif self.weight_type == 'kernel_triangular':
                        vote = self.kernel_triangular(dist_val / self.h)
                    elif self.weight_type == 'kernel_quartic':
                        vote = self.kernel_quartic(dist_val / self.h)
                    elif self.weight_type == 'kernel_epanechnikov':
                        vote = self.kernel_epanechnikov(dist_val / self.h)
                    elif self.weight_type == 'kernel_gaussian':
                        vote = self.kernel_gaussian(dist_val / self.h)
                    else:
                        vote = 1
                if self.rg == 'classic':
                    numerator[id] = yt
                    denominator[id] = 1
                else: # 'nadaraya_watson'
                    numerator[id] = vote * yt
                    denominator[id] = vote
            
            _ans[i] = numerator.sum()/denominator.sum()
        return _ans

