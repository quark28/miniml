import numpy as np 


class KNNClassifier:

    def linear_weights(self, i):
        return (self.K + 1 - i) / self.K
    def exponential_weights(self, i):
        return self.q_for_exp_w ** i
    def dummy_weights(self, i):
        return 0
    
    _euclid = lambda a, b: np.sqrt( ( (a - b) ** 2 ).sum(axis=1) )
    _manhattan = lambda a, b: np.abs( (a - b) ).sum(axis=1)

    @staticmethod
    def _minkowski_maker(p):
        def _minkowski(a, b):
            return ((np.abs( (a - b) ) ** p).sum(axis=1)) ** (1 / p)
        return _minkowski

    _cosine = lambda a, b: 1 - (
    (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
)
    _jacquard = lambda a, b: np.array([1 - len(np.intersect1d(a[0], row)) / len(np.union1d(a[0], row)) for row in b])

    def __init__(self, neighbours=5, metric='euclid', p=None, weights=None, q=None, kernel=None):
        # weights OR kernel
        self.K = neighbours

        if metric == 'euclid':
            self.metric = KNNClassifier._euclid
        elif metric == 'manhattan':
            self.metric = KNNClassifier._manhattan
        elif metric == 'minkowski':
            self.metric = KNNClassifier._minkowski_maker(p)
        elif metric == 'cosine':
            self.metric = KNNClassifier._cosine
        elif metric == 'jacquard':
            self.metric = KNNClassifier._jacquard

        if weights:
            if weights == 'linear':
                self.w = self.linear_weights
            elif weights == 'exp':
                self.q_for_exp_w = q
                self.w = self.exponential_weights
        else:
            self.w = self.dummy_weights

        if kernel:
            if kernel == 'rectangular':
                self.kernel = lambda x: 1/2 * (x <= 1)
            elif kernel == 'triangular':
                self.kernel = lambda x: (1 - np.abs(x)) * (x <= 1)
            elif kernel in ('quartic', 'biquadrate'):
                self.kernel = lambda x: 15/16 * ( (1 - x**2)**2 ) * (x <= 1)
            elif kernel == 'epanechnikov':
                self.kernel = lambda x: 3/4 * (1 - x**2) * (x <= 1)
            elif kernel == 'gaussian':
                self.kernel = lambda x: 1/np.sqrt(2 * np.pi) * np.exp( -(x ** 2)/2 )
        else:
            self.kernel = lambda x: 0
    
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
            
            classes = {}
            for id in range(K_actual):
                yt = self.y_data[idx_k][id]
                classes[yt] = classes.get(yt, 0) + self.w(id) + self.kernel(kneighbours[id]/h)
            
            _ans[i] = max(classes, key=classes.get)
        return _ans

