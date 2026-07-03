import numpy as np
from collections import deque


class DBSCAN():

    _euclid = lambda a, b: np.sqrt( ( (a - b) ** 2 ).sum() )
    _manhattan = lambda a, b: np.abs( (a - b) ).sum()

    @staticmethod
    def _minkowski_maker(p):
        def _minkowski(a, b):
            return ((np.abs( (a - b) ) ** p).sum()) ** (1 / p)
        return _minkowski

    _cosine = lambda a, b: 1 - ( a @ b / (np.linalg.norm(a, ord=2) * np.linalg.norm(b, ord=2)) )
    _jacquard = lambda a, b: 1 - (np.intersect1d(a, b).shape[0] / np.union1d(a, b).shape[0])

    def __init__(self, eps, min_samples=1, distance_func='euclid', p=None):
        if distance_func == 'euclid':
            self.dfunc = DBSCAN._euclid
        elif distance_func == 'manhattan':
            self.dfunc = DBSCAN._manhattan
        elif distance_func == 'minkowski':
            self.dfunc = DBSCAN._minkowski_maker(p)
        elif distance_func == 'cosine':
            self.dfunc = DBSCAN._cosine
        elif distance_func == 'jacquard':
            self.dfunc = DBSCAN._jacquard
        
        self.eps = eps
        self.ms = min_samples
        
    def _get_neighbours(self, point, candidates, data):
        return deque(
            i for i in candidates 
            if self.dfunc(data[point], data[i]) <= self.eps 
            and point != i
            )
    
    def _bfs(self, _neighbours, current_cluster, _visited_points, X):
        
        while _neighbours:
            current_point = _neighbours.popleft()

            if current_point not in _visited_points:
                _visited_points.add(current_point)
                nbs = self._get_neighbours(current_point, range(X.shape[0]), X)
               
                if len(nbs) >= self.ms:
                    for n in nbs:
                        if n not in _visited_points:
                            _neighbours.append(n)
                
                current_cluster.append(current_point)
                
                if current_point in self._noise_points:
                    self._noise_points.discard(current_point)
        return current_cluster, _visited_points

    def _get_labels(self, X):
        self.labels_ = np.full(X.shape[0], -1, dtype=int)
        for i, cluster in enumerate(self._clusters):
            for idx in cluster:
                self.labels_[idx] = i
        return self.labels_

    def fit(self, X):

        _visited_points = set()
        self._noise_points = set()
        self._clusters = []

        for i in range(X.shape[0]):

            if i in _visited_points:
                continue

            _neighbours = self._get_neighbours(i, np.arange(X.shape[0]), X)
            
            if len(_neighbours) < self.ms:
                self._noise_points.add(i)
            else:
                _visited_points.add(i)
                cluster, _visited_points = self._bfs(_neighbours, deque([i]), _visited_points, X)
                self._clusters.append(cluster)
            
        self._clusters.sort(key=lambda x: x[0])
        self.n_clusters_ = len(self._clusters)
        self.n_noise_ = len(self._noise_points)
        return self
    
    def fit_predict(self, X):

        self.fit(X)

        return self._get_labels(X)