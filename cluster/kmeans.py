import numpy as np


class KMeans:

    _euclid = lambda a, b: np.sqrt( ( (a - b) ** 2 ).sum(axis=-1) )
    _manhattan = lambda a, b: np.abs((a - b)).sum(axis=-1)

    @staticmethod
    def _minkowski_maker(p):
        def _minkowski(a, b):
            return ((np.abs((a - b)) ** p).sum(axis=-1)) ** (1 / p)
        return _minkowski

    @staticmethod
    def _cosine(a, b):
        a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
        return 1 - (a_norm * b_norm).sum(axis=-1)

    def __init__(self, random_state=42):
        self.random_state = random_state

    def get_metric(self, metric):
        if metric == 'euclid':
            return KMeans._euclid
        elif metric == 'manhattan':
            return KMeans._manhattan
        elif metric == 'minkowski':
            return KMeans._minkowski_maker(self.p)
        elif metric == 'cosine':
            return KMeans._cosine

    def lloyd_algorithm(self, X, K, max_iter, tol):

        for _ in range(max_iter):

            distances = self.metric(X[:, None, :], self.coord_centers[None, :, :])
            labels = np.argmin(distances, axis=1)

            new_coord_centers = np.array([
                X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else self.coord_centers[i] 
                for i in range(K)
                ])
            
            if np.all(np.linalg.norm(new_coord_centers - self.coord_centers, axis=1) < tol):
                break
            self.coord_centers = new_coord_centers
        return labels

    def get_start_centroids(self, X, K, init_type):

        if init_type == 'random':
            idx = np.random.choice(X.shape[0], K, replace=False)

        elif init_type == 'kmeans++':
            idx = [np.random.randint(0, X.shape[0])]

            for _ in range(K-1):
                matrix_centroid_to_object_distance = self.metric(X[idx][:, None, :], X)
                min_distance_to_centroids_per_object = matrix_centroid_to_object_distance.min(axis=0)
                min_distance_to_centroids_per_object **= 2
                min_distance_to_centroids_per_object /= min_distance_to_centroids_per_object.sum()
                next_centroid_idx = np.random.choice(X.shape[0], p=min_distance_to_centroids_per_object)
                idx.append(next_centroid_idx)
        return X[idx]

    def fit(self, X, K, metric='euclid', p=None, max_iter=100, tol=1e-4, init_type='random'):
        self.p = p
        self.metric = self.get_metric(metric)
        X = np.array(X)

        np.random.seed(self.random_state)

        self.coord_centers = self.get_start_centroids(X, K, init_type)

        
        self.train_labels_ = self.lloyd_algorithm(X, K, max_iter, tol)
        return self

    def predict(self, X, K, metric='euclid', p=None, max_iter=100, tol=1e-4):
        self.p = p
        self.metric = self.get_metric(metric)
        X = np.array(X)

        distances = self.metric(X[:, None, :], self.coord_centers[None, :, :])
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X, K, metric='euclid', p=None, max_iter=100, tol=1e-4, init_type='random'):
        self.fit(X, K, metric, p, max_iter, tol, init_type)
        
        return self.train_labels_        