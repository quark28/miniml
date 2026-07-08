import numpy as np


class PCA:

    def __init__(self):
        self.mean_ = None
        self.components_ = None
        self.eigenvalues_ = None

    def fit(self, X, copy=True):
        if copy:
            X_ = X.copy()
        else:
            X_ = X

        X_ = X_.astype(float)
        self.mean_ = np.mean(X_, axis=0, keepdims=True)
        X_ -= self.mean_
        cov_matrix = ( X_.T @ X_ ) / X_.shape[0]

        self.eigenvalues_, self.components_ = np.linalg.eigh(cov_matrix)

        idx = np.argsort(self.eigenvalues_)[::-1]
        self.eigenvalues_, self.components_ = self.eigenvalues_[idx], self.components_[:, idx]

        return self
    
    def transform(self, X, k=None, copy=True):
        
        if copy:
            X_ = X.copy()
        else:
            X_ = X


        if k == None:
            k = self.components_.shape[0]

        X_ = X_.astype(float)

        X_ -= self.mean_

        U = self.components_[:, :k]
        return X_ @ U

    def fit_transform(self, X, k=None, copy=True):

        self.fit(X, copy)

        return self.transform(X, k, copy)