import numpy as np
from .decisiontreeregressor import DecisionTreeRegression


class GradientBoostingBinaryClassification:

    @staticmethod
    def _logloss_loss_func(y_pred, y_true):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1-eps)

        return -np.mean(
            y_true*np.log(y_pred)
            +(1-y_true)*np.log(1-y_pred)
        )
    @staticmethod
    def _logloss_loss_func_derivative(y_pred, y_true):
        return y_pred - y_true
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.01,
                 common_loss_fn='log',
                 #early_stopping_rounds=None,
                 tree_mode='solo',
                 max_depth=5,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 tol=0,
                 max_features=None,
                 impurity_func='mse',
                 random_state=42
                 ):
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate

        if common_loss_fn == 'log':
            self.global_loss = self._logloss_loss_func
            self.global_loss_der = self._logloss_loss_func_derivative
        #elif common_loss_fn == ...:
            #...
        
        self.common_loss_fn=common_loss_fn
        #self.early_stopping_rounds=early_stopping_rounds
        self.tree_mode=tree_mode
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.tol=tol
        self.max_features=max_features
        self.impurity_func=impurity_func
        self.rng=np.random.default_rng(random_state)
    
    def get_max_features(self, N):
        if self.max_features == 'sqrt':
            return int(np.sqrt(N))
        elif isinstance(self.max_features, int):
            return min(self.max_features, N)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * N))
        else:
            return N
        
    def fit(self, X, y):
        mx_features = self.get_max_features(X.shape[1])
        self.ensemble = []
        eps = 1e-12
        p = np.clip(np.mean(y), eps, 1-eps)
        self.init_prediction = np.log(p/(1-p))
        act_predict = np.full(len(y), self.init_prediction)

        for i in range(self.n_estimators):

            tree_seed = self.rng.integers(0, 1_000_000)
            estimator = DecisionTreeRegression()
            loss_grad = -self.global_loss_der(self.sigmoid(act_predict), y)
            estimator.fit(
                X, loss_grad, max_depth = self.max_depth, min_samples_split = self.min_samples_split,
                min_samples_leaf = self.min_samples_leaf, tol = self.tol, impurity_func = self.impurity_func,
                mode = self.tree_mode, max_features = mx_features, random_state=tree_seed
            )
            act_predict += self.learning_rate * estimator.predict(X)
            self.ensemble.append(estimator)
        return self

    def predict(self, X):
        scores = np.full(X.shape[0], self.init_prediction)

        for tree in self.ensemble:
            scores += self.learning_rate * tree.predict(X)

        prob = self.sigmoid(scores)

        return (prob >= 0.5).astype(int)
