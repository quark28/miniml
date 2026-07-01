import numpy as np


class LinearRegression():
    '''
    _losses = ('mse', 'mae', 'huber', 'log-cosh')
    _regularizers = {'lasso': 'L1', 'ridge': 'L2', 'elasticnet': 'elasticnet', 'none': None}
    _metrics = ('mse', 'rmse', 'mae', 'r^2', 'adjusted_r^2', 'mape')
    _lr's = ('constant', 'decay')
    _learning types = ('analytic' - ('classic', 'svd'), ...
    '''
    #===========================================================================
    def _lasso_func(self, w):
        result = self.alpha * np.abs(w).sum()
        return result
    def _lasso_func_derivative(self, w):
        result = self.alpha * np.sign(w)
        result[-1] = 0
        return result

    def _ridge_func(self, w):
        result = self.alpha * (w ** 2).sum()
        return result
    def _ridge_func_derivative(self, w):
        result = self.alpha * 2 * w
        result[-1] = 0 # do not regularize bias
        return result

    def _elasticnet_func(self, w):
        # alphas = (alpha_l1, alpha_l2)
        result = self.alpha_l1 * np.abs(w).sum() + self.alpha_l2 * (w ** 2).sum()
        return result
    def _elasticnet_func_derivative(self, w):
        result = self.alpha_l1 * np.sign(w) + 2 *  self.alpha_l2 * w
        result[-1] = 0
        return result
    
    def _none_func(*args):
        return 0
    def _none_func_derivative(*args):
        return 0
    
    # Loss
    @staticmethod
    def _mse_loss_func(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    @staticmethod
    def _mse_loss_func_derivative(y_pred, y_true, X):
        N = X.shape[0]
        return 2 / N * X.T @ (y_pred - y_true)

    @staticmethod
    def _mae_loss_func(y_pred, y_true,):
        return np.mean(np.abs(y_true - y_pred))
    @staticmethod
    def _mae_loss_func_derivative(y_pred, y_true, X):
        N = X.shape[0]
        return (1 / N) * X.T @ np.sign(y_pred - y_true)

    def _huber_loss_func(self, y_pred, y_true):
        delta = self.delta
        quadratic = 1/2 * (y_pred - y_true) ** 2
        linear = delta * (np.abs(y_pred - y_true) - 1/2 * delta)
        return np.where(np.abs(y_pred - y_true) <= delta, quadratic, linear).mean()
    def _huber_loss_func_derivative(self, y_pred, y_true, X):
        delta = self.delta
        N = X.shape[0]
        grad_loss = np.where(np.abs(y_pred - y_true) <= delta, (y_pred - y_true), delta * np.sign(y_pred - y_true))
        return (1 / N) * X.T @ grad_loss

    @staticmethod
    def _logcosh_loss_func(y_pred, y_true):
        return np.mean(np.log(np.cosh(y_pred - y_true)))
    @staticmethod
    def _logcosh_loss_func_derivative(y_pred, y_true, X):
        N = X.shape[0]
        return -(1 / N) * X.T @ np.tanh(y_pred - y_true)
    #===========================================================================
    def _get_lr(self, initial_lr, lr_type, epoch, decay_rate=0.01):
        if lr_type == 'constant':
            return initial_lr
        elif lr_type == 'decay':
            return initial_lr / (1 + decay_rate * epoch)

    def __init__(self, loss, regularizator = None, delta = None, alpha = None):
        self.alpha = alpha
        
        if regularizator == 'elasticnet':
            # alpha = (alpha_l1, alpha_l2)
            if isinstance(alpha, (tuple, list)) and len(alpha) == 2:
                self.alpha_l1 = alpha[0]
                self.alpha_l2 = alpha[1]
            else:
                raise Exception("elasticnet requires alpha=(alpha_l1, alpha_l2)")

        self.loss = loss
        self.regularizator = regularizator
        self.delta = delta # huber
        if loss == 'mse':
            self.loss_func = self._mse_loss_func
            self.loss_func_derivative = self._mse_loss_func_derivative
        elif loss == 'mae':
            self.loss_func = self._mae_loss_func
            self.loss_func_derivative = self._mae_loss_func_derivative
        elif loss == 'huber':
            self.loss_func = self._huber_loss_func
            self.loss_func_derivative = self._huber_loss_func_derivative
        elif loss == 'log-cosh':
            self.loss_func = self._logcosh_loss_func
            self.loss_func_derivative = self._logcosh_loss_func_derivative

        if regularizator == 'lasso':
            self.regularizator_func = self._lasso_func
            self.regularizator_func_derivative = self._lasso_func_derivative
        elif regularizator == 'ridge':
            self.regularizator_func = self._ridge_func
            self.regularizator_func_derivative = self._ridge_func_derivative
        elif regularizator == 'elasticnet':
            self.regularizator_func = self._elasticnet_func
            self.regularizator_func_derivative = self._elasticnet_func_derivative
        else:
            self.regularizator_func = self._none_func
            self.regularizator_func_derivative = self._none_func_derivative

    def _analytic_solution(self, X, y, ad_type):

        if self.regularizator is None: # no care for not to regularize bias
            X = np.c_[X, np.ones(X.shape[0])] #add 1-column for bias
            if ad_type == 'classic':
                w = np.linalg.inv(X.T @ X) @ X.T @ y
            elif ad_type == 'svd':
                w = np.linalg.pinv(X) @ y

        elif self.regularizator == 'ridge':
            X = np.c_[X, np.ones(X.shape[0])]
            if ad_type == 'classic':
                penalty = self.alpha * np.eye(X.shape[1])
                penalty[-1, -1] = 0.0 # do not regularize bias
                w = np.linalg.inv(X.T @ X + penalty) @ X.T @ y
        
            elif ad_type == 'svd':
                X_features = X[:, :-1]

                mean_X = np.mean(X_features, axis=0)
                mean_y = np.mean(y)

                X_centered = X_features - mean_X
                y_centered = y - mean_y

                U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
                d = s / (s**2 + self.alpha)

                coef = Vt.T @ (d * (U.T @ y_centered))
                bias = mean_y - mean_X @ coef

                w = np.append(coef, bias)
        else:
            raise Exception('ERROR: Analytic solution only exists for none- and ridge- regularizators.')

        return w

    def _compute_gradient(self, X, y, w=None):
        if w is None:
            w = self.w
        y_pred = X @ w
        grad_loss = self.loss_func_derivative(y_pred, y, X)
        grad_reg = self.regularizator_func_derivative(w)
        return grad_loss + grad_reg

    def _gradient_descent(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate):
        X_with_bias = np.c_[X, np.ones(X.shape[0])] #add 1-column for bias
        self.w = np.random.randn(X_with_bias.shape[1]) if not hasattr(self, 'w') else self.w

        Q_prev = self.loss_func(X_with_bias @ self.w, y)
        for i in range(n_steps):
            lr = self._get_lr(lr_, lr_type, i, decay_rate)
            self.w = self.w - lr * self._compute_gradient(X_with_bias, y)
            y_prednew = X_with_bias @ self.w
            Q_current = self.loss_func(y_prednew, y)
            if quality_limit and abs(Q_prev - Q_current) < quality_limit:
                break
            Q_prev = Q_current
        return self.w

    def _stochastic_gradient_descent(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, batch_size):
        X_with_bias = np.c_[X, np.ones(X.shape[0])] #add 1-column for bias
        self.w = np.random.randn(X_with_bias.shape[1]) if not hasattr(self, 'w') else self.w

        Q_prev = self.loss_func(X_with_bias @ self.w, y)
        for i in range(n_steps):
            idx = np.random.choice(
                np.arange(0, X_with_bias.shape[0]), batch_size, replace=False
                )
            object_X = X_with_bias[idx]
            object_y = y[idx]
            lr = self._get_lr(lr_, lr_type, i, decay_rate)
            self.w = self.w - lr * self._compute_gradient(object_X, object_y)
            y_prednew = object_X @ self.w
            Q_current = self.loss_func(y_prednew, object_y)
            if quality_limit and abs(Q_prev - Q_current) < quality_limit:
                break
            Q_prev = Q_current
        return self.w

    def _stochastic_average_gradient(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, batch_size):
        X_with_bias = np.c_[X, np.ones(X.shape[0])] #add 1-column for bias
        self.w = np.random.randn(X_with_bias.shape[1]) if not hasattr(self, 'w') else self.w

        grads = np.empty(X_with_bias.shape)
        for idx in range(X_with_bias.shape[0]):
            grads[idx] = self._compute_gradient(X_with_bias[idx:idx+1], y[idx:idx+1])
        avg_grad = np.mean(grads, axis=0)

        y_pred = X_with_bias @ self.w
        Q_prev = self.loss_func(y_pred, y)
        for i in range(n_steps):
            idx = np.random.choice(
                np.arange(0, X_with_bias.shape[0]), batch_size, replace=False
                )
            object_X = X_with_bias[idx]
            object_y = y[idx]

            lr = self._get_lr(lr_, lr_type, i, decay_rate)

            for j in idx:
                X_j = X_with_bias[j:j+1]
                y_j = y[j:j+1]
                new_grad = self._compute_gradient(X_j, y_j)
                avg_grad += (new_grad - grads[j]) / X_with_bias.shape[0]
                grads[j] = new_grad

            self.w = self.w - lr * avg_grad
            y_prednew = object_X @ self.w
            Q_current = self.loss_func(y_prednew, object_y)
            if quality_limit and abs(Q_prev - Q_current) < quality_limit:
                break
            Q_prev = Q_current
        return self.w

    def _momentum(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, gamma):
        X_with_bias = np.c_[X, np.ones(X.shape[0])] #add 1-column for bias
        self.w = np.random.randn(X_with_bias.shape[1]) if not hasattr(self, 'w') else self.w

        velocity = np.zeros(X_with_bias.shape[1])

        Q_prev = self.loss_func(X_with_bias @ self.w, y)
        for i in range(n_steps):

            lr = self._get_lr(lr_, lr_type, i, decay_rate)
            velocity = gamma * velocity + lr * self._compute_gradient(X_with_bias, y)

            self.w = self.w - velocity

            y_prednew = X_with_bias @ self.w
            Q_current = self.loss_func(y_prednew, y)
            if quality_limit and abs(Q_prev - Q_current) < quality_limit:
                break
            Q_prev = Q_current
        return self.w

    def _nesterov_accelerated_gradient(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, gamma):
        X_with_bias = np.c_[X, np.ones(X.shape[0])] #add 1-column for bias
        self.w = np.random.randn(X_with_bias.shape[1]) if not hasattr(self, 'w') else self.w

        velocity = np.zeros(X_with_bias.shape[1])

        Q_prev = self.loss_func(X_with_bias @ self.w, y)
        for i in range(n_steps):
            lr = self._get_lr(lr_, lr_type, i, decay_rate)
            w = self.w - gamma * velocity
            velocity = gamma * velocity + lr * self._compute_gradient(X_with_bias, y, w)

            self.w = self.w - velocity
            y_prednew = X_with_bias @ self.w
            Q_current = self.loss_func(y_prednew, y)
            if quality_limit and abs(Q_prev - Q_current) < quality_limit:
                break
            Q_prev = Q_current
        return self.w

    def _root_mean_square_propagation(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, alpha, const):
        X_with_bias = np.c_[X, np.ones(X.shape[0])] #add 1-column for bias
        self.w = np.random.randn(X_with_bias.shape[1]) if not hasattr(self, 'w') else self.w

        Q_prev = self.loss_func(X_with_bias @ self.w, y)
        G = np.zeros(X_with_bias.shape[1])
        for i in range(n_steps):
            lr = self._get_lr(lr_, lr_type, i, decay_rate)
            grad = self._compute_gradient(X_with_bias, y)
            G = alpha * G + (1 - alpha) * (grad * grad)

            self.w = self.w - lr * (grad / (np.sqrt(G) + const))
            y_prednew = X_with_bias @ self.w
            Q_current = self.loss_func(y_prednew, y)
            if quality_limit and abs(Q_prev - Q_current) < quality_limit:
                break
            Q_prev = Q_current
        return self.w

    def _adaptive_learning_rate(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, alpha, const):
        X_with_bias = np.c_[X, np.ones(X.shape[0])] #add 1-column for bias
        self.w = np.random.randn(X_with_bias.shape[1]) if not hasattr(self, 'w') else self.w

        Q_prev = self.loss_func(X_with_bias @ self.w, y)
        G = np.zeros(X_with_bias.shape[1])
        DELTA = np.zeros(X_with_bias.shape[1])
        for i in range(n_steps):
            lr = self._get_lr(lr_, lr_type, i, decay_rate)
            grad = self._compute_gradient(X_with_bias, y)
            G = alpha * G + (1 - alpha) * (grad * grad)
            delta = grad * (
                (np.sqrt(DELTA) + const) / (np.sqrt(G) + const)
            )
            DELTA = alpha * DELTA + (1 - alpha) * (delta * delta)

            self.w = self.w - lr * delta
            y_prednew = X_with_bias @ self.w
            Q_current = self.loss_func(y_prednew, y)
            if quality_limit and abs(Q_prev - Q_current) < quality_limit:
                break
            Q_prev = Q_current
        return self.w

    def _adaptive_momentum(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, gamma, alpha, const):
        X_with_bias = np.c_[X, np.ones(X.shape[0])] #add 1-column for bias
        self.w = np.random.randn(X_with_bias.shape[1]) if not hasattr(self, 'w') else self.w

        velocity = np.zeros(X_with_bias.shape[1])
        G = np.zeros(X_with_bias.shape[1])

        Q_prev = self.loss_func(X_with_bias @ self.w, y)
        for i in range(n_steps):

            lr = self._get_lr(lr_, lr_type, i, decay_rate)
            grad = self._compute_gradient(X_with_bias, y)

            velocity = gamma * velocity + (1 - gamma) * grad
            G = alpha * G + (1 - alpha) * (grad * grad)

            velocity_normed = velocity / (1 - gamma ** (i + 1))
            G_normed = G / (1 - alpha ** (i + 1))
            
            self.w = self.w - lr * (
                velocity_normed / (np.sqrt(G_normed) + const)
            )

            y_prednew = X_with_bias @ self.w
            Q_current = self.loss_func(y_prednew, y)
            if quality_limit and abs(Q_prev - Q_current) < quality_limit:
                break
            Q_prev = Q_current
        return self.w

    def _nesterov_accelerated_adaptive_momentum(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, gamma, alpha, const):
        X_with_bias = np.c_[X, np.ones(X.shape[0])] #add 1-column for bias
        self.w = np.random.randn(X_with_bias.shape[1]) if not hasattr(self, 'w') else self.w

        velocity = np.zeros(X_with_bias.shape[1])
        G = np.zeros(X_with_bias.shape[1])

        Q_prev = self.loss_func(X_with_bias @ self.w, y)
        for i in range(n_steps):

            lr = self._get_lr(lr_, lr_type, i, decay_rate)
            grad = self._compute_gradient(X_with_bias, y)

            velocity = gamma * velocity + (1 - gamma) * grad
            G = alpha * G + (1 - alpha) * (grad * grad)

            velocity_normed = velocity / (1 - gamma ** (i + 1))
            G_normed = G / (1 - alpha ** (i + 1))
            
            self.w = self.w - lr * (
                gamma * velocity_normed + ((1 - gamma)/(1 - gamma ** (i + 1))) * grad
            ) * (1/(np.sqrt(G_normed) + const))

            y_prednew = X_with_bias @ self.w
            Q_current = self.loss_func(y_prednew, y)
            if quality_limit and abs(Q_prev - Q_current) < quality_limit:
                break
            Q_prev = Q_current
        return self.w

    def _adaptive_gradient(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, const):
        X_with_bias = np.c_[X, np.ones(X.shape[0])] #add 1-column for bias
        self.w = np.random.randn(X_with_bias.shape[1]) if not hasattr(self, 'w') else self.w

        r = np.zeros(X_with_bias.shape[1])
        Q_prev = self.loss_func(X_with_bias @ self.w, y)
        for i in range(n_steps):

            lr = self._get_lr(lr_, lr_type, i, decay_rate)
            grad = self._compute_gradient(X_with_bias, y)

            r += grad ** 2
            self.w = self.w - lr * (1/(np.sqrt(r) + const)) * grad

            y_prednew = X_with_bias @ self.w
            Q_current = self.loss_func(y_prednew, y)
            if quality_limit and abs(Q_prev - Q_current) < quality_limit:
                break
            Q_prev = Q_current
        return self.w

    def fit(self, X, y, 
            learning_type,
            ad_type = None, 
            n_steps = 2000, 
            lr = 0.01, 
            lr_type = 'constant', 
            quality_limit = None,
            decay_rate = None,
            batch_size = 1,
            gamma = 0.9,
            alpha = 0.999,
            const=1e-8
            ):

        if learning_type in ('analytic_solution', 'as', 'AS'):
            ad_type = 'classic' if ad_type is None else ad_type
            self.w = self._analytic_solution(X, y, ad_type)
            self.coefficients = self.w[:-1]
            self.bias = self.w[-1]
        
        elif learning_type in ('gradient_descent', 'gd', 'GD'):
            self.w = self._gradient_descent(
                X, y, n_steps, quality_limit, lr, lr_type, decay_rate
                )
            self.coefficients = self.w[:-1]
            self.bias = self.w[-1]
        
        elif learning_type in ('stochastic_gradient_descent', 'sgd', 'SGD'):
            self.w = self._stochastic_gradient_descent(X, y, n_steps, quality_limit, lr, lr_type, decay_rate, batch_size)
            self.coefficients = self.w[:-1]
            self.bias = self.w[-1]
    
        elif learning_type in ('stochastic_average_gradient', 'sag', 'SAG'):
            self.w = self._stochastic_average_gradient(X, y, n_steps, quality_limit, lr, lr_type, decay_rate, batch_size)
            self.coefficients = self.w[:-1]
            self.bias = self.w[-1]
        
        elif learning_type in ('momentum', 'mm', 'MM'):
            self.w = self._momentum(X, y, n_steps, quality_limit, lr, lr_type, decay_rate, gamma)
            self.coefficients = self.w[:-1]
            self.bias = self.w[-1]

        elif learning_type in ('nesterov_accelerated_gradient', 'nag', 'NAG'):
            self.w = self._nesterov_accelerated_gradient(X, y, n_steps, quality_limit, lr, lr_type, decay_rate, gamma)
            self.coefficients = self.w[:-1]
            self.bias = self.w[-1]
        
        elif learning_type in ('root_mean_square_propagation', 'rms', 'RMSProp'):
            self.w = self._root_mean_square_propagation(X, y, n_steps, quality_limit, lr, lr_type, decay_rate, alpha, const)
            self.coefficients = self.w[:-1]
            self.bias = self.w[-1]

        elif learning_type in ('adaptive_learning_rate', 'adadelta', 'AdaDelta'):
            self.w = self._adaptive_learning_rate(X, y, n_steps, quality_limit, lr, lr_type, decay_rate, alpha, const)
            self.coefficients = self.w[:-1]
            self.bias = self.w[-1]

        elif learning_type in ('adaptive_momentum', 'adam', 'Adam'):
            self.w = self._adaptive_momentum(X, y, n_steps, quality_limit, lr, lr_type, decay_rate, gamma, alpha, const)
            self.coefficients = self.w[:-1]
            self.bias = self.w[-1]
        
        elif learning_type in ('nesterov_accelerated_adaptive_momentum', 'nadam', 'Nadam'):
            self.w = self._nesterov_accelerated_adaptive_momentum(X, y, n_steps, quality_limit, lr, lr_type, decay_rate, gamma, alpha, const)
            self.coefficients = self.w[:-1]
            self.bias = self.w[-1]
        
        elif learning_type in ('adaptive_gradient_algorithm', 'adagrad', 'AdaGrad'):
            self.w = self._adaptive_gradient(X, y, n_steps, quality_limit, lr, lr_type, decay_rate, const)
            self.coefficients = self.w[:-1]
            self.bias = self.w[-1]

        else:
            raise Exception('ERROR: Wrong choice of model learning type.')
        return self
    
    def predict(self, X):
        X_with_bias = np.c_[X, np.ones(X.shape[0])]
        return X_with_bias @ self.w
