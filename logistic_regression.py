import numpy as np


class LogisticRegression:

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
    #===========================================================================
    # no classic loss version (production) because of derivative difficulty
    @staticmethod
    def _logloss_loss_func(probs, y_true):
        eps = 1e-9
        probs = np.clip(probs, eps, 1 - eps)
        return -(y_true * np.log(probs) + (1 - y_true) * np.log((1 - probs))).sum()
    @staticmethod
    def _logloss_loss_func_derivative(probs, y_true, X):
        return ((probs - y_true)[:, None] * X).sum(axis=0)

    #===========================================================================
    def __init__(self, classic_loss = False, regularizator = None, delta = None, alpha = None):
        self.alpha = alpha
        
        if regularizator == 'elasticnet':
            # alpha = (alpha_l1, alpha_l2)
            if isinstance(alpha, (tuple, list)) and len(alpha) == 2:
                self.alpha_l1 = alpha[0]
                self.alpha_l2 = alpha[1]
            else:
                raise Exception("elasticnet requires alpha=(alpha_l1, alpha_l2)")

        self.regularizator = regularizator
        self.delta = delta # huber

        self.loss_func = self._logloss_loss_func
        self.loss_func_derivative = self._logloss_loss_func_derivative

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
    #===========================================================================
    @staticmethod
    def _sigmoid(X, w):
        return 1 / (1 + np.exp(-X @ w))
    
    def _get_lr(self, initial_lr, lr_type, epoch, decay_rate=0.01):
        if lr_type == 'constant':
            return initial_lr
        elif lr_type == 'decay':
            return initial_lr / (1 + decay_rate * epoch)
        
    def _compute_gradient(self, X, y_true, w=None):
        if w is None:
            w = self.w
        _y_pred = self._sigmoid(X, w)
        grad_loss = self.loss_func_derivative(_y_pred, y_true, X)
        grad_reg = self.regularizator_func_derivative(w)
        return grad_loss + grad_reg

    def _optimizer_start(self, X):
        X = np.c_[X, np.ones(X.shape[0])] # add 1-column for bias
        self.w = np.random.randn(X.shape[1]) if not hasattr(self, 'w') else self.w
        return X

    def _omptimizer_early_stop(self, X_with_bias, y, quality_limit, Q_prev):
        _y_pred = self._sigmoid(X_with_bias, self.w)
        Q_current = self.loss_func(_y_pred, y)
        if quality_limit and abs(Q_prev - Q_current) < quality_limit:
            return Q_current, True
        return Q_current, False
    #===========================================================================
    def _gradient_descent(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate):

        X_with_bias = self._optimizer_start(X)

        Q_prev = self.loss_func(self._sigmoid(X_with_bias, self.w), y)
        for i in range(n_steps):
            lr = self._get_lr(lr_, lr_type, i, decay_rate)
            self.w = self.w - lr * self._compute_gradient(X_with_bias, y)

            Q_prev, flag = self._omptimizer_early_stop(X_with_bias, y, quality_limit, Q_prev)
            if flag:
                break
        return self.w

    def _stochastic_gradient_descent(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, batch_size):
        
        X_with_bias = self._optimizer_start(X)

        Q_prev = self.loss_func(self._sigmoid(X_with_bias, self.w), y)
        for i in range(n_steps):
            idx = np.random.choice(
                np.arange(0, X_with_bias.shape[0]), batch_size, replace=False
                )
            object_X = X_with_bias[idx]
            object_y = y[idx]
            lr = self._get_lr(lr_, lr_type, i, decay_rate)
            self.w = self.w - lr * self._compute_gradient(object_X, object_y)

            Q_prev, flag = self._omptimizer_early_stop(X_with_bias, y, quality_limit, Q_prev)
            if flag:
                break
        return self.w

    def _stochastic_average_gradient(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, batch_size):
        
        X_with_bias = self._optimizer_start(X)

        grads = np.empty(X_with_bias.shape)
        for idx in range(X_with_bias.shape[0]):
            grads[idx] = self._compute_gradient(X_with_bias[idx:idx+1], y[idx:idx+1])
        avg_grad = np.mean(grads, axis=0)

        Q_prev = self.loss_func(self._sigmoid(X_with_bias, self.w), y)
        for i in range(n_steps):
            idx = np.random.choice(
                np.arange(0, X_with_bias.shape[0]), batch_size, replace=False
                )
            object_X, object_y = X_with_bias[idx], y[idx]
            lr = self._get_lr(lr_, lr_type, i, decay_rate)

            for j in idx:
                X_j, y_j = X_with_bias[j:j+1], y[j:j+1]
                new_grad = self._compute_gradient(X_j, y_j)
                avg_grad += (new_grad - grads[j]) / X_with_bias.shape[0]
                grads[j] = new_grad
            self.w = self.w - lr * avg_grad

            Q_prev, flag = self._omptimizer_early_stop(X_with_bias, y, quality_limit, Q_prev)
            if flag:
                break
        return self.w

    def _momentum(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, gamma):

        X_with_bias = self._optimizer_start(X)

        velocity = np.zeros(X_with_bias.shape[1])
        Q_prev = self.loss_func(self._sigmoid(X_with_bias, self.w), y)
        for i in range(n_steps):

            lr = self._get_lr(lr_, lr_type, i, decay_rate)
            velocity = gamma * velocity + lr * self._compute_gradient(X_with_bias, y)

            self.w = self.w - velocity

            Q_prev, flag = self._omptimizer_early_stop(X_with_bias, y, quality_limit, Q_prev)
            if flag:
                break
        return self.w

    def _nesterov_accelerated_gradient(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, gamma):
        
        X_with_bias = self._optimizer_start(X)

        velocity = np.zeros(X_with_bias.shape[1])
        Q_prev = self.loss_func(self._sigmoid(X_with_bias, self.w), y)
        for i in range(n_steps):
            lr = self._get_lr(lr_, lr_type, i, decay_rate)
            w = self.w - gamma * velocity
            velocity = gamma * velocity + lr * self._compute_gradient(X_with_bias, y, w)
            self.w = self.w - velocity

            Q_prev, flag = self._omptimizer_early_stop(X_with_bias, y, quality_limit, Q_prev)
            if flag:
                break
        return self.w

    def _root_mean_square_propagation(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, alpha, const):
        
        X_with_bias = self._optimizer_start(X)

        Q_prev = self.loss_func(self._sigmoid(X_with_bias, self.w), y)
        G = np.zeros(X_with_bias.shape[1])
        for i in range(n_steps):
            lr = self._get_lr(lr_, lr_type, i, decay_rate)
            grad = self._compute_gradient(X_with_bias, y)
            G = alpha * G + (1 - alpha) * (grad * grad)

            self.w = self.w - lr * (grad / (np.sqrt(G) + const))

            Q_prev, flag = self._omptimizer_early_stop(X_with_bias, y, quality_limit, Q_prev)
            if flag:
                break
        return self.w

    def _adaptive_learning_rate(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, alpha, const):
        
        X_with_bias = self._optimizer_start(X)

        Q_prev = self.loss_func(self._sigmoid(X_with_bias, self.w), y)
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

            Q_prev, flag = self._omptimizer_early_stop(X_with_bias, y, quality_limit, Q_prev)
            if flag:
                break
        return self.w

    def _adaptive_momentum(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, gamma, alpha, const):
        
        X_with_bias = self._optimizer_start(X)

        velocity = np.zeros(X_with_bias.shape[1])
        G = np.zeros(X_with_bias.shape[1])
        Q_prev = self.loss_func(self._sigmoid(X_with_bias, self.w), y)
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

            Q_prev, flag = self._omptimizer_early_stop(X_with_bias, y, quality_limit, Q_prev)
            if flag:
                break
        return self.w

    def _nesterov_accelerated_adaptive_momentum(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, gamma, alpha, const):
        
        X_with_bias = self._optimizer_start(X)

        velocity = np.zeros(X_with_bias.shape[1])
        G = np.zeros(X_with_bias.shape[1])
        Q_prev = self.loss_func(self._sigmoid(X_with_bias, self.w), y)
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

            Q_prev, flag = self._omptimizer_early_stop(X_with_bias, y, quality_limit, Q_prev)
            if flag:
                break
        return self.w

    def _adaptive_gradient(self, X, y, n_steps, quality_limit, lr_, lr_type, decay_rate, const):
        
        X_with_bias = self._optimizer_start(X)

        r = np.zeros(X_with_bias.shape[1])
        Q_prev = self.loss_func(self._sigmoid(X_with_bias, self.w), y)
        for i in range(n_steps):

            lr = self._get_lr(lr_, lr_type, i, decay_rate)
            grad = self._compute_gradient(X_with_bias, y)

            r += grad ** 2
            self.w = self.w - lr * (1/(np.sqrt(r) + const)) * grad

            Q_prev, flag = self._omptimizer_early_stop(X_with_bias, y, quality_limit, Q_prev)
            if flag:
                break
        return self.w
    #===========================================================================
    def fit(self, X, y, 
            learning_type, 
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
        self._orig_classes=set(np.unique(y))
        if self._orig_classes <= {-1, 1}:
            y = np.where(y == -1, 0, 1)

        if learning_type in ('gradient_descent', 'gd', 'GD'):
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
    
    def predict_proba(self, X):
        X_with_bias = np.c_[X, np.ones(X.shape[0])]
        return self._sigmoid(X_with_bias, self.w)
        
    def predict(self, X, treshold=0.5):
        probabilities = self.predict_proba(X)

        ans = np.where(probabilities >= treshold, 1, 0)
        if hasattr(self, '_orig_classes') and self._orig_classes <= {-1, 1}:
            ans = np.where(ans == 0, -1, 1)
        return ans