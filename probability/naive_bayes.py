import numpy as np


class NBGaussianClassifier:
    
    def calculations(self, X, y):

        for i in range(self.classes_num):
            mask = y == self.classes[i]
            class_probability = np.sum(mask) / X.shape[0]
            class_feature_math_expectations = np.mean(X[mask], axis=0)
            class_feature_var = np.var(X[mask], axis = 0)
            class_feature_std = np.std(X[mask], axis = 0)

            self.classes_apriori_probabilities[i] = class_probability
            self.feature_math_expectations[i] = class_feature_math_expectations
            self.feature_var[i] = class_feature_var + 1e-9
            self.feature_std[i] = class_feature_std + 1e-9

    def __init__(self):
        pass

    def fit(self, X, y):
        
        n_features = X.shape[1]
        self.classes = np.unique(y)
        self.classes_num = self.classes.shape[0]

        self.feature_math_expectations = np.zeros((self.classes_num, n_features))
        self.feature_var = np.zeros((self.classes_num, n_features))
        self.feature_std = np.zeros((self.classes_num, n_features))
    
        self.classes_apriori_probabilities = np.zeros(self.classes_num)

        self.calculations(X, y)
        return self

    def predict(self, X, use_logarithm = True, penalty = None):

        penalty = np.ones(self.classes.shape) if penalty is None else penalty

        tensor_class_feature_probability = ( 1 / (np.sqrt(2 * np.pi) * self.feature_std[:, None, :]) ) * np.exp(
            -((X[None, :, :] - self.feature_math_expectations[:, None, :])**2 / (2 * self.feature_var[:, None, :]))
            )

        if use_logarithm:
            matrix_class_object_probability = np.log(penalty * self.classes_apriori_probabilities)[:, None] + \
            np.sum(np.log(tensor_class_feature_probability + 1e-9), axis=-1)
        else:
            matrix_class_object_probability = (penalty * self.classes_apriori_probabilities)[:, None] * \
                np.prod(tensor_class_feature_probability, axis=-1)

        self.predicted = matrix_class_object_probability
        predicted = np.argmax(self.predicted, axis=0)

        return self.classes[predicted]
    
    def predict_proba(self, X, use_logarithm = True, penalty = None):
        self.predict(X, use_logarithm, penalty)
        if use_logarithm:
            max_log = self.predicted.max(axis=0, keepdims=True)
            probs = np.exp(self.predicted - max_log)
            probs /= probs.sum(axis=0, keepdims=True)
        else:
            probs = self.predicted / self.predicted.sum(axis=0, keepdims=True)

        return probs