import numpy as np


class NBMultinominalClassifier:

    def calculations(self, X, y):

        for i in range(self.classes_num):
            mask = y == self.classes[i]
            class_probability = np.sum(mask) / X.shape[0]

            N_yi = np.sum(X[mask], axis=0)
            N_y = X[mask].sum()

            self.classes_apriori_probabilities[i] = class_probability
            self.p_y_per_feature[i] = (N_yi + self.alpha) / (N_y + self.alpha * self.n_features)

    def __init__(self):
        pass

    def fit(self, X, y, alpha = 1):
        self.alpha = alpha
        self.n_features = X.shape[1]
        self.classes = np.unique(y)
        self.classes_num = self.classes.shape[0]

        self.classes_apriori_probabilities = np.zeros(self.classes_num)
        self.p_y_per_feature = np.zeros((self.classes_num, self.n_features)) # row-class, col-feature

        self.calculations(X, y)
        return self

    def predict(self, X, use_logarithm = True, penalty = None):

        penalty = np.ones(self.classes.shape) if penalty is None else penalty

        if use_logarithm:

            tensor_class_feature_probability_logarithmic = X * np.log(self.p_y_per_feature[:, None, :] + 1e-9)
            
            matrix_class_object_probability = np.log(penalty * self.classes_apriori_probabilities)[:, None] + \
            np.sum(tensor_class_feature_probability_logarithmic, axis=-1)
        else:

            tensor_class_feature_probability =  self.p_y_per_feature[:, None, :] ** X

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
            sum_probs = self.predicted.sum(axis=0, keepdims=True)
            probs = np.divide(self.predicted, sum_probs, out=np.zeros_like(self.predicted), where=sum_probs != 0)

        return probs