import numpy as np
class StatisticCollector():
    __slots__ = ('TP', 'TN', 'FP', 'FN')
    ra_rule = lambda x, treshold: x >= treshold
    def __init__(self, y_true, y_pred, pos_label=None, treshold=None, auc=False):

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        
        for i in range(y_true.shape[0]):
            a, b = y_true[i], y_pred[i]

            if auc:
                b = pos_label if StatisticCollector.ra_rule(b, treshold) else None # None=negative_class

            if a == pos_label:
                if a == b:
                    self.TP += 1
                else:
                    self.FN += 1
            else:
                if a == b or b is None:
                    self.TN += 1
                else:
                    self.FP += 1
            

def accuracy(y_true, y_pred, get_stats=False, pos_label=1):
    stc = StatisticCollector(y_true, y_pred, pos_label)
    _accuracy = (stc.TP + stc.TN) / (stc.TP + stc.TN + stc.FP + stc.FN) if (stc.TP + stc.TN + stc.FP + stc.FN) else 0.0 
    if get_stats:
        return _accuracy, stc
    return _accuracy

def precision(y_true, y_pred, get_stats=False, pos_label=1):
    stc = StatisticCollector(y_true, y_pred, pos_label)
    _precision = (stc.TP) / (stc.TP + stc.FP) if (stc.TP + stc.FP) else 0.0
    if get_stats:
        return _precision, stc
    return _precision

def recall(y_true, y_pred, get_stats=False, pos_label=1):
    stc = StatisticCollector(y_true, y_pred, pos_label)
    _recall = (stc.TP) / (stc.TP + stc.FN) if (stc.TP + stc.FN) else 0.0
    if get_stats:
        return _recall, stc
    return _recall

def specificity(y_true, y_pred, get_stats=False, pos_label=1):
    stc = StatisticCollector(y_true, y_pred, pos_label)
    _specificity = (stc.TN) / (stc.TN + stc.FP) if (stc.TN + stc.FP) else 0.0
    if get_stats:
        return _specificity, stc
    return _specificity

def fscore(y_true, y_pred, beta=1, get_stats=False, pos_label=1):
    stc = StatisticCollector(y_true, y_pred, pos_label)
    p = stc.TP / (stc.TP + stc.FP) if (stc.TP + stc.FP) else 0.0
    r = stc.TP / (stc.TP + stc.FN) if (stc.TP + stc.FN) else 0.0
    denom = beta**2 * p + r
    _fscore = (1 + beta**2) * p * r / denom if denom else 0.0
    if get_stats:
        return _fscore, stc
    return _fscore

def general_binary_analysis(y_true, y_pred, beta=1, pos_label=1):
    stc = StatisticCollector(y_true, y_pred, pos_label)
    
    acc = (stc.TP + stc.TN) / (stc.TP + stc.TN + stc.FP + stc.FN) if (stc.TP + stc.TN + stc.FP + stc.FN) else 0.0
    p = stc.TP / (stc.TP + stc.FP) if (stc.TP + stc.FP) else 0.0
    r = stc.TP / (stc.TP + stc.FN) if (stc.TP + stc.FN) else 0.0
    spec = stc.TN / (stc.TN + stc.FP) if (stc.TN + stc.FP) else 0.0
    denom = beta**2 * p + r
    f = (1 + beta**2) * p * r / denom if denom else 0.0
    
    return {
        'accuracy': acc,
        'precision': p,
        'recall': r,
        'specificity': spec,
        f'f{beta}_score': f,
        'TP': stc.TP, 'TN': stc.TN, 'FP': stc.FP, 'FN': stc.FN
    }

def roc_auc(y_true, y_pred, get_stats=False, pos_value=1, treshold=None):
    if treshold is None:
        treshold = np.sort(np.unique(y_pred))[::-1]
    treshold=np.array(treshold, ndmin=1)
    _sorted_indices = np.argsort(y_pred)[::-1]
    y_true = y_true[_sorted_indices]
    y_pred = y_pred[_sorted_indices]
    _stats = np.zeros((treshold.shape[0], 3))
    for i in range(treshold.shape[0]):
        stc = StatisticCollector(y_true, y_pred, pos_value, treshold[i], auc=True)
        TPR = stc.TP / (stc.TP + stc.FN) if (stc.TP + stc.FN) else 0.0
        FPR = stc.FP / (stc.TN + stc.FP) if (stc.TN + stc.FP) else 0.0
        _stats[i] = np.array((treshold[i], TPR, FPR))
    if get_stats:
        return np.trapz(_stats[:, 1], _stats[:, 2]), _stats
    return np.trapz(_stats[:, 1], _stats[:, 2])

def auc_prc(y_true, y_pred, get_stats=False, pos_value=1, treshold=None):
    if treshold is None:
        treshold = np.sort(np.unique(y_pred))[::-1]
    treshold=np.array(treshold, ndmin=1)
    _sorted_indices = np.argsort(y_pred)[::-1]
    y_true = y_true[_sorted_indices]
    y_pred = y_pred[_sorted_indices]
    _stats = np.zeros((treshold.shape[0], 3))
    for i in range(treshold.shape[0]):
        stc = StatisticCollector(y_true, y_pred, pos_value, treshold[i], auc=True)
        r = stc.TP / (stc.TP + stc.FN) if (stc.TP + stc.FN) else 0.0
        p = stc.TP / (stc.TP + stc.FP) if (stc.TP + stc.FP) else 0.0
        _stats[i] = np.array((treshold[i], p, r))
    if get_stats:
        return np.trapz(_stats[:, 1], _stats[:, 2]), _stats
    return np.trapz(_stats[:, 1], _stats[:, 2])


