import numpy as np


class Node:
    __slots__ = ('is_leaf', 'level', 'pred_value', 'feature', 'threshold', 'left', 'right', "data_idx")
    
    def __init__(self, data_idx=None):
        self.is_leaf = False
        self.level = None
        self.pred_value = None
        self.feature = None
        self.threshold = None
        self.data_idx = data_idx
        self.left = None
        self.right = None

class DecisionTreeRegression:
    
    @staticmethod
    def MSE(y):
        pred_value = np.mean(y)
        loss = ((y - pred_value) ** 2).sum(axis=0)
        return loss, pred_value
    
    @staticmethod
    def MAE(y):
        pred_value = np.median(y)
        loss = np.abs((y - pred_value)).sum(axis=0)
        return loss, pred_value

    def get_func(self, name):
        if name == 'mse':
            return self.MSE
        elif name == 'mae':
            return self.MAE
        
    def __init__(self):
        pass

    def compute_split(self, X, y):

        if self.mode in ('solo', 'bagging', 'gdboosting'):
            idx_features_to_check = np.arange(X.shape[1])
        elif self.mode == 'randomforest':
            n = np.random.randint(1, X.shape[1] + 1)
            idx_features_to_check = np.random.choice(np.arange(X.shape[1]), n, replace=False)
        

        best_IG = -1
        best_feature_idx = None
        best_treshold_value = None
        best_right_split_idx = None
        best_left_split_idx = None
        best_right_pred = None
        best_left_pred = None
        
        for feature_idx in idx_features_to_check:
            obj_idx = np.argsort(X[:, feature_idx])

            root_imp, _ = self.impurity_func(y)
            for i in range(1, obj_idx.shape[0]):

                if i < obj_idx.shape[0] - 1 and X[obj_idx[i], feature_idx] == X[obj_idx[i+1], feature_idx]:
                    continue

                left_split_idx = obj_idx[:i]
                right_split_idx = obj_idx[i:]

                if self.min_samples_split > right_split_idx.shape[0] or self.min_samples_split > left_split_idx.shape[0]:
                    continue

                right_imp, rp = self.impurity_func(y[right_split_idx])
                left_imp, lp = self.impurity_func(y[left_split_idx])
                
                n_total = y.shape[0]
                n_left = left_split_idx.shape[0]
                n_right = right_split_idx.shape[0]
                
                IG = root_imp - ((n_left / n_total) * left_imp + (n_right / n_total) * right_imp)

                if IG > best_IG and IG > self.tol:
                    best_IG = IG
                    best_feature_idx = feature_idx
                    best_treshold_value = X[:, feature_idx][obj_idx[i]]
                    best_right_split_idx = right_split_idx
                    best_left_split_idx = left_split_idx
                    best_right_pred = rp
                    best_left_pred = lp
        
        return best_IG, best_feature_idx, best_treshold_value, best_right_split_idx, best_left_split_idx, best_right_pred, best_left_pred

    def build_tree(self, level_id, level_of_nodes, X, y):
        next_level_of_nodes = []

        for node in level_of_nodes:
            if node.is_leaf:
                continue

            node_IG, node_fidx, node_trv, node_ridx, node_lidx, node_rp, node_lp = self.compute_split(
                X[node.data_idx], y[node.data_idx]
                )

            if level_id == self.max_depth or node_IG <= 0 or node_fidx is None:
                node.is_leaf = True
                _, node.pred_value = self.impurity_func(y[node.data_idx])
                continue
            
            node.feature = node_fidx
            node.threshold = node_trv
            node.level = level_id
            
            left_child = Node(data_idx = node.data_idx[node_lidx])
            left_child.pred_value = node_lp
            left_child.level = level_id + 1
            right_child = Node(data_idx = node.data_idx[node_ridx])
            right_child.pred_value = node_rp
            right_child.level = level_id + 1

            node.left = left_child
            node.right = right_child
            next_level_of_nodes.extend(( left_child, right_child ))
        return next_level_of_nodes

    def fit(
            self, X, y, max_depth=10, min_samples_split=1, tol=0.01, impurity_func='mse', mode='solo'
            ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tol = tol
        self.impurity_func = self.get_func(impurity_func)
        self.mode = mode

        
        root = Node(data_idx=np.arange(X.shape[0]))
        self.tree = [[root]]
        act_level = 0
        while act_level <= self.max_depth:
            act_level += 1
            new_level = self.build_tree(act_level, self.tree[-1], X, y)
            
            if new_level:
                self.tree.append(new_level)
            else:
                break
        for node in self.tree[-1]:
            if not node.is_leaf:
                node.is_leaf = True
                if node.pred_value is None:
                    _, node.pred_value = self.impurity_func(y[node.data_idx])
        return self

    def predict(self, X):
        ans = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            curr_node = self.tree[0][0]

            while not curr_node.is_leaf:
                if X[i, curr_node.feature] >= curr_node.threshold:
                    curr_node = curr_node.right
                else:
                    curr_node = curr_node.left
            else:
                ans[i] = curr_node.pred_value
        return ans