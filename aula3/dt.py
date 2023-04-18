import numpy as np

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTree:
    def __init__(self, max_depth=None, criterion='gini', min_samples_split=2):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def _best_split(self, X, y):
        if self.criterion == 'entropy':
            return self._best_split_entropy(X, y)
        elif self.criterion == 'gini':
            return self._best_split_gini(X, y)
        elif self.criterion == 'gain_ratio':
            return self._best_split_gain_ratio(X, y)

    def _best_split_entropy(self, X, y):
        n_samples, n_features = X.shape
        entropy_parent = self._entropy(y)
        best_gain = 0.0
        best_feature_index = 0
        best_threshold = 0.0
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gain = entropy_parent - self._entropy_weighted(y, X[:, feature_index], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold
        return best_feature_index, best_threshold

    def _best_split_gini(self, X, y):
        n_samples, n_features = X.shape
        gini_parent = self._gini(y)
        best_gain = 0.0
        best_feature_index = 0
        best_threshold = 0.0
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gain = gini_parent - self._gini_weighted(y, X[:, feature_index], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold
        return best_feature_index, best_threshold

    def _best_split_gain_ratio(self, X, y):
        n_samples, n_features = X.shape
        gain_ratio_parent = self._gain_ratio(y)
        best_gain_ratio = 0.0
        best_feature_index = 0
        best_threshold = 0.0
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                split_info = self._split_information(X[:, feature_index], threshold)
                if split_info == 0:
                    continue
                gain = gain_ratio_parent - self._gain_ratio_weighted(y, X[:, feature_index], threshold, split_info)
                split_info_ratio = gain / split_info
                if split_info_ratio > best_gain_ratio:
                    best_gain_ratio = split_info_ratio
                    best_feature_index = feature_index
                    best_threshold = threshold
        return best_feature_index, best_threshold
        
    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return sum(probabilities * -np.log2(probabilities))

    def _entropy_weighted(self, y, feature, threshold):
        left_indices = feature <= threshold
        right_indices = feature > threshold
        num_left = np.sum(left_indices)
        num_right = np.sum(right_indices)
        entropy_left = self._entropy(y[left_indices])
        entropy_right = self._entropy(y[right_indices])
        weight_left = num_left / len(y)
        weight_right = num_right / len(y)
        return weight_left * entropy_left + weight_right * entropy_right

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - sum(probabilities ** 2)

    def _gini_weighted(self, y, feature, threshold):
        left_indices = feature <= threshold
        right_indices = feature > threshold
        num_left = np.sum(left_indices)
        num_right = np.sum(right_indices)
        gini_left = self._gini(y[left_indices])
        gini_right = self._gini(y[right_indices])
        weight_left = num_left / len(y)
        weight_right = num_right / len(y)
        return weight_left * gini_left + weight_right * gini_right

    def _gain_ratio(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = sum(probabilities * -np.log2(probabilities))
        return entropy

    def _split_information(self, feature, threshold):
        left_indices = feature <= threshold
        right_indices = feature > threshold
        num_left = np.sum(left_indices)
        num_right = np.sum(right_indices)
        total = num_left + num_right
        if num_left == 0 or num_right == 0:
            return 0
        p_left = num_left / total
        p_right = num_right / total
        return -(p_left * np.log2(p_left) + p_right * np.log2(p_right))

    def _gain_ratio_weighted(self, y, feature, threshold, split_info):
        left_indices = feature <= threshold
        right_indices = feature > threshold
        num_left = np.sum(left_indices)
        num_right = np.sum(right_indices)
        entropy_left = self._entropy(y[left_indices])
        entropy_right = self._entropy(y[right_indices])
        weight_left = num_left / len(y)
        weight_right = num_right / len(y)
        information_left = self._split_information(feature[left_indices], threshold)
        information_right = self._split_information(feature[right_indices], threshold)
        information = weight_left * information_left + weight_right * information_right
        gain = weight_left * entropy_left + weight_right * entropy_right
        if information == 0:
            return gain
        return (gain / information) / split_info

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            best_feature_index, best_threshold = self._best_split(X, y)
            if best_feature_index is not None:
                left_indices = X[:, best_feature_index] <= best_threshold
                X_left, y_left = X[left_indices], y[left_indices]
                X_right, y_right = X[~left_indices], y
                node.feature_index = best_feature_index
                node.threshold = best_threshold
                node.left = self._grow_tree(X_left, y_left, depth=depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth=depth + 1)
        return node
