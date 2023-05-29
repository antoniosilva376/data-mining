import numpy as np
import unittest
from sklearn.metrics import accuracy_score, precision_score
from sklearn.tree import DecisionTreeClassifier

"""The Node class represents a node in the decision tree. It has several attributes,
including feature_index, threshold, value, left, and right. These attributes store
information about the feature used to split the data at this node, the threshold value
for the split, the predicted value if this node is a leaf node, and the left and right
child nodes."""
class Node:
    def __init__(
            self,
            feature_index=None,
            threshold=None,
            value=None,
            left=None,
            right=None,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right


"""The _most_common_label function takes in a target vector y and returns the most common label in y.
If there is more than one label with the highest count, one of them is chosen randomly."""
def _most_common_label(y):
    unique, counts = np.unique(y, return_counts=True)
    max_indices = np.argwhere(counts == counts.max()).flatten()
    if len(max_indices) > 1:
        max_index = np.random.choice(max_indices)
    else:
        max_index = max_indices[0]
    return unique[max_index]

"""The _entropy function calculates the entropy of a target vector y. Entropy is a measure of impurity or disorder in a dataset.
The goal of a decision tree is to split the data in such a way that the resulting subsets have lower entropy (i.e., are more pure)
than the original dataset."""
def _entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy

"""The _gini_index function calculates the Gini index of a target vector y. The Gini index is another
measure of impurity or disorder in a dataset. Like entropy, it can be used to evaluate the quality of
a split in a decision tree."""
def _gini_index(y):
    unique, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    gini_index = 1 - sum(probabilities ** 2)
    return gini_index

"""The _information_gain function calculates the information gain of a split. Information gain measures
the reduction in entropy (or Gini index) achieved by splitting the data on a particular feature and threshold.
The function takes in several arguments, including the target vector y, a column of the feature matrix X_column,
a threshold value for the split, and the criterion used to measure impurity ('entropy' or 'gini'). It returns
the information gain achieved by splitting X_column on the given threshold."""
def _information_gain(y, X_column, threshold, criterion):
    if criterion == 'entropy':
        parent = _entropy(y)
    elif criterion == 'gini':
        parent = _gini_index(y)

    left_indices, right_indices = _split(X_column, threshold)

    if len(left_indices) == 0 or len(right_indices) == 0:
        return 0

    n = len(y)
    len_left, len_right = len(left_indices), len(right_indices)

    if criterion == 'entropy':
        left, right = (
            _entropy(y[left_indices]),
            _entropy(y[right_indices]),
        )
    elif criterion == 'gini':
        left, right = (
            _gini_index(y[left_indices]),
            _gini_index(y[right_indices]),
        )

    child = (len_left / n) * left + (len_right / n) * right

    ig = parent - child
    return ig

"""The _gain_ratio function calculates the gain ratio of a split. Gain ratio is similar to information
gain but normalizes the information gain by dividing it by the entropy of the feature used for splitting.
This can help prevent bias towards features with many possible values."""
def _gain_ratio(y, X_column, threshold):
    info_gain = _information_gain(y, X_column, threshold, 'entropy')

    left_indices, right_indices = _split(X_column, threshold)

    if len(left_indices) == 0 or len(right_indices) == 0:
        return 0

    n = len(y)
    n_left, n_right = len(left_indices), len(right_indices)
    p_left, p_right = n_left / n, n_right / n
    attr_entropy = (
            -p_left * np.log2(p_left) - p_right * np.log2(p_right)
    )

    gain_ratio = info_gain / attr_entropy
    return gain_ratio

"""The _split function takes in a column of the feature matrix X_column and a threshold value and
returns two arrays containing the indices of rows that should go to the left and right child nodes
after splitting on this feature and threshold."""
def _split(X_column, threshold):
    left_indices = np.argwhere(X_column <= threshold).flatten()
    right_indices = np.argwhere(X_column > threshold).flatten()

    return left_indices, right_indices

"""The DecisionTree class implements a decision tree classifier. It has several attributes that control its behavior,
including criterion, which specifies the criterion used to measure impurity ('entropy', 'gini', or 'gain_ratio'),
max_depth, which specifies the maximum depth of the tree, min_samples_split, which specifies the minimum number of
samples required to split an internal node, and max_leaf_nodes, which specifies the maximum number of leaf nodes in the tree."""
class DecisionTree:

    def __init__(
            self,
            criterion='entropy',
            max_depth=None,
            min_samples_split=2,
            max_leaf_nodes=20,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leaf_nodes = max_leaf_nodes

    """The fit method takes in a feature matrix X and a target vector y and builds a decision tree using these data.
    It calls the _grow_tree method to recursively build the tree."""
    def fit(self, X, y):
        self.tree_ = self._grow_tree(X, y)

    """The predict method takes in a feature matrix X and returns an array containing predicted values for each row in X.
    It calls the _predict method to make predictions for individual rows."""
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    """The _grow_tree method recursively builds a decision tree using depth-first search. It takes in several arguments,
    including a feature matrix X, a target vector y, and an optional depth argument that keeps track of how deep we are in"""
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (
                depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split
        ):
            leaf_value = _most_common_label(y)
            return Node(value=leaf_value)

        feature_indices = np.arange(n_features)
        best_feature, best_threshold = self._best_criteria(
            X, y, feature_indices
        )
        left_indices, right_indices = _split(
            X[:, best_feature], best_threshold
        )

        left = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)

        return Node(
            best_feature,
            best_threshold,
            _most_common_label(y),
            left,
            right,
        )

    """The _predict method takes in a single row of data inputs and traverses the decision tree to make a prediction.
    It starts at the root node and moves down the tree by comparing inputs to the threshold at each node and following
    the appropriate child node until it reaches a leaf node. The predicted value is then returned."""
    def _predict(self, inputs):
        node = self.tree_

        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.value

    """The _best_criteria method finds the best feature and threshold to split the data on. It does this by iterating over all
    features and all possible thresholds for each feature and calculating the information gain (or gain ratio) achieved by
    splitting on that feature and threshold. The feature and threshold with the highest information gain (or gain ratio) are returned."""
    def _best_criteria(self, X, y, feature_indices):
        best_gain = -1
        split_index, split_threshold = None, None

        for feature_index in feature_indices:
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                if self.criterion in ['entropy', 'gini']:
                    gain = _information_gain(
                        y,
                        X_column,
                        threshold,
                        self.criterion,
                    )
                elif self.criterion == 'gain_ratio':
                    gain = _gain_ratio(y, X_column, threshold)
                else:
                    raise ValueError(
                        "Invalid criterion: {}".format(self.criterion)
                    )

                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_threshold = threshold

        return split_index, split_threshold


#test class
class TestDecisionTree(unittest.TestCase):

    def generate_random_dataset(self, n_samples, n_features, n_classes):
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, size=n_samples)
        return X, y

    def test_predict(self):
        X, y = self.generate_random_dataset(1000, 10, 2)
        dt = DecisionTree(max_depth=5)
        dt.fit(X, y)
        y_pred = dt.predict(X)
        self.assertEqual(len(y_pred), len(y))

        print("Custom Decision Tree")
        print("Accuracy:", accuracy_score(y, y_pred))
        print("Precision:", precision_score(y, y_pred))

        # scikit-learn implementation
        dt_sklearn = DecisionTreeClassifier(max_depth=5)
        dt_sklearn.fit(X,y)
        y_pred_sklearn = dt_sklearn.predict(X)

        print("Scikit-learn Decision Tree")
        print("Accuracy:", accuracy_score(y, y_pred_sklearn))
        print("Precision:", precision_score(y, y_pred_sklearn))


def main():
    test_dt = TestDecisionTree()
    test_dt.test_predict()


if __name__ == "__main__":
    main()
