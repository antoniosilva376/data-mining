import numpy as np
from typing import Callable
import sys

sys.path.append('..')
from aula1.dataset import Dataset
from f_regression import f_regression
from f_classif import f_classif


# needs some correction on the callable function
class SelectKBest:
    def __init__(self, score_func: Callable, k: int):
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> None:
        scores, pvalues = self.score_func(dataset).calculate()
        self.F = []
        self.p = []
        for i in range(dataset.X.shape[1]):
            if self.F:
                self.F.append(scores[i])
            self.p.append(pvalues[i])

    def transform(self, dataset):
        X, y = dataset.X, dataset.y
        sorted_indices = np.argsort(self.p)
        selected_indices = sorted_indices[:self.k]
        selected_features = [dataset.features[i] for i in selected_indices]
        selected_X = X[:, selected_indices]
        selected_dataset = Dataset(X=selected_X, y=y, features=selected_features, label=dataset.label)
        return selected_dataset

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)


# Create a random dataset with 5 features and 100 samples
X1 = np.random.randn(100, 5)
y1 = np.random.randn(100)
features = ['f1', 'f2', 'f3', 'f4', 'f5']
label = 'target'
dataset1 = Dataset(X=X1, y=y1, features=features, label=label)

X2 = np.random.randn(100, 5)
y2 = np.random.choice(["A", "B", "C"], 100)
features = ['f1', 'f2', 'f3', 'f4', 'f5']
label = 'target'
dataset2 = Dataset(X=X2, y=y2,features=features, label=label)

# Create a SelectKBest object with f_regression scoring and k=2
selector1 = SelectKBest(score_func=f_regression, k=2)
selector2 = SelectKBest(score_func=f_classif, k=2)

# Fit and transform the dataset
selected_dataset1 = selector1.fit_transform(dataset1)
selected_dataset2 = selector2.fit_transform(dataset2)

# Print the selected features
print("exemplo f_regression:")
print(selected_dataset1.features)

print("\nexemplo f_classif:")
print(selected_dataset2.features)


