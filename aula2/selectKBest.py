import numpy as np
from typing import Callable
import sys

sys.path.append('..')
from aula1.dataset import Dataset
from f_regression import f_regression


# needs some correction on the callable function
class SelectKBest:
    def __init__(self, score_func: Callable, k: int):
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> None:
        #scores,
        pvalues = self.score_func(dataset).calculate()
        #self.F = []
        self.p = []
        for i in range(dataset.X.shape[1]):
            #self.F.append(scores[i])
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

# Create a SelectKBest object with f_regression scoring and k=2
selector1 = SelectKBest(score_func=f_regression, k=2)

# Fit and transform the dataset
selected_dataset1 = selector1.fit_transform(dataset1)

# Print the selected features
print(selected_dataset1.features)



