import numpy as np
import sys
sys.path.append('..')
from aula1.dataset import Dataset
from scipy.stats import f_oneway

class f_classif:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def calculate(self):
        classes = np.unique(self.dataset.y)
        class_data = [self.dataset.X[self.dataset.y == c] for c in classes]
        F, p = f_oneway(*class_data)
        return F, p


# create a toy dataset
X = np.random.randn(100, 5)
y = np.random.choice(["A", "B", "C"], 100)
dataset = Dataset(X=X, y=y)

# calculate the one-way ANOVA using the f_classif class
f = f_classif(dataset)
Fv, pv = f.calculate()

print("F value:", Fv)
print("p value:", pv)
