import numpy as np
import sys
from scipy.stats import f_oneway
from sklearn.feature_selection import f_classif as sk_f_classif
import unittest
sys.path.append('..')
from aula1.dataset import Dataset

class f_classif:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def calculate(self):
        classes = np.unique(self.dataset.y)
        class_data = [self.dataset.X[self.dataset.y == c] for c in classes]
        F, p = f_oneway(*class_data)
        print(p)
        return F, p


class TestFClassif(unittest.TestCase):
    def test_f_classif(self):
        # create a toy dataset
        X = np.random.randn(100, 5)
        y = np.random.choice(["A", "B", "C"], 100)
        dataset = Dataset(X=X, y=y)

        # calculate the one-way ANOVA using the f_classif class
        f = f_classif(dataset)
        Fv, pv = f.calculate()

        # calculate the one-way ANOVA using scikit-learn's f_classif function
        Fv_sk, pv_sk = sk_f_classif(X, y)

        # compare the results
        np.testing.assert_allclose(Fv, Fv_sk)
        np.testing.assert_allclose(pv, pv_sk)


if __name__ == '__main__':
    unittest.main()

"""
# create a toy dataset
X = np.random.randn(100, 5)
y = np.random.choice(["A", "B", "C"], 100)
dataset = Dataset(X=X, y=y)

# calculate the one-way ANOVA using the f_classif class
f = f_classif(dataset)
Fv, pv = f.calculate()

print("F value:", Fv)
print("p value:", pv)
"""
