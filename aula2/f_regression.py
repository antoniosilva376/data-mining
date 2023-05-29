import numpy as np
import sys
from scipy.stats import f
import unittest
from sklearn.feature_selection import f_regression as sk_f_regression

sys.path.append('..')
from aula1.dataset import Dataset


class f_regression:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def calculate(self):
        X = self.dataset.X
        y = self.dataset.y

        # center the data
        X = X - X.mean(axis=0)
        y = y - y.mean()

        # calculate the correlation between each feature and the target
        corr = np.dot(X.T, y) / np.sqrt(np.dot(y.T, y) * np.sum(X ** 2, axis=0))

        # calculate the degrees of freedom
        degrees_of_freedom = X.shape[0] - 2

        # calculate the F-statistic for each feature
        F = corr ** 2 / (1 - corr ** 2) * degrees_of_freedom

        # calculate the p-values for each feature
        p_values = f.sf(F, 1, degrees_of_freedom)

        return F, p_values


        print(p_values[1:])
        return None, p_values[1:]  # exclude the intercept term


class TestFRegression(unittest.TestCase):
    def test_f_regression(self):
        # create a toy dataset
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        dataset = Dataset(X=X, y=y)

        # perform linear regression using the f_regression class
        f = f_regression(dataset)
        Fv, pv = f.calculate()

        # perform linear regression using scikit-learn's f_regression function
        Fv_sk, pv_sk = sk_f_regression(X, y)

        # compare the results
        np.testing.assert_allclose(Fv, Fv_sk)
        np.testing.assert_allclose(pv, pv_sk)


if __name__ == '__main__':
    unittest.main()

"""
# create a toy dataset
# generates values from 0 to 1
X1 = np.random.randn(100, 5)
y1 = np.random.randn(100)
dS = Dataset(X=X1, y=y1)

# perform linear regression using the f_regression class
f1 = f_regression(dS)
pv = f1.calculate()

print("p-values:", pv)
"""
