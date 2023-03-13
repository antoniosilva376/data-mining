import numpy as np
import sys
from scipy.stats import f

sys.path.append('..')
from aula1.dataset import Dataset


class f_regression:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def calculate(self):
        X = self.dataset.X
        y = self.dataset.y

        # add a column of ones to X to account for the intercept term
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # calculate the coefficients of the linear regression model
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

        # calculate the residual sum of squares
        e = y - X @ beta_hat
        RSS = e.T @ e

        # calculate the total sum of squares
        y_bar = y.mean()
        TSS = ((y - y_bar) ** 2).sum()

        # calculate the degrees of freedom for the numerator and denominator
        df_num = X.shape[1] - 1
        df_denom = X.shape[0] - X.shape[1]

        # calculate the F-statistic for each feature
        F = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            X_i = X[:, i]
            beta_i = beta_hat[i]
            e_i = y - beta_i * X_i
            RSS_i = e_i.T @ e_i
            F[i] = ((TSS - RSS_i) / df_num) / (RSS_i / df_denom)

        # calculate the p-values for each feature
        p_values = [1 - f.cdf(f_stat, df_num, df_denom) for f_stat in F]

        return p_values[1:]  # exclude the intercept term



# create a toy dataset
# generates values from 0 to 1
X1 = np.random.randn(100, 5)
y1 = np.random.randn(100)
dS = Dataset(X=X1, y=y1)

# perform linear regression using the f_regression class
f1 = f_regression(dS)
pv = f1.calculate()

print("p-values:", pv)
