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

        # calculate the F-statistic and p-value
        F = (TSS - RSS) / df_num / (RSS / df_denom)
        p_value = 1 - f.cdf(F, df_num, df_denom)

        return p_value



# create a toy dataset
# generates values from 0 to 1
X1 = np.random.randn(100, 5)
y1 = np.random.randn(100)
dS = Dataset(X=X1, y=y1)

# perform linear regression using the f_regression class
f1 = f_regression(dS)
pv = f1.calculate()

print("p-values:", pv)
