import numpy as np
import unittest
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.priors = None
        self.means = None
        self.stds = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Calculate class priors
        self.priors = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            self.priors[i] = np.sum(y == c) / len(y)

        # Calculate class means and standard deviations
        n_features = X.shape[1]
        self.means = np.zeros((n_classes, n_features))
        self.stds = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[i, :] = X_c.mean(axis=0)
            self.stds[i, :] = X_c.std(axis=0)

    def predict(self, X):
        n_samples = X.shape[0]
        posteriors = np.zeros((n_samples, len(self.classes)))

        # Calculate class posteriors
        for i, c in enumerate(self.classes):
            prior = self.priors[i]
            likelihoods = self._pdf(X, i)
            posteriors[:, i] = prior * likelihoods

        # Return class with highest posterior probability
        return self.classes[np.argmax(posteriors, axis=1)]

    def _pdf(self, X, class_idx):
        # Calculate probability density function for a given class
        mean = self.means[class_idx]
        std = self.stds[class_idx]
        numerator = np.exp(- (X - mean) ** 2 / (2 * std ** 2))
        denominator = np.sqrt(2 * np.pi * std ** 2)
        return np.prod(numerator / denominator, axis=1)


class TestNaiveBayes(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_iris(return_X_y=True)
        self.nb = NaiveBayes()
        self.sk_nb = GaussianNB()

    def test_fit(self):
        self.nb.fit(self.X, self.y)
        self.assertIsNotNone(self.nb.classes)
        self.assertIsNotNone(self.nb.priors)
        self.assertIsNotNone(self.nb.means)
        self.assertIsNotNone(self.nb.stds)

    def test_predict(self):
        self.nb.fit(self.X, self.y)
        y_pred = self.nb.predict(self.X)
        accuracy = np.sum(y_pred == self.y) / len(y_pred)
        self.assertGreater(accuracy, 0.9)

    # compare to sklearn implementation
    def test_sklearn_comparison(self):
        self.sk_nb.fit(self.X, self.y)
        sk_y_pred = self.sk_nb.predict(self.X)
        sk_accuracy = np.sum(sk_y_pred == self.y) / len(sk_y_pred)

        self.nb.fit(self.X, self.y)
        y_pred = self.nb.predict(self.X)
        accuracy = np.sum(y_pred == self.y) / len(y_pred)

        np.testing.assert_almost_equal(accuracy, sk_accuracy)


if __name__ == '__main__':
    unittest.main()
