"""
Regression models for mathematical modeling.

Provides linear regression, polynomial regression, and multiple linear
regression implementations using least-squares fitting.
"""

import numpy as np


class LinearRegression:
    """Simple linear regression: y = slope * x + intercept."""

    def __init__(self):
        self.slope = None
        self.intercept = None
        self.r_squared = None

    def fit(self, x, y):
        """Fit the model to data.

        Parameters
        ----------
        x : array-like
            Independent variable values.
        y : array-like
            Dependent variable values.

        Returns
        -------
        self
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        if x.ndim != 1:
            raise ValueError("x must be a 1-D array")

        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)

        denom = n * sum_x2 - sum_x ** 2
        if denom == 0:
            raise ValueError("All x values are identical; cannot fit a line")

        self.slope = (n * sum_xy - sum_x * sum_y) / denom
        self.intercept = (sum_y - self.slope * sum_x) / n

        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

        return self

    def predict(self, x):
        """Predict y values for given x.

        Parameters
        ----------
        x : array-like
            Independent variable values.

        Returns
        -------
        numpy.ndarray
            Predicted y values.
        """
        if self.slope is None:
            raise RuntimeError("Model has not been fitted yet")
        return self.slope * np.asarray(x, dtype=float) + self.intercept

    def __repr__(self):
        if self.slope is None:
            return "LinearRegression(not fitted)"
        return (
            f"LinearRegression(slope={self.slope:.4f}, "
            f"intercept={self.intercept:.4f}, "
            f"R²={self.r_squared:.4f})"
        )


class PolynomialRegression:
    """Polynomial regression: y = a_n * x^n + ... + a_1 * x + a_0."""

    def __init__(self, degree=2):
        """
        Parameters
        ----------
        degree : int
            Degree of the polynomial (>= 1).
        """
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree
        self.coefficients = None
        self.r_squared = None

    def fit(self, x, y):
        """Fit the polynomial model to data.

        Parameters
        ----------
        x : array-like
            Independent variable values.
        y : array-like
            Dependent variable values.

        Returns
        -------
        self
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        if x.ndim != 1:
            raise ValueError("x must be a 1-D array")

        self.coefficients = np.polyfit(x, y, self.degree)

        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

        return self

    def predict(self, x):
        """Predict y values for given x.

        Parameters
        ----------
        x : array-like
            Independent variable values.

        Returns
        -------
        numpy.ndarray
            Predicted y values.
        """
        if self.coefficients is None:
            raise RuntimeError("Model has not been fitted yet")
        return np.polyval(self.coefficients, np.asarray(x, dtype=float))

    def __repr__(self):
        if self.coefficients is None:
            return f"PolynomialRegression(degree={self.degree}, not fitted)"
        return (
            f"PolynomialRegression(degree={self.degree}, "
            f"R²={self.r_squared:.4f})"
        )


class MultipleLinearRegression:
    """Multiple linear regression: y = X @ beta, fitted via least squares.

    The intercept term is included automatically.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.r_squared = None

    def fit(self, X, y):
        """Fit the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.
        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1-D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        n = X.shape[0]
        ones = np.ones((n, 1))
        X_aug = np.hstack([ones, X])

        beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        self.intercept = beta[0]
        self.coefficients = beta[1:]

        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

        return self

    def predict(self, X):
        """Predict target values.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Predicted values.
        """
        if self.coefficients is None:
            raise RuntimeError("Model has not been fitted yet")
        X = np.asarray(X, dtype=float)
        return X @ self.coefficients + self.intercept

    def __repr__(self):
        if self.coefficients is None:
            return "MultipleLinearRegression(not fitted)"
        return (
            f"MultipleLinearRegression(n_features={len(self.coefficients)}, "
            f"R²={self.r_squared:.4f})"
        )
