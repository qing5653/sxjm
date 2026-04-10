import math
import pytest
import numpy as np
from sxjm.regression import LinearRegression, PolynomialRegression, MultipleLinearRegression


class TestLinearRegression:
    def test_perfect_fit(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        model = LinearRegression().fit(x, y)
        assert math.isclose(model.slope, 2.0, rel_tol=1e-9)
        assert math.isclose(model.intercept, 0.0, abs_tol=1e-9)
        assert math.isclose(model.r_squared, 1.0, rel_tol=1e-9)

    def test_predict(self):
        x = [0, 1, 2]
        y = [1, 3, 5]
        model = LinearRegression().fit(x, y)
        pred = model.predict([3])
        assert math.isclose(pred[0], 7.0, rel_tol=1e-9)

    def test_r_squared_less_than_one(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 10, 50)
        y = 3 * x + rng.normal(0, 1, 50)
        model = LinearRegression().fit(x, y)
        assert 0.9 < model.r_squared <= 1.0

    def test_raises_on_identical_x(self):
        with pytest.raises(ValueError):
            LinearRegression().fit([1, 1, 1], [2, 3, 4])

    def test_raises_if_not_fitted(self):
        with pytest.raises(RuntimeError):
            LinearRegression().predict([1, 2, 3])

    def test_repr(self):
        model = LinearRegression().fit([1, 2], [2, 4])
        assert "slope" in repr(model)


class TestPolynomialRegression:
    def test_quadratic_perfect_fit(self):
        x = np.linspace(-3, 3, 20)
        y = 2 * x ** 2 + 3 * x - 1
        model = PolynomialRegression(degree=2).fit(x, y)
        assert math.isclose(model.r_squared, 1.0, rel_tol=1e-6)

    def test_predict(self):
        x = [0, 1, 2, 3]
        y = [0, 1, 4, 9]
        model = PolynomialRegression(degree=2).fit(x, y)
        np.testing.assert_allclose(model.predict([0, 1, 2, 3]), y, atol=1e-8)

    def test_invalid_degree(self):
        with pytest.raises(ValueError):
            PolynomialRegression(degree=0)

    def test_raises_if_not_fitted(self):
        with pytest.raises(RuntimeError):
            PolynomialRegression(degree=2).predict([1])

    def test_repr(self):
        model = PolynomialRegression(degree=3).fit([1, 2, 3, 4], [1, 8, 27, 64])
        assert "degree=3" in repr(model)


class TestMultipleLinearRegression:
    def test_perfect_fit(self):
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([5, 8, 11, 14])  # y = 2*x1 + x2 + 0 * intercept? No: 2+2*2=6...
        # Actually y = 1*x1 + 2*x2 - 1
        y = X[:, 0] + 2 * X[:, 1] - 1  # [1+4-1, 2+6-1, 3+8-1, 4+10-1] = [4, 7, 10, 13]
        model = MultipleLinearRegression().fit(X, y)
        assert math.isclose(model.r_squared, 1.0, rel_tol=1e-6)

    def test_predict(self):
        X = np.array([[1, 0], [0, 1], [1, 1]])
        y = np.array([3, 5, 8])  # y = 3*x1 + 5*x2
        model = MultipleLinearRegression().fit(X, y)
        np.testing.assert_allclose(model.predict(X), y, atol=1e-8)

    def test_raises_on_wrong_dims(self):
        with pytest.raises(ValueError):
            MultipleLinearRegression().fit([1, 2, 3], [1, 2, 3])

    def test_raises_if_not_fitted(self):
        with pytest.raises(RuntimeError):
            MultipleLinearRegression().predict([[1, 2]])
