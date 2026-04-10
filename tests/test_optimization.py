import math
import pytest
import numpy as np
from sxjm.optimization import LinearProgramming, GradientDescent


class TestLinearProgramming:
    def test_simple_minimization(self):
        # minimize -x - 2y  subject to x + y <= 4, x >= 0, y >= 0
        c = [-1, -2]
        A_ub = [[1, 1]]
        b_ub = [4]
        bounds = [(0, None), (0, None)]
        lp = LinearProgramming().solve(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        assert lp.success
        # Optimal: x=0, y=4  -> fun = -8
        assert math.isclose(lp.optimal_value, -8.0, rel_tol=1e-6)

    def test_equality_constraint(self):
        # minimize x + y  subject to x + y = 3, x >= 0, y >= 0
        c = [1, 1]
        A_eq = [[1, 1]]
        b_eq = [3]
        bounds = [(0, None), (0, None)]
        lp = LinearProgramming().solve(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        assert lp.success
        assert math.isclose(lp.optimal_value, 3.0, rel_tol=1e-6)

    def test_repr(self):
        lp = LinearProgramming()
        assert "not solved" in repr(lp)
        lp.solve([-1], bounds=[(0, 10)])
        assert "optimal" in repr(lp)


class TestGradientDescent:
    def test_minimize_quadratic(self):
        # f(x) = x^2, minimum at x=0
        f = lambda x: float(x[0] ** 2)
        grad_f = lambda x: np.array([2 * x[0]])
        gd = GradientDescent(learning_rate=0.1, max_iterations=500).minimize(
            f, grad_f, x0=[5.0]
        )
        assert math.isclose(gd.optimal_x[0], 0.0, abs_tol=1e-4)
        assert math.isclose(gd.optimal_value, 0.0, abs_tol=1e-8)

    def test_minimize_rosenbrock_like(self):
        # f(x, y) = (x-1)^2 + (y-2)^2
        f = lambda x: (x[0] - 1) ** 2 + (x[1] - 2) ** 2
        grad_f = lambda x: np.array([2 * (x[0] - 1), 2 * (x[1] - 2)])
        gd = GradientDescent(learning_rate=0.1, max_iterations=1000).minimize(
            f, grad_f, x0=[0.0, 0.0]
        )
        np.testing.assert_allclose(gd.optimal_x, [1.0, 2.0], atol=1e-4)

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError):
            GradientDescent(learning_rate=-0.1)

    def test_invalid_max_iterations(self):
        with pytest.raises(ValueError):
            GradientDescent(max_iterations=0)

    def test_repr_before_run(self):
        assert "not run" in repr(GradientDescent())

    def test_repr_after_run(self):
        f = lambda x: x[0] ** 2
        grad_f = lambda x: np.array([2 * x[0]])
        gd = GradientDescent().minimize(f, grad_f, [1.0])
        assert "iterations" in repr(gd)
