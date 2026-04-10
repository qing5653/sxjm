"""
Optimization models for mathematical modeling.

Provides linear programming (via SciPy) and gradient descent implementations.
"""

import numpy as np
from scipy.optimize import linprog


class LinearProgramming:
    """Linear programming solver.

    Solves problems of the form::

        minimize    c @ x
        subject to  A_ub @ x <= b_ub   (inequality constraints)
                    A_eq @ x == b_eq   (equality constraints)
                    bounds              (variable bounds)

    Wraps :func:`scipy.optimize.linprog` with the HiGHS method.
    """

    def __init__(self):
        self.result = None
        self.optimal_value = None
        self.optimal_x = None

    def solve(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
        """Solve the linear program.

        Parameters
        ----------
        c : array-like
            Coefficients of the objective function (minimized).
        A_ub : array-like or None
            Inequality constraint matrix.
        b_ub : array-like or None
            Inequality constraint upper bounds.
        A_eq : array-like or None
            Equality constraint matrix.
        b_eq : array-like or None
            Equality constraint values.
        bounds : sequence of (min, max) or None
            Bounds for each variable.  ``None`` means no bound.

        Returns
        -------
        self
        """
        self.result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if self.result.success:
            self.optimal_value = self.result.fun
            self.optimal_x = self.result.x
        return self

    @property
    def success(self):
        """bool: Whether the solver found an optimal solution."""
        return self.result is not None and self.result.success

    def __repr__(self):
        if self.result is None:
            return "LinearProgramming(not solved)"
        status = "optimal" if self.success else self.result.message
        return f"LinearProgramming(status={status!r}, fun={self.optimal_value})"


class GradientDescent:
    """Gradient descent optimizer for unconstrained minimization.

    Minimizes a scalar function ``f(x)`` using its gradient ``grad_f(x)``.
    """

    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Parameters
        ----------
        learning_rate : float
            Step size for each gradient update.
        max_iterations : int
            Maximum number of update steps.
        tolerance : float
            Stop when the gradient norm falls below this threshold.
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.optimal_x = None
        self.optimal_value = None
        self.n_iterations = None
        self.history = None

    def minimize(self, f, grad_f, x0):
        """Run gradient descent.

        Parameters
        ----------
        f : callable
            Objective function ``f(x) -> float``.
        grad_f : callable
            Gradient of ``f``: ``grad_f(x) -> array-like``.
        x0 : array-like
            Initial point.

        Returns
        -------
        self
        """
        x = np.asarray(x0, dtype=float).copy()
        self.history = [x.copy()]

        for i in range(self.max_iterations):
            grad = np.asarray(grad_f(x), dtype=float)
            if np.linalg.norm(grad) < self.tolerance:
                self.n_iterations = i
                break
            x = x - self.learning_rate * grad
            self.history.append(x.copy())
        else:
            self.n_iterations = self.max_iterations

        self.optimal_x = x
        self.optimal_value = f(x)
        return self

    def __repr__(self):
        if self.optimal_x is None:
            return "GradientDescent(not run)"
        return (
            f"GradientDescent(iterations={self.n_iterations}, "
            f"optimal_value={self.optimal_value:.6f})"
        )
