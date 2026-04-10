"""
ODE solver for mathematical modeling.

Implements Euler's method and the classical 4th-order Runge-Kutta (RK4)
method for solving first-order systems of ordinary differential equations::

    dy/dt = f(t, y),   y(t0) = y0
"""

import numpy as np


class ODESolver:
    """Solver for first-order ODE systems.

    Supports Euler's method and the 4th-order Runge-Kutta method.

    Parameters
    ----------
    method : {'rk4', 'euler'}
        Numerical integration method.  Defaults to ``'rk4'``.
    """

    METHODS = ("rk4", "euler")

    def __init__(self, method="rk4"):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS!r}")
        self.method = method
        self.t = None
        self.y = None

    def solve(self, f, t_span, y0, n_steps=1000):
        """Integrate the ODE system.

        Parameters
        ----------
        f : callable
            Right-hand side ``f(t, y) -> array-like``.  ``y`` is a 1-D
            array even for scalar problems.
        t_span : tuple of float
            ``(t0, tf)`` — start and end times.
        y0 : array-like
            Initial conditions.
        n_steps : int
            Number of integration steps.

        Returns
        -------
        self
        """
        t0, tf = float(t_span[0]), float(t_span[1])
        if tf <= t0:
            raise ValueError("t_span[1] must be greater than t_span[0]")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")

        y0 = np.asarray(y0, dtype=float)
        scalar = y0.ndim == 0
        y0 = np.atleast_1d(y0)

        dt = (tf - t0) / n_steps
        self.t = np.linspace(t0, tf, n_steps + 1)
        self.y = np.empty((n_steps + 1, len(y0)))
        self.y[0] = y0

        integrate = self._euler_step if self.method == "euler" else self._rk4_step

        for i in range(n_steps):
            self.y[i + 1] = integrate(f, self.t[i], self.y[i], dt)

        if scalar:
            self.y = self.y[:, 0]

        return self

    @staticmethod
    def _euler_step(f, t, y, dt):
        return y + dt * np.asarray(f(t, y), dtype=float)

    @staticmethod
    def _rk4_step(f, t, y, dt):
        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + dt / 2, y + dt * k1 / 2), dtype=float)
        k3 = np.asarray(f(t + dt / 2, y + dt * k2 / 2), dtype=float)
        k4 = np.asarray(f(t + dt, y + dt * k3), dtype=float)
        return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def __repr__(self):
        if self.t is None:
            return f"ODESolver(method={self.method!r}, not solved)"
        return (
            f"ODESolver(method={self.method!r}, "
            f"t=[{self.t[0]}, {self.t[-1]}], "
            f"n_steps={len(self.t) - 1})"
        )
