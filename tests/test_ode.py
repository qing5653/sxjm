import math
import pytest
import numpy as np
from sxjm.ode import ODESolver


class TestODESolver:
    def test_exponential_growth_rk4(self):
        # dy/dt = y, y(0) = 1  ->  y(t) = e^t
        f = lambda t, y: y
        solver = ODESolver(method="rk4").solve(f, (0, 2), [1.0], n_steps=1000)
        np.testing.assert_allclose(solver.y[-1, 0], math.exp(2), rtol=1e-6)

    def test_exponential_growth_euler(self):
        f = lambda t, y: y
        solver = ODESolver(method="euler").solve(f, (0, 1), [1.0], n_steps=10000)
        np.testing.assert_allclose(solver.y[-1, 0], math.exp(1), rtol=1e-3)

    def test_simple_harmonic_oscillator(self):
        # d²x/dt² + x = 0  rewritten as system:
        # dy0/dt = y1,  dy1/dt = -y0
        # Solution: x(t) = cos(t),  v(t) = -sin(t)
        f = lambda t, y: np.array([y[1], -y[0]])
        solver = ODESolver(method="rk4").solve(f, (0, 2 * math.pi), [1.0, 0.0], n_steps=2000)
        # After one full period the state should be back to initial
        np.testing.assert_allclose(solver.y[-1], [1.0, 0.0], atol=1e-4)

    def test_scalar_output(self):
        f = lambda t, y: -y
        solver = ODESolver().solve(f, (0, 1), 1.0, n_steps=100)
        assert solver.y.ndim == 1

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            ODESolver(method="invalid")

    def test_invalid_t_span(self):
        with pytest.raises(ValueError):
            ODESolver().solve(lambda t, y: y, (5, 0), [1.0])

    def test_t_array_shape(self):
        solver = ODESolver().solve(lambda t, y: y, (0, 1), [1.0], n_steps=50)
        assert len(solver.t) == 51

    def test_repr_before_solve(self):
        assert "not solved" in repr(ODESolver())

    def test_repr_after_solve(self):
        solver = ODESolver().solve(lambda t, y: y, (0, 1), [1.0])
        assert "rk4" in repr(solver)
