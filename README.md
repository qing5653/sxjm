# sxjm — Mathematical Modeling Toolkit

A Python library providing common mathematical modeling building blocks:

| Module | What it provides |
|---|---|
| `regression` | Linear, polynomial, and multiple linear regression |
| `optimization` | Linear programming (LP) and gradient descent |
| `ode` | ODE solver (Euler & 4th-order Runge-Kutta) |
| `statistics` | Descriptive stats, t-tests, confidence intervals, correlation |

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from sxjm import LinearRegression, ODESolver, LinearProgramming, Statistics
import numpy as np

# ── Regression ──────────────────────────────────────────────
x = [1, 2, 3, 4, 5]
y = [2.1, 4.0, 5.9, 8.2, 9.8]
model = LinearRegression().fit(x, y)
print(model)          # LinearRegression(slope=1.9600, intercept=0.1600, R²=0.9994)
print(model.predict([6]))

# ── ODE Solver ───────────────────────────────────────────────
# dy/dt = -0.5 y,  y(0) = 10
f = lambda t, y: -0.5 * y
solver = ODESolver(method="rk4").solve(f, (0, 10), [10.0], n_steps=1000)

# ── Linear Programming ───────────────────────────────────────
# maximize 5x + 4y  s.t.  6x + 4y <= 24,  x + 2y <= 6
lp = LinearProgramming().solve(
    c=[-5, -4],
    A_ub=[[6, 4], [1, 2]],
    b_ub=[24, 6],
    bounds=[(0, None), (0, None)],
)
print(f"Optimal value: {-lp.optimal_value:.1f}")   # 21.0

# ── Statistics ───────────────────────────────────────────────
data = np.random.default_rng(0).normal(50, 10, 200)
print(Statistics.describe(data))
print(Statistics.confidence_interval(data, confidence=0.95))
```

See [`examples/quickstart.py`](examples/quickstart.py) for a full runnable demo.

## Modules

### `regression`

| Class | Description |
|---|---|
| `LinearRegression` | Ordinary least-squares: `y = slope·x + intercept` |
| `PolynomialRegression(degree)` | Polynomial least-squares fit |
| `MultipleLinearRegression` | Multi-feature OLS via `numpy.linalg.lstsq` |

All models expose `.fit(x, y)`, `.predict(x)`, and `.r_squared`.

### `optimization`

| Class | Description |
|---|---|
| `LinearProgramming` | Wraps `scipy.optimize.linprog` (HiGHS solver) |
| `GradientDescent(lr, max_iter, tol)` | First-order gradient descent |

### `ode`

`ODESolver(method)` — `method` is `'rk4'` (default) or `'euler'`.

```python
solver = ODESolver("rk4").solve(f, t_span=(0, 10), y0=[1.0, 0.0], n_steps=1000)
# solver.t  — time array
# solver.y  — solution array (n_steps+1, n_vars) or (n_steps+1,) for scalar
```

### `statistics`

All methods are static:

| Method | Description |
|---|---|
| `describe(data)` | Mean, std, IQR, skewness, kurtosis, … |
| `t_test_one_sample(data, popmean)` | One-sample t-test |
| `t_test_two_sample(data1, data2)` | Two-sample t-test (Student or Welch) |
| `confidence_interval(data, confidence)` | CI for the mean |
| `correlation(x, y)` | Pearson r and p-value |
| `normality_test(data)` | Shapiro-Wilk test |

## Testing

```bash
pip install pytest
pytest tests/ -v
```
