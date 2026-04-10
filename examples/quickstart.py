"""
Quick-start examples for the sxjm mathematical modeling toolkit.

Run from the repository root:

    python examples/quickstart.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

from sxjm import (
    LinearRegression,
    PolynomialRegression,
    MultipleLinearRegression,
    LinearProgramming,
    GradientDescent,
    ODESolver,
    Statistics,
)

# ── 1. Linear Regression ────────────────────────────────────────────────────
print("=" * 60)
print("1. Linear Regression")
rng = np.random.default_rng(0)
x = np.linspace(0, 10, 50)
y = 2.5 * x + 1.0 + rng.normal(0, 1, 50)
model = LinearRegression().fit(x, y)
print(f"   {model}")

# ── 2. Polynomial Regression ────────────────────────────────────────────────
print("2. Polynomial Regression (degree=3)")
x2 = np.linspace(-3, 3, 30)
y2 = x2 ** 3 - 2 * x2 + 0.5 + rng.normal(0, 0.5, 30)
poly = PolynomialRegression(degree=3).fit(x2, y2)
print(f"   {poly}")

# ── 3. Multiple Linear Regression ───────────────────────────────────────────
print("3. Multiple Linear Regression")
X = rng.uniform(0, 5, (100, 3))
y3 = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 3.0 * X[:, 2] + rng.normal(0, 0.5, 100)
mlr = MultipleLinearRegression().fit(X, y3)
print(f"   {mlr}")

# ── 4. Linear Programming ───────────────────────────────────────────────────
print("4. Linear Programming  (maximize 5x + 4y)")
# maximize 5x + 4y  s.t.  6x + 4y <= 24,  x + 2y <= 6,  x >= 0, y >= 0
c = [-5, -4]
A_ub = [[6, 4], [1, 2]]
b_ub = [24, 6]
bounds = [(0, None), (0, None)]
lp = LinearProgramming().solve(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
print(f"   optimal x = {lp.optimal_x}, obj = {-lp.optimal_value:.2f}")

# ── 5. Gradient Descent ─────────────────────────────────────────────────────
print("5. Gradient Descent  (minimize (x-3)^2 + (y+1)^2)")
f = lambda v: (v[0] - 3) ** 2 + (v[1] + 1) ** 2
grad_f = lambda v: np.array([2 * (v[0] - 3), 2 * (v[1] + 1)])
gd = GradientDescent(learning_rate=0.1, max_iterations=500).minimize(f, grad_f, [0.0, 0.0])
print(f"   {gd},  x* ≈ {gd.optimal_x}")

# ── 6. ODE Solver ───────────────────────────────────────────────────────────
print("6. ODE Solver  (simple harmonic oscillator)")
f_osc = lambda t, y: np.array([y[1], -y[0]])
solver = ODESolver(method="rk4").solve(f_osc, (0, 4 * np.pi), [1.0, 0.0], n_steps=2000)
print(f"   {solver}")
print(f"   Final state: x={solver.y[-1, 0]:.6f}, v={solver.y[-1, 1]:.6f}")

# ── 7. Statistics ───────────────────────────────────────────────────────────
print("7. Descriptive Statistics")
data = rng.normal(50, 10, 200)
stats = Statistics.describe(data)
print(f"   mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
      f"skewness={stats['skewness']:.3f}")
ci = Statistics.confidence_interval(data, confidence=0.95)
print(f"   95% CI for mean: ({ci[0]:.2f}, {ci[1]:.2f})")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Regression plot
axes[0].scatter(x, y, s=10, alpha=0.6, label="data")
axes[0].plot(x, model.predict(x), color="red", label=f"fit (R²={model.r_squared:.3f})")
axes[0].set_title("Linear Regression")
axes[0].legend()

# ODE plot
axes[1].plot(solver.t, solver.y[:, 0], label="x(t)")
axes[1].plot(solver.t, solver.y[:, 1], label="v(t)")
axes[1].set_title("Harmonic Oscillator (RK4)")
axes[1].legend()

# GD convergence
values = [f(v) for v in gd.history]
axes[2].semilogy(values)
axes[2].set_title("Gradient Descent Convergence")
axes[2].set_xlabel("iteration")
axes[2].set_ylabel("f(x)")

plt.tight_layout()
plt.savefig("examples/quickstart_output.png", dpi=100)
print("\nPlot saved to examples/quickstart_output.png")
print("=" * 60)
