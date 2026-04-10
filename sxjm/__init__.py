"""
sxjm - Mathematical Modeling Toolkit
"""

from .regression import LinearRegression, PolynomialRegression, MultipleLinearRegression
from .optimization import LinearProgramming, GradientDescent
from .ode import ODESolver
from .statistics import Statistics

__all__ = [
    "LinearRegression",
    "PolynomialRegression",
    "MultipleLinearRegression",
    "LinearProgramming",
    "GradientDescent",
    "ODESolver",
    "Statistics",
]

__version__ = "0.1.0"
