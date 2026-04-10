from setuptools import setup, find_packages

setup(
    name="sxjm",
    version="0.1.0",
    description="Mathematical modeling toolkit",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "sympy>=1.12",
    ],
)
