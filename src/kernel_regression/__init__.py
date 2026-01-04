"""
Kernel Regression Package

Multivariate Nadaraya-Watson and Local Polynomial Kernel Regression
with sklearn-compatible interface.

Features:
- Nadaraya-Watson and Local Polynomial regression
- Automatic cross-validated bandwidth and polynomial order selection
- Per-dimension bandwidth optimization
- O(n) LOOCV via hat matrix diagonal shortcut
- KDTree-accelerated neighborhood search
- Adaptive k-NN based bandwidth
- Boundary kernel corrections
- Heteroscedasticity tests (White, Breusch-Pagan, Goldfeld-Quandt, Dette-Munk-Wagner)
- Goodness of fit diagnostics
"""

from kernel_regression.bandwidth import (
    CrossValidatedBandwidth,
    RuleOfThumbBandwidth,
    loocv_hat_matrix_shortcut,
    silverman_bandwidth,
)
from kernel_regression.diagnostics import (
    ConfidenceIntervalResult,
    GoodnessOfFit,
    heteroscedasticity_test,
    residual_diagnostics,
    wild_bootstrap_confidence_intervals,
)
from kernel_regression.estimators import (
    KernelRegression,
    LocalPolynomialRegression,
    NadarayaWatson,
)
from kernel_regression.fast_search import (
    KDTreeSearch,
    adaptive_bandwidth_knn,
)
from kernel_regression.kernels import (
    epanechnikov_kernel,
    gaussian_kernel,
    tricube_kernel,
    uniform_kernel,
)

__version__ = "0.1.0"

__all__ = [
    # Estimators
    "KernelRegression",
    "NadarayaWatson",
    "LocalPolynomialRegression",
    # Kernels
    "gaussian_kernel",
    "epanechnikov_kernel",
    "uniform_kernel",
    "tricube_kernel",
    # Bandwidth
    "CrossValidatedBandwidth",
    "RuleOfThumbBandwidth",
    "silverman_bandwidth",
    "loocv_hat_matrix_shortcut",
    # Fast search
    "KDTreeSearch",
    "adaptive_bandwidth_knn",
    # Diagnostics
    "GoodnessOfFit",
    "heteroscedasticity_test",
    "residual_diagnostics",
    "wild_bootstrap_confidence_intervals",
    "ConfidenceIntervalResult",
]
