# Kernel Regression

Multivariate kernel regression package with sklearn-compatible interface.

Implements Nadaraya-Watson and Local Polynomial regression with automatic cross-validated bandwidth selection, heteroscedasticity diagnostics, and numerical stability guarantees.

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from kernel_regression import NadarayaWatson, LocalPolynomialRegression

# Generate data
X = np.random.randn(100, 2)
y = np.sin(X[:, 0]) + X[:, 1]**2 + 0.1 * np.random.randn(100)

# Nadaraya-Watson with automatic bandwidth selection
model = NadarayaWatson(bandwidth="cv").fit(X, y)
predictions = model.predict(X)

# Local Polynomial (order 1 = local linear)
model = LocalPolynomialRegression(order=1, bandwidth="cv").fit(X, y)
predictions = model.predict(X)
```

## Features

### Estimators
- **NadarayaWatson**: Local constant kernel regression
- **LocalPolynomialRegression**: Local polynomial regression (orders 0-3+)

### Bandwidth Selection
- **Cross-Validation**: Leave-one-out (LOOCV) or k-fold CV
- **Per-Dimension CV**: Automatic variable selection via anisotropic bandwidth
- **Rule of Thumb**: Silverman's and Scott's rules for initialization
- **O(n) LOOCV**: Hat matrix diagonal shortcut (not O(n²) refitting)

### Kernel Functions
- Gaussian (default)
- Epanechnikov (optimal for MSE)
- Tricube, Biweight, Triweight
- Uniform, Cosine

### Diagnostics
- **Heteroscedasticity Tests**: White, Breusch-Pagan, Goldfeld-Quandt, Dette-Munk-Wagner
- **Residual Diagnostics**: Normality tests, skewness, kurtosis
- **Goodness of Fit**: R², adjusted R², AIC, BIC, effective degrees of freedom
- **Wild Bootstrap**: Confidence intervals robust to heteroscedasticity
- **Fan-Yao Variance Function Estimation**: Nonparametric variance estimation
- **Conformal Calibration**: Finite-sample coverage guarantees
- **Honest Confidence Intervals**: Armstrong-Kolesár bias-adjusted critical values

### Numerical Stability
- **Tikhonov Regularization**: Ridge penalty for ill-conditioned matrices
- **scipy.linalg.lstsq**: SVD-based solver handles rank-deficient matrices
- **Boundary Corrections**: Reflection and local linear methods

## API Reference

### NadarayaWatson

```python
from kernel_regression import NadarayaWatson

model = NadarayaWatson(
    kernel="gaussian",           # Kernel function
    bandwidth="cv",              # "cv", "silverman", or float/array
    cv="loo",                    # "loo" or int for k-fold
    boundary_correction=None,    # None, "reflection", or "local_linear"
)
model.fit(X, y)
predictions = model.predict(X_new)
```

### LocalPolynomialRegression

```python
from kernel_regression import LocalPolynomialRegression

model = LocalPolynomialRegression(
    kernel="gaussian",
    bandwidth="cv",
    cv="loo",
    order=1,                     # 0=constant, 1=linear, 2=quadratic
    max_order=2,                 # Max order when order="cv"
    regularization=1e-10,        # Tikhonov regularization
)
model.fit(X, y)
predictions = model.predict(X_new)
```

### CrossValidatedBandwidth

```python
from kernel_regression import CrossValidatedBandwidth

selector = CrossValidatedBandwidth(
    kernel="gaussian",
    cv="loo",
    per_dimension=True,          # Separate bandwidth per feature
    polynomial_order=0,
)
bandwidth = selector(X, y)
print(selector.cv_results_)      # CV results dictionary
```

### Heteroscedasticity Testing

```python
from kernel_regression import heteroscedasticity_test

result = heteroscedasticity_test(
    model, X, y,
    test="dette_munk_wagner",    # "white", "breusch_pagan", "goldfeld_quandt"
    alpha=0.05,
)
print(f"p-value: {result.p_value}")
print(f"Heteroscedastic: {result.is_heteroscedastic}")
```

### Goodness of Fit

```python
from kernel_regression import GoodnessOfFit

gof = GoodnessOfFit(model, X, y)
print(f"R²: {gof.r_squared}")
print(f"Adjusted R²: {gof.adjusted_r_squared}")
print(f"AIC: {gof.aic}")
print(f"BIC: {gof.bic}")
print(f"Effective DF: {gof.effective_df}")
print(gof.summary())

# Leverage values (hat matrix diagonal)
leverage = gof.get_leverage_values()
```

### Wild Bootstrap Confidence Intervals

```python
from kernel_regression import wild_bootstrap_confidence_intervals

ci = wild_bootstrap_confidence_intervals(
    model, X, y,
    X_pred=X_new,
    confidence_level=0.95,
    n_bootstrap=1000,
    distribution="rademacher",
    bias_correction="rbc_studentized",  # "none", "undersmooth", "rbc", "bigbrother", "rbc_studentized"
    honest_cv=True,                # Armstrong-Kolesár bias-adjusted critical values
    variance_inflation=False,      # CCT variance inflation
)
print(f"Lower: {ci.lower}")
print(f"Upper: {ci.upper}")
```

### Variance Function Estimation

```python
from kernel_regression import NadarayaWatson, fan_yao_variance_estimation

model = NadarayaWatson(bandwidth='silverman').fit(X, y)
var_result = fan_yao_variance_estimation(model, X, y)

# Access variance at each point
sigma_squared = var_result.variance_estimate  # σ²(x)
sigma = var_result.std_estimate               # σ(x)
```

### Conformal Calibration

```python
from kernel_regression import conformal_calibrate_ci

# Calibrate for guaranteed finite-sample coverage
ci_calibrated = conformal_calibrate_ci(model, X, y, ci, calibration_fraction=0.2)
```

## Mathematical Standards

This package implements rigorous statistical methodology:

### 1. Hat Matrix LOOCV Shortcut (O(n) Efficiency)

Instead of O(n²) refitting, uses the formula:

```
CV = (1/n) * Σ((y_i - ŷ_i) / (1 - H_ii))²
```

where H_ii is the diagonal of the smoothing matrix.

**Note**: This O(n) shortcut applies to Nadaraya-Watson (`polynomial_order=0`). Local polynomial regression falls back to O(n²) refitting since the hat matrix structure is more complex.

### 2. Silverman/Scott Rule Initialization

Data-driven bandwidth initialization before optimization:

```
h = 1.06 * σ_robust * n^(-1/5)  (Silverman)
h = σ * n^(-1/(d+4))            (Scott)
```

**Note**: Silverman's rule uses a robust scale estimate: `σ_robust = min(std, IQR/1.349)` for stability with outliers.

### 3. Numerical Regularization

- Uses `scipy.linalg.lstsq` with `lapack_driver='gelsd'` for SVD-based solving
- Tikhonov regularization (λI) prevents singular matrix errors
- Handles collinear and rank-deficient data

### 4. Product Kernel for Multivariate Data

Per-dimension bandwidth allows anisotropic smoothing:

```
K(x) = Π_j K((x_j - X_ij) / h_j)
```

### 5. Dette-Munk-Wagner Test

Non-parametric heteroscedasticity test using kernel smoothing of squared residuals with bootstrap p-values. Does not assume linear variance structure.

### 6. Wild Bootstrap

Robust confidence intervals that preserve heteroscedasticity structure:
- Rademacher: ±1 with probability 0.5
- Mammen: Two-point distribution with E[w]=0, E[w²]=1

### 7. Big Brother Bias Correction

Uses a higher-order polynomial to compute cleaner residuals for the bootstrap, combined with undersmoothing:

```
residuals_clean = y - m_p+1(x)  # Higher-order model captures more signal
h_CI = h × 0.75                 # Undersmoothing reduces bias
```

Achieves ~92% coverage (vs ~61% without correction). For 95%+ coverage, use `rbc_studentized`.

### 8. Honest Critical Values (Armstrong-Kolesár)

Bias-adjusted critical values that account for worst-case bias:

```
cv_honest ≥ z_{1-α/2}  such that P(|Z + b| ≤ cv) ≥ 1-α for all |b| ≤ M
```

### 9. Fan-Yao Variance Estimation

Nonparametric estimation of conditional variance σ²(x) via local linear regression on squared residuals:

```
σ̂²(x) = E[ε² | X=x]  estimated by local linear on (y - ŷ)²
```

### 10. RBC Studentized (CCF 2018, 2022)

Robust bias-corrected confidence intervals using coverage-error optimal methodology from Calonico, Cattaneo, and Farrell:

```
h_CE = h × 0.6                 # Coverage-error optimal bandwidth (smaller than MSE-optimal)
bias_estimate = m_p(x) - m_{p+1}(x)  # Higher-order polynomial estimates bias
SE_rbc = SE × √(1 + V_bias)    # Variance inflation for bias estimation uncertainty
```

**Achieves ~96% coverage** (vs ~92% with Big Brother alone, ~61% without correction).

## Adversarial Test Results

The package passes all adversarial tests:

### Test 1: Boundary Bias Trap
**Challenge**: Predict y=1.0 at edge of [0,1] interval for y=x

| Method | Prediction | Bias |
|--------|------------|------|
| Nadaraya-Watson | 0.8436 | 0.1564 |
| Local Polynomial (order=1) | **1.0000** | 0.0000 |
| NW + Boundary Correction | **1.0000** | 0.0000 |

**Result**: PASS - Local polynomial eliminates boundary bias

### Test 2: Heteroscedasticity Ghost
**Challenge**: Detect variance that increases with x

| Test | p-value | Detection |
|------|---------|-----------|
| White | 0.0000 | DETECTED |
| Breusch-Pagan | 0.0000 | DETECTED |
| Dette-Munk-Wagner | 0.0000 | DETECTED |

**Result**: PASS - All tests detect heteroscedasticity

### Test 3: Curse of Irrelevance
**Challenge**: Automatic variable selection (X1=signal, X2=noise)

| Variable | Bandwidth | Role |
|----------|-----------|------|
| X1 (signal) | 0.154 | Tight fit |
| X2 (noise) | 6.269 | Smoothed out |
| Ratio | **40.8x** | |

**Result**: PASS - Irrelevant variable automatically smoothed

### Test 4: Matrix Kill (Collinearity)
**Challenge**: Fit model with X1 = X2 (perfectly collinear)

| Metric | Value |
|--------|-------|
| Fit Succeeded | True |
| Has NaN | False |
| R² | 1.0000 |

**Result**: PASS - Regularization handles singular matrices

## Test Suite

Run all tests:
```bash
pytest tests/ -v
```

Current status: **244 passed**, 2 skipped

### Test Categories
- `test_estimators.py`: Basic estimator functionality
- `test_bandwidth.py`: Bandwidth selection methods
- `test_diagnostics.py`: Goodness of fit and heteroscedasticity tests
- `test_stress.py`: Edge cases, sklearn compliance, performance
- `test_adversarial.py`: Adversarial verification

## Examples

### Visual Examples Notebook

For comprehensive visual examples with plots and detailed explanations, see the Jupyter notebook:

**[examples/visual_examples.ipynb](examples/visual_examples.ipynb)**

The notebook covers:
- Basic Nadaraya-Watson regression
- Local polynomial regression with different orders
- Boundary bias comparison
- Bandwidth selection and bias-variance tradeoff
- Per-dimension bandwidth (variable selection)
- Heteroscedasticity detection
- Wild bootstrap confidence intervals
- Goodness of fit diagnostics
- 2D regression visualization

To run the notebook:
```bash
# Note: [notebooks] extra is blocked until CVE-2025-53000 is patched
# Linux/macOS users can use notebooks-hierarchical (vulnerability is Windows-only)
pip install -e ".[notebooks-hierarchical]"
jupyter notebook examples/visual_examples.ipynb
```

> ⚠️ **Security Note (CVE-2025-53000)**: nbconvert ≤7.16.6 has a Windows code execution
> vulnerability. Do NOT run `jupyter nbconvert --to pdf` on untrusted notebooks.
> See: https://github.com/advisories/GHSA-xm59-rqc7-hhvf

### Detecting Heteroscedasticity

```python
import numpy as np
from kernel_regression import NadarayaWatson, GoodnessOfFit

# Heteroscedastic data (variance increases with x)
X = np.linspace(1, 10, 200).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.randn(200) * X.ravel() * 0.1

model = NadarayaWatson(bandwidth=0.5).fit(X, y)
gof = GoodnessOfFit(model, X, y)

print(gof.summary())
# Shows heteroscedasticity test results
```

### Per-Dimension Bandwidth Selection

```python
import numpy as np
from kernel_regression import CrossValidatedBandwidth

# X1 is signal, X2 is noise
X = np.column_stack([
    np.linspace(0, 4*np.pi, 200),  # Signal
    np.random.randn(200) * 5,       # Noise
])
y = np.sin(X[:, 0]) + 0.1 * np.random.randn(200)

selector = CrossValidatedBandwidth(cv="loo", per_dimension=True)
bandwidth = selector(X, y)

print(f"Signal bandwidth: {bandwidth[0]:.3f}")  # Small
print(f"Noise bandwidth: {bandwidth[1]:.3f}")   # Large
```

### Confidence Intervals

```python
import numpy as np
from kernel_regression import NadarayaWatson, wild_bootstrap_confidence_intervals

X = np.linspace(0, 2*np.pi, 50).reshape(-1, 1)
y = np.sin(X.ravel()) + 0.2 * np.random.randn(50)

model = NadarayaWatson(bandwidth=0.5).fit(X, y)

ci = wild_bootstrap_confidence_intervals(
    model, X, y,
    confidence_level=0.95,
    n_bootstrap=500,
)

# Plot with confidence bands
import matplotlib.pyplot as plt
plt.fill_between(X.ravel(), ci.lower, ci.upper, alpha=0.3)
plt.plot(X, ci.predictions, 'b-')
plt.scatter(X, y, c='k', s=10)
plt.show()
```

## References

- Nadaraya, E. A. (1964). "On Estimating Regression." Theory of Probability & Its Applications.
- Watson, G. S. (1964). "Smooth Regression Analysis." Sankhyā.
- Fan, J., & Gijbels, I. (1996). "Local Polynomial Modelling and Its Applications."
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning."
- Dette, H., Munk, A., & Wagner, T. (1998). "Estimating the variance in nonparametric regression."
- Wu, C.F.J. (1986). "Jackknife, Bootstrap and Other Resampling Methods in Regression Analysis."
- Armstrong, T. B., & Kolesár, M. (2020). "Simple and Honest Confidence Intervals in Nonparametric Regression." Quantitative Economics, 11(1), 1-39.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). "Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs." Econometrica, 82(6), 2295-2326.
- Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2018). "On the Effect of Bias Estimation on Coverage Accuracy in Nonparametric Inference." Journal of the American Statistical Association, 113(522), 767-779.
- Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2022). "Coverage Error Optimal Confidence Intervals for Local Polynomial Regression." Bernoulli, 28(4), 2998-3022.
- Fan, J., & Yao, Q. (1998). "Efficient Estimation of Conditional Variance Functions in Stochastic Regression." Biometrika, 85(3), 645-660.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). "Distribution-Free Predictive Inference for Regression." Journal of the American Statistical Association, 113(523), 1094-1111.
- Mammen, E. (1993). "Bootstrap and Wild Bootstrap for High Dimensional Linear Models." Annals of Statistics, 21(1), 255-285.

## License

MIT License
