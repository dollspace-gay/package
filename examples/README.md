# Examples

This directory contains examples demonstrating the features of the kernel-regression package.

## Files

### `visual_examples.ipynb` - Interactive Visual Examples ⭐

**Recommended starting point** for new users!

A comprehensive Jupyter notebook with visual plots demonstrating all major features:

1. **Basic Nadaraya-Watson Regression** - Introduction to kernel regression
2. **Local Polynomial Regression** - Effect of polynomial orders (0, 1, 2)
3. **Boundary Bias Comparison** - How local polynomial eliminates boundary bias
4. **Bandwidth Selection** - Bias-variance tradeoff visualization
5. **Per-Dimension Bandwidth** - Automatic variable selection
6. **Heteroscedasticity Detection** - Tests for non-constant variance
7. **Confidence Intervals** - Wild bootstrap uncertainty quantification
8. **Goodness of Fit Diagnostics** - R², AIC, BIC, leverage plots
9. **2D Regression Visualization** - Multivariate regression examples

**To run:**
```bash
pip install -e ".[notebooks]"
jupyter notebook examples/visual_examples.ipynb
```

### `formal_validation.py` - Mathematical Validation

PhD-level validation tests using synthetic data where the true function is known:

- **Consistency Test**: MSE → 0 as n → ∞
- **Convergence Rate**: Verify O(n^(-4/5)) rate for d=1
- **CI Coverage**: 95% confidence intervals contain truth ~95% of time
- **Bias-Variance Tradeoff**: Verify theoretical predictions
- **Heteroscedasticity Calibration**: Test size and power
- **Boundary Bias Reduction**: Local polynomial vs Nadaraya-Watson
- **Variable Selection**: Per-dimension bandwidth identifies irrelevant features

**To run:**
```bash
python examples/formal_validation.py
```

### `heteroscedasticity_calibration.py` - Calibration Study

Detailed calibration of heteroscedasticity tests:

- **Nominal Size**: False positive rate should be ~5% at α=0.05
- **Statistical Power**: Detection rate for moderate effects
- **Power Curve**: Detection rate vs effect size

Tests evaluated:
- White test
- Breusch-Pagan test
- Dette-Munk-Wagner test (recommended for nonparametric regression)

**To run:**
```bash
python examples/heteroscedasticity_calibration.py
```

## Quick Start

For a quick visual introduction, start with `visual_examples.ipynb`.

For rigorous mathematical validation, run `formal_validation.py` to verify the implementation is correct.
