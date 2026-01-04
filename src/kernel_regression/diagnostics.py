"""
Goodness of fit diagnostics for kernel regression.

Includes tests for homoscedasticity/heteroscedasticity and
residual analysis.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.base import BaseEstimator

from kernel_regression.kernels import get_kernel, multivariate_kernel_weights


@dataclass
class HeteroscedasticityTestResult:
    """Result of heteroscedasticity test."""

    statistic: float
    p_value: float
    is_heteroscedastic: bool
    test_name: str
    alpha: float

    def __str__(self) -> str:
        conclusion = "heteroscedastic" if self.is_heteroscedastic else "homoscedastic"
        return (
            f"{self.test_name} Test\n"
            f"  Statistic: {self.statistic:.4f}\n"
            f"  P-value: {self.p_value:.4f}\n"
            f"  Conclusion (α={self.alpha}): {conclusion}"
        )


@dataclass
class ResidualDiagnosticsResult:
    """Result of residual diagnostics."""

    residuals: NDArray[np.floating]
    standardized_residuals: NDArray[np.floating]
    mean: float
    std: float
    skewness: float
    kurtosis: float
    normality_statistic: float
    normality_p_value: float
    is_normal: bool

    def __str__(self) -> str:
        return (
            f"Residual Diagnostics\n"
            f"  Mean: {self.mean:.6f}\n"
            f"  Std: {self.std:.6f}\n"
            f"  Skewness: {self.skewness:.4f}\n"
            f"  Kurtosis: {self.kurtosis:.4f}\n"
            f"  Normality test p-value: {self.normality_p_value:.4f}\n"
            f"  Normal residuals: {self.is_normal}"
        )


def heteroscedasticity_test(
    model: BaseEstimator,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    test: Literal["white", "breusch_pagan", "goldfeld_quandt", "dette_munk_wagner"] = "white",
    alpha: float = 0.05,
    n_bootstrap: int = 500,
) -> HeteroscedasticityTestResult:
    """
    Test for heteroscedasticity in kernel regression residuals.

    Args:
        model: Fitted kernel regression model with predict method.
        X: Feature matrix of shape (n_samples, n_features).
        y: Target values of shape (n_samples,).
        test: Test to use. Options are:
            - "white": White's test (general, linear auxiliary regression)
            - "breusch_pagan": Breusch-Pagan test (linear auxiliary regression)
            - "goldfeld_quandt": Goldfeld-Quandt test (split sample F-test)
            - "dette_munk_wagner": Non-parametric test using kernel smoothing
              of squared residuals. Preferred for kernel regression models.
        alpha: Significance level for the test.
        n_bootstrap: Number of bootstrap samples for dette_munk_wagner test.

    Returns:
        Test results with statistic, p-value, and conclusion.

    Example:
        >>> model = NadarayaWatson(bandwidth=0.5).fit(X, y)
        >>> result = heteroscedasticity_test(model, X, y, test="dette_munk_wagner")
        >>> print(f"Heteroscedastic: {result.is_heteroscedastic}")
    """
    X = np.atleast_2d(X)
    y = np.asarray(y).flatten()

    # Get residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    residuals_sq = residuals**2

    if test == "white":
        return _white_test(X, residuals_sq, alpha)
    elif test == "breusch_pagan":
        return _breusch_pagan_test(X, residuals_sq, alpha)
    elif test == "goldfeld_quandt":
        return _goldfeld_quandt_test(X, y, model, alpha)
    elif test == "dette_munk_wagner":
        return _dette_munk_wagner_test(X, residuals, alpha, n_bootstrap)
    else:
        raise ValueError(f"Unknown test: {test}")


def _white_test(
    X: NDArray[np.floating],
    residuals_sq: NDArray[np.floating],
    alpha: float,
) -> HeteroscedasticityTestResult:
    """
    White's test for heteroscedasticity.

    Regresses squared residuals on X, X^2, and cross products.
    """
    n_samples, n_features = X.shape

    # Build auxiliary regression matrix
    # Include: constant, X, X^2, and cross products
    aux_cols = [np.ones(n_samples)]

    for j in range(n_features):
        aux_cols.append(X[:, j])

    for j in range(n_features):
        aux_cols.append(X[:, j] ** 2)

    for j in range(n_features):
        for k in range(j + 1, n_features):
            aux_cols.append(X[:, j] * X[:, k])

    Z = np.column_stack(aux_cols)

    # OLS regression of squared residuals on Z
    try:
        beta, residuals_aux, rank, s = np.linalg.lstsq(Z, residuals_sq, rcond=None)
        y_pred_aux = Z @ beta
        ss_reg = np.sum((y_pred_aux - np.mean(residuals_sq)) ** 2)
        ss_tot = np.sum((residuals_sq - np.mean(residuals_sq)) ** 2)
        r_squared = ss_reg / ss_tot if ss_tot > 0 else 0

        # Test statistic: n * R^2 ~ chi2(k)
        statistic = n_samples * r_squared
        df = Z.shape[1] - 1
        p_value = 1 - stats.chi2.cdf(statistic, df)
    except np.linalg.LinAlgError:
        statistic = 0.0
        p_value = 1.0

    return HeteroscedasticityTestResult(
        statistic=statistic,
        p_value=p_value,
        is_heteroscedastic=p_value < alpha,
        test_name="White",
        alpha=alpha,
    )


def _breusch_pagan_test(
    X: NDArray[np.floating],
    residuals_sq: NDArray[np.floating],
    alpha: float,
) -> HeteroscedasticityTestResult:
    """
    Breusch-Pagan test for heteroscedasticity.

    Regresses squared residuals on X.
    """
    n_samples, n_features = X.shape

    # Add constant
    Z = np.column_stack([np.ones(n_samples), X])

    # Normalize squared residuals
    sigma_sq = np.mean(residuals_sq)
    g = residuals_sq / sigma_sq

    try:
        beta, _, _, _ = np.linalg.lstsq(Z, g, rcond=None)
        g_pred = Z @ beta
        ss_reg = np.sum((g_pred - np.mean(g)) ** 2)

        # Test statistic: 0.5 * ESS ~ chi2(k)
        statistic = 0.5 * ss_reg
        df = n_features
        p_value = 1 - stats.chi2.cdf(statistic, df)
    except np.linalg.LinAlgError:
        statistic = 0.0
        p_value = 1.0

    return HeteroscedasticityTestResult(
        statistic=statistic,
        p_value=p_value,
        is_heteroscedastic=p_value < alpha,
        test_name="Breusch-Pagan",
        alpha=alpha,
    )


def _goldfeld_quandt_test(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    model: BaseEstimator,
    alpha: float,
) -> HeteroscedasticityTestResult:
    """
    Goldfeld-Quandt test for heteroscedasticity.

    Compares variance of residuals in two subsamples.
    """
    n_samples = X.shape[0]

    # Sort by first predictor (or fitted values if available)
    try:
        y_pred = model.predict(X)
        sort_idx = np.argsort(y_pred)
    except Exception:
        sort_idx = np.argsort(X[:, 0])

    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]

    # Drop middle observations (typically 1/3)
    drop_frac = 1 / 3
    n_drop = int(n_samples * drop_frac)
    n_subsample = (n_samples - n_drop) // 2

    # First subsample (low values)
    X1 = X_sorted[:n_subsample]
    y1 = y_sorted[:n_subsample]
    y1_pred = model.predict(X1)
    ss1 = np.sum((y1 - y1_pred) ** 2)

    # Second subsample (high values)
    X2 = X_sorted[-n_subsample:]
    y2 = y_sorted[-n_subsample:]
    y2_pred = model.predict(X2)
    ss2 = np.sum((y2 - y2_pred) ** 2)

    # F-test: larger variance / smaller variance
    if ss1 > 0 and ss2 > 0:
        if ss2 > ss1:
            statistic = ss2 / ss1
        else:
            statistic = ss1 / ss2
        df1 = df2 = n_subsample - 1
        p_value = 2 * (1 - stats.f.cdf(statistic, df1, df2))
        p_value = min(p_value, 1.0)
    else:
        statistic = 1.0
        p_value = 1.0

    return HeteroscedasticityTestResult(
        statistic=statistic,
        p_value=p_value,
        is_heteroscedastic=p_value < alpha,
        test_name="Goldfeld-Quandt",
        alpha=alpha,
    )


def _dette_munk_wagner_test(
    X: NDArray[np.floating],
    residuals: NDArray[np.floating],
    alpha: float,
    n_bootstrap: int = 500,
) -> HeteroscedasticityTestResult:
    """
    Dette-Munk-Wagner non-parametric test for heteroscedasticity.

    This test uses kernel smoothing to estimate the variance function
    and compares it against the assumption of constant variance using
    a permutation test to compute p-values.

    Unlike White's or Breusch-Pagan tests, this does not assume a
    linear relationship between variance and predictors.

    The permutation approach provides proper size calibration because
    under H0 (constant variance), the squared residuals have no
    relationship with x.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        residuals: Model residuals of shape (n_samples,).
        alpha: Significance level.
        n_bootstrap: Number of permutations for p-value computation.

    Returns:
        Test result with statistic and p-value.

    References:
        Dette, H., Munk, A., & Wagner, T. (1998). "Estimating the variance
        in nonparametric regression - what is a reasonable choice?"
        Journal of the Royal Statistical Society: Series B, 60(4), 751-764.
    """
    from kernel_regression.bandwidth import silverman_bandwidth

    n_samples = X.shape[0]
    residuals_sq = residuals ** 2

    # Estimate global variance
    sigma_sq_global = np.mean(residuals_sq)

    # Use first feature for 1D test, or projected values for multivariate
    if X.shape[1] == 1:
        x_order = X.flatten()
    else:
        # Project to first principal component
        X_centered = X - np.mean(X, axis=0)
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        x_order = X_centered @ Vt[0]

    # Sort by x
    sort_idx = np.argsort(x_order)
    x_sorted = x_order[sort_idx]
    resid_sq_sorted = residuals_sq[sort_idx]

    # Kernel smooth the squared residuals to get variance estimate
    bandwidth = silverman_bandwidth(x_sorted.reshape(-1, 1))[0]

    def kernel_smooth_variance_vectorized(
        x: NDArray, r_sq: NDArray, h: float
    ) -> tuple[NDArray, NDArray]:
        """Vectorized Nadaraya-Watson smoother for variance estimation."""
        u = (x[np.newaxis, :] - x[:, np.newaxis]) / h
        w = np.exp(-0.5 * u**2)
        w_sums = np.sum(w, axis=1, keepdims=True)
        w_sums = np.where(w_sums > 0, w_sums, 1.0)
        w_normalized = w / w_sums
        var_est = w_normalized @ r_sq
        return var_est, w_normalized

    # Estimate local variance and get smoothing weights
    sigma_sq_local, W = kernel_smooth_variance_vectorized(
        x_sorted, resid_sq_sorted, bandwidth
    )

    # Test statistic: integrated squared difference from global mean
    # Use trimmed statistic to reduce boundary effects
    trim = max(int(0.1 * n_samples), 10)  # Trim 10% from each end
    sigma_sq_local_trimmed = sigma_sq_local[trim:-trim]
    T_observed = np.sum((sigma_sq_local_trimmed - sigma_sq_global) ** 2) / len(sigma_sq_local_trimmed)

    # Permutation test for p-value under null hypothesis of constant variance
    # Under H0, squared residuals have no relationship with x, so permutation is valid
    T_perm = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Permute squared residuals (break any relationship with x)
        perm_idx = np.random.permutation(n_samples)
        perm_resid_sq = residuals_sq[perm_idx]

        # Smooth the permuted residuals in the original x order
        perm_resid_sq_sorted = perm_resid_sq[sort_idx]
        sigma_sq_perm = W @ perm_resid_sq_sorted

        # Compute trimmed statistic
        sigma_sq_perm_trimmed = sigma_sq_perm[trim:-trim]
        perm_global = np.mean(perm_resid_sq)
        T_perm[b] = np.sum((sigma_sq_perm_trimmed - perm_global) ** 2) / len(sigma_sq_perm_trimmed)

    # P-value: proportion of permutation values >= observed
    p_value = float(np.mean(T_perm >= T_observed))

    return HeteroscedasticityTestResult(
        statistic=float(T_observed),
        p_value=p_value,
        is_heteroscedastic=p_value < alpha,
        test_name="Dette-Munk-Wagner",
        alpha=alpha,
    )


def residual_diagnostics(
    model: BaseEstimator,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    alpha: float = 0.05,
) -> ResidualDiagnosticsResult:
    """
    Compute residual diagnostics for kernel regression.

    Parameters
    ----------
    model : fitted estimator
        Fitted kernel regression model
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Target values
    alpha : float, default=0.05
        Significance level for normality test

    Returns
    -------
    ResidualDiagnosticsResult
        Comprehensive residual diagnostics
    """
    X = np.atleast_2d(X)
    y = np.asarray(y).flatten()

    y_pred = model.predict(X)
    residuals = y - y_pred

    std = np.std(residuals, ddof=1)
    std = std if std > 0 else 1.0
    standardized = residuals / std

    # Normality test (Shapiro-Wilk for n < 5000, else D'Agostino-Pearson)
    n = len(residuals)
    if n < 5000:
        try:
            norm_stat, norm_p = stats.shapiro(residuals)
        except Exception:
            norm_stat, norm_p = 0.0, 1.0
    else:
        try:
            norm_stat, norm_p = stats.normaltest(residuals)
        except Exception:
            norm_stat, norm_p = 0.0, 1.0

    return ResidualDiagnosticsResult(
        residuals=residuals,
        standardized_residuals=standardized,
        mean=float(np.mean(residuals)),
        std=float(np.std(residuals, ddof=1)),
        skewness=float(stats.skew(residuals)),
        kurtosis=float(stats.kurtosis(residuals)),
        normality_statistic=float(norm_stat),
        normality_p_value=float(norm_p),
        is_normal=norm_p >= alpha,
    )


class GoodnessOfFit:
    """
    Comprehensive goodness of fit diagnostics for kernel regression.

    Provides R², adjusted R², AIC, BIC, and effective degrees of freedom.

    Parameters
    ----------
    model : fitted estimator
        Fitted kernel regression model
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Target values

    Attributes
    ----------
    r_squared : float
        Coefficient of determination
    adjusted_r_squared : float
        Adjusted R² accounting for effective degrees of freedom
    mse : float
        Mean squared error
    rmse : float
        Root mean squared error
    mae : float
        Mean absolute error
    aic : float
        Akaike Information Criterion
    bic : float
        Bayesian Information Criterion
    effective_df : float
        Effective degrees of freedom (trace of hat matrix)
    residual_diagnostics : ResidualDiagnosticsResult
        Detailed residual analysis
    heteroscedasticity_tests : dict
        Results of heteroscedasticity tests
    """

    def __init__(
        self,
        model: BaseEstimator,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ):
        self.model = model
        self.X = np.atleast_2d(X)
        self.y = np.asarray(y).flatten()

        self._compute_metrics()

    def _compute_hat_matrix_trace(self) -> float:
        """
        Estimate effective degrees of freedom.

        For kernel regression, this is the trace of the hat matrix H,
        where ŷ = Hy.
        """
        # Check if model has the necessary attributes
        if not hasattr(self.model, "X_") or not hasattr(self.model, "bandwidth_"):
            # Fallback: estimate from bandwidth
            return self._estimate_df_from_bandwidth()

        kernel_func = getattr(
            self.model, "kernel_func_", get_kernel("gaussian")
        )
        bandwidth = self.model.bandwidth_
        X_train = self.model.X_

        # Compute weights for training points
        weights = multivariate_kernel_weights(
            self.X, X_train, bandwidth, kernel_func
        )

        # Normalize weights (hat matrix rows)
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)
        H_rows = weights / weight_sums

        # Trace of hat matrix is sum of diagonal elements
        # For prediction at training points, diagonal is H[i, i]
        if np.array_equal(self.X, X_train):
            trace = np.sum(np.diag(H_rows))
        else:
            # Approximate trace
            trace = np.sum(np.max(H_rows, axis=1))

        return float(trace)

    def _estimate_df_from_bandwidth(self) -> float:
        """Rough estimate of df from bandwidth."""
        # Simplified: use sqrt(n) as approximation for effective df
        return float(np.sqrt(self.X.shape[0]))

    def _compute_metrics(self) -> None:
        """Compute all goodness of fit metrics."""
        y_pred = self.model.predict(self.X)
        residuals = self.y - y_pred
        n = len(self.y)

        # Basic metrics
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)

        self.r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        self.mse = float(np.mean(residuals**2))
        self.rmse = float(np.sqrt(self.mse))
        self.mae = float(np.mean(np.abs(residuals)))

        # Effective degrees of freedom
        self.effective_df = self._compute_hat_matrix_trace()

        # Adjusted R²
        df_residual = n - self.effective_df
        if df_residual > 0 and n > 1:
            self.adjusted_r_squared = 1 - (ss_res / df_residual) / (ss_tot / (n - 1))
        else:
            self.adjusted_r_squared = self.r_squared

        # Information criteria
        sigma_sq = ss_res / n
        if sigma_sq > 0:
            log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma_sq) + 1)
            self.aic = -2 * log_likelihood + 2 * self.effective_df
            self.bic = -2 * log_likelihood + np.log(n) * self.effective_df
        else:
            self.aic = np.nan
            self.bic = np.nan

        # Residual diagnostics
        self.residual_diagnostics = residual_diagnostics(
            self.model, self.X, self.y
        )

        # Heteroscedasticity tests
        self.heteroscedasticity_tests = {}
        for test_name in ["white", "breusch_pagan", "goldfeld_quandt"]:
            try:
                self.heteroscedasticity_tests[test_name] = heteroscedasticity_test(
                    self.model, self.X, self.y, test=test_name
                )
            except Exception:
                self.heteroscedasticity_tests[test_name] = None

    def is_homoscedastic(self, alpha: float = 0.05) -> bool:
        """
        Check if residuals are homoscedastic.

        Returns True if all heteroscedasticity tests fail to reject
        the null hypothesis of homoscedasticity.
        """
        for test_result in self.heteroscedasticity_tests.values():
            if test_result is not None and test_result.is_heteroscedastic:
                return False
        return True

    def get_robust_standard_errors(self) -> NDArray[np.floating]:
        """
        Compute heteroscedasticity-robust standard errors.

        Uses the HC1 (White) estimator for robust variance estimation.

        Returns
        -------
        ndarray of shape (n_samples,)
            Robust standard errors for predictions
        """
        y_pred = self.model.predict(self.X)
        residuals = self.y - y_pred
        n = len(residuals)

        # HC1 correction
        df_correction = n / (n - self.effective_df) if n > self.effective_df else 1.0

        # For kernel regression, robust SE at each point
        robust_se = np.abs(residuals) * np.sqrt(df_correction)

        return robust_se

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "=" * 60,
            "Kernel Regression Goodness of Fit Summary",
            "=" * 60,
            f"R²:                 {self.r_squared:.6f}",
            f"Adjusted R²:        {self.adjusted_r_squared:.6f}",
            f"RMSE:               {self.rmse:.6f}",
            f"MAE:                {self.mae:.6f}",
            f"Effective DF:       {self.effective_df:.2f}",
            f"AIC:                {self.aic:.2f}",
            f"BIC:                {self.bic:.2f}",
            "",
            "-" * 60,
            "Residual Diagnostics",
            "-" * 60,
            f"Mean:               {self.residual_diagnostics.mean:.6f}",
            f"Std:                {self.residual_diagnostics.std:.6f}",
            f"Skewness:           {self.residual_diagnostics.skewness:.4f}",
            f"Kurtosis:           {self.residual_diagnostics.kurtosis:.4f}",
            f"Normality p-value:  {self.residual_diagnostics.normality_p_value:.4f}",
            f"Normal residuals:   {self.residual_diagnostics.is_normal}",
            "",
            "-" * 60,
            "Heteroscedasticity Tests",
            "-" * 60,
        ]

        for name, result in self.heteroscedasticity_tests.items():
            if result is not None:
                status = "HETEROSCEDASTIC" if result.is_heteroscedastic else "homoscedastic"
                lines.append(f"{name:20s} p={result.p_value:.4f} ({status})")
            else:
                lines.append(f"{name:20s} (test failed)")

        conclusion = "HOMOSCEDASTIC" if self.is_homoscedastic() else "HETEROSCEDASTIC"
        lines.extend([
            "",
            f"Overall variance:   {conclusion}",
            "=" * 60,
        ])

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def get_leverage_values(self) -> NDArray[np.floating]:
        """
        Compute leverage values (diagonal of the hat matrix).

        The leverage h_ii measures the influence of observation i on its
        own fitted value. High leverage points have disproportionate
        influence on the regression fit.

        Returns
        -------
        ndarray of shape (n_samples,)
            Leverage values (diagonal of hat matrix H where y_hat = Hy)

        Notes
        -----
        For Nadaraya-Watson regression:
            h_ii = K(0) / sum_j K((x_i - x_j) / h)

        Leverage values should satisfy:
        - 0 <= h_ii <= 1
        - sum(h_ii) = effective degrees of freedom
        """
        return _compute_leverage_values(
            self.X, self.model.X_, self.model.bandwidth_,
            getattr(self.model, "kernel_func_", get_kernel("gaussian"))
        )


def _compute_leverage_values(
    X: NDArray[np.floating],
    X_train: NDArray[np.floating],
    bandwidth: NDArray[np.floating],
    kernel_func,
) -> NDArray[np.floating]:
    """
    Compute leverage values (hat matrix diagonal) for kernel regression.

    Args:
        X: Points to compute leverage at, shape (n_samples, n_features).
        X_train: Training points, shape (n_train, n_features).
        bandwidth: Bandwidth per feature, shape (n_features,).
        kernel_func: Kernel function.

    Returns:
        Leverage values of shape (n_samples,).
    """
    weights = multivariate_kernel_weights(X, X_train, bandwidth, kernel_func)
    weight_sums = np.sum(weights, axis=1)
    weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)

    # For points in training set, leverage is self-weight / total weight
    # For Nadaraya-Watson, self-weight is K(0) for diagonal entries
    if np.array_equal(X, X_train):
        leverage = np.diag(weights) / weight_sums
    else:
        # For prediction points not in training set
        leverage = np.max(weights, axis=1) / weight_sums

    return leverage


@dataclass
class ConfidenceIntervalResult:
    """Result of confidence interval computation."""

    predictions: NDArray[np.floating]
    lower: NDArray[np.floating]
    upper: NDArray[np.floating]
    confidence_level: float
    method: str

    def __str__(self) -> str:
        return (
            f"Confidence Intervals ({self.method})\n"
            f"  Confidence level: {self.confidence_level:.0%}\n"
            f"  Mean width: {np.mean(self.upper - self.lower):.4f}"
        )


def wild_bootstrap_confidence_intervals(
    model: BaseEstimator,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    X_pred: NDArray[np.floating] | None = None,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    distribution: Literal["rademacher", "mammen", "normal"] = "rademacher",
    bias_correction: Literal["none", "undersmooth", "rbc", "bigbrother"] = "bigbrother",
) -> ConfidenceIntervalResult:
    """
    Compute confidence intervals using Wild Bootstrap.

    The Wild Bootstrap is robust to heteroscedasticity and non-normality.
    It perturbs residuals using random weights that preserve the
    variance structure without assuming a specific error distribution.

    Args:
        model: Fitted kernel regression model.
        X: Training features of shape (n_samples, n_features).
        y: Training targets of shape (n_samples,).
        X_pred: Points for prediction, shape (n_pred, n_features).
            If None, uses X.
        confidence_level: Confidence level (default 0.95 for 95% CI).
        n_bootstrap: Number of bootstrap samples.
        distribution: Distribution for wild bootstrap weights:
            - "rademacher": +1 or -1 with prob 0.5 each (default)
            - "mammen": Two-point distribution with E[w]=0, E[w^2]=1
            - "normal": Standard normal N(0,1)
        bias_correction: Method for bias correction to improve coverage:
            - "none": Use original bandwidth (may undercover due to bias)
            - "undersmooth": Use smaller bandwidth h * 0.75 for CIs
            - "rbc": Robust Bias Correction using higher-order polynomial
              to estimate and subtract bias (CCT approach)
            - "bigbrother": (default) Combines undersmoothing with higher-order
              residuals. Uses a higher-order model to compute cleaner residuals
              plus undersmoothing for predictions. Achieves best coverage.

    Returns:
        ConfidenceIntervalResult with predictions, lower and upper bounds.

    References:
        Wu, C.F.J. (1986). "Jackknife, Bootstrap and Other Resampling
        Methods in Regression Analysis." Annals of Statistics 14, 1261-1295.

        Mammen, E. (1993). "Bootstrap and Wild Bootstrap for High
        Dimensional Linear Models." Annals of Statistics 21, 255-285.

        Calonico, Cattaneo, Titiunik (2014). "Robust Nonparametric
        Confidence Intervals for Regression-Discontinuity Designs."
        Econometrica, 82(6), 2295-2326.

    Example:
        >>> model = NadarayaWatson(bandwidth=0.5).fit(X, y)
        >>> ci = wild_bootstrap_confidence_intervals(model, X, y)
        >>> print(f"95% CI width: {np.mean(ci.upper - ci.lower):.4f}")
    """
    X = np.atleast_2d(X)
    y = np.asarray(y).flatten()

    if X_pred is None:
        X_pred = X

    X_pred = np.atleast_2d(X_pred)
    n_samples = X.shape[0]
    n_pred = X_pred.shape[0]

    # Get original bandwidth
    original_bandwidth = np.atleast_1d(model.bandwidth_)

    # Original predictions and residuals (using original model)
    y_pred_original = model.predict(X)
    residuals = y - y_pred_original

    # Handle bias correction approaches
    if bias_correction == "rbc":
        # Robust Bias Correction (CCT approach):
        # 1. Use original bandwidth for predictions
        # 2. Estimate bias using higher-order polynomial
        # 3. Correct predictions and widen CIs accordingly
        from kernel_regression.estimators import LocalPolynomialRegression

        # Fit higher-order model for bias estimation
        bias_model = LocalPolynomialRegression(
            bandwidth=original_bandwidth,
            order=2,
        ).fit(X, y)
        bias_pred = bias_model.predict(X_pred)

        # Original model predictions
        predictions = model.predict(X_pred)

        # Bias estimate: difference between low and high order fits
        bias_estimate = predictions - bias_pred

        # Bias-corrected predictions
        predictions_corrected = predictions - bias_estimate
        ci_bandwidth = original_bandwidth  # Use original bandwidth for bootstrap

    elif bias_correction == "bigbrother":
        # Big Brother approach:
        # Combines two techniques for best coverage:
        # 1. Use a higher-order model to compute residuals for the bootstrap.
        #    The higher-order model captures more of the true signal (including
        #    curvature that the lower-order model misses as "bias"), leaving
        #    purer noise in the residuals.
        # 2. Use undersmoothing for predictions and bootstrap refits to further
        #    reduce bias in the confidence intervals.
        from kernel_regression.estimators import LocalPolynomialRegression

        # Determine current model order
        current_order = getattr(model, "order_", 0)  # NW is order 0

        # Fit "big brother" model one order higher for cleaner residuals
        bigbrother_model = LocalPolynomialRegression(
            bandwidth=original_bandwidth,
            order=current_order + 1,
        ).fit(X, y)

        # Use big brother's predictions to compute cleaner residuals
        y_pred_bigbrother = bigbrother_model.predict(X)
        residuals = y - y_pred_bigbrother

        # Apply undersmoothing for predictions and bootstrap
        undersmooth_factor = 0.75
        ci_bandwidth = original_bandwidth * undersmooth_factor

        # Refit with undersmoothed bandwidth for predictions
        params = model.get_params()
        params['bandwidth'] = ci_bandwidth
        undersmooth_model = model.__class__(**params)
        undersmooth_model.fit(X, y)

        predictions_corrected = undersmooth_model.predict(X_pred)
        y_pred_original = undersmooth_model.predict(X)

    elif bias_correction == "undersmooth":
        # Undersmoothing approach:
        # Use smaller bandwidth to reduce bias, making variance dominate
        # h_CI = h * factor where factor makes bias negligible
        # Factor of 0.7-0.8 is typically sufficient
        undersmooth_factor = 0.75
        ci_bandwidth = original_bandwidth * undersmooth_factor

        # Refit model with undersmoothed bandwidth for predictions
        params = model.get_params()
        params['bandwidth'] = ci_bandwidth
        undersmooth_model = model.__class__(**params)
        undersmooth_model.fit(X, y)

        predictions_corrected = undersmooth_model.predict(X_pred)

        # Update residuals from undersmoothed model for consistency
        y_pred_original = undersmooth_model.predict(X)
        residuals = y - y_pred_original

    else:  # bias_correction == "none"
        predictions_corrected = model.predict(X_pred)
        ci_bandwidth = original_bandwidth

    # Bootstrap predictions
    bootstrap_preds = np.zeros((n_bootstrap, n_pred))

    for b in range(n_bootstrap):
        # Generate wild bootstrap weights
        if distribution == "rademacher":
            w = np.random.choice([-1.0, 1.0], size=n_samples)
        elif distribution == "mammen":
            sqrt5 = np.sqrt(5)
            p = (sqrt5 + 1) / (2 * sqrt5)
            w1 = -(sqrt5 - 1) / 2
            w2 = (sqrt5 + 1) / 2
            w = np.where(np.random.random(n_samples) < p, w1, w2)
        else:  # normal
            w = np.random.standard_normal(n_samples)

        # Perturbed response
        y_star = y_pred_original + w * residuals

        # Refit model with CI bandwidth
        params = model.get_params()
        params['bandwidth'] = ci_bandwidth
        model_star = model.__class__(**params)
        model_star.fit(X, y_star)

        # Predict at X_pred
        bootstrap_preds[b] = model_star.predict(X_pred)

    # Compute pivotal bootstrap confidence intervals
    bootstrap_deviations = bootstrap_preds - predictions_corrected

    alpha = 1 - confidence_level
    lower_q = np.percentile(bootstrap_deviations, 100 * alpha / 2, axis=0)
    upper_q = np.percentile(bootstrap_deviations, 100 * (1 - alpha / 2), axis=0)

    # Pivotal CI: swap the quantiles
    lower = predictions_corrected - upper_q
    upper = predictions_corrected - lower_q

    method_name = f"Wild Bootstrap ({distribution})"
    if bias_correction != "none":
        method_name += f" + {bias_correction.upper()}"

    return ConfidenceIntervalResult(
        predictions=predictions_corrected,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        method=method_name,
    )
