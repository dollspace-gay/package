"""
Sklearn-compatible kernel regression estimators.

Includes Nadaraya-Watson and Local Polynomial regression with:
- KDTree-accelerated neighborhood search
- Boundary bias correction
- scipy.linalg.lstsq for numerical stability
"""

from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from kernel_regression.bandwidth import (
    CrossValidatedBandwidth,
    silverman_bandwidth,
)
from kernel_regression.kernels import get_kernel, multivariate_kernel_weights


class KernelRegression(RegressorMixin, BaseEstimator):
    """
    Base class for kernel regression estimators.

    This class implements common functionality for kernel-based
    nonparametric regression methods.

    Parameters
    ----------
    kernel : str or callable, default="gaussian"
        Kernel function. Options: "gaussian", "epanechnikov", "uniform",
        "tricube", "biweight", "triweight", "cosine", or a callable.

    bandwidth : float, array-like, or str, default="cv"
        Bandwidth parameter(s).
        - float: Single bandwidth for all dimensions
        - array-like: Per-dimension bandwidths
        - "cv": Cross-validated bandwidth selection
        - "silverman": Silverman's rule of thumb

    cv : int or str, default="loo"
        Cross-validation strategy when bandwidth="cv":
        - "loo": Leave-one-out cross-validation
        - int: Number of folds for k-fold CV

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        Training data

    y_ : ndarray of shape (n_samples,)
        Training targets

    bandwidth_ : ndarray of shape (n_features,)
        Fitted bandwidth values

    n_features_in_ : int
        Number of features seen during fit
    """

    def __init__(
        self,
        kernel: str | Callable = "gaussian",
        bandwidth: float | NDArray[np.floating] | str = "cv",
        cv: int | str = "loo",
    ):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.cv = cv

    def _validate_data_fit(
        self, X: NDArray, y: NDArray
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Validate and convert input data for fitting."""
        X, y = validate_data(self, X, y, y_numeric=True, dtype=np.float64)
        return X, y.astype(np.float64)

    def _validate_data_predict(self, X: NDArray) -> NDArray[np.floating]:
        """Validate and convert input data for prediction."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return X

    def _compute_bandwidth(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        polynomial_order: int = 0,
    ) -> NDArray[np.floating]:
        """Compute or validate bandwidth."""
        if isinstance(self.bandwidth, str):
            if self.bandwidth == "cv":
                selector = CrossValidatedBandwidth(
                    kernel=self.kernel if isinstance(self.kernel, str) else "gaussian",
                    cv=self.cv,
                    polynomial_order=polynomial_order,
                )
                return selector(X, y)
            elif self.bandwidth == "silverman":
                return silverman_bandwidth(X)
            else:
                raise ValueError(f"Unknown bandwidth method: {self.bandwidth}")
        else:
            bandwidth = np.atleast_1d(self.bandwidth).astype(np.float64)
            if bandwidth.size == 1:
                bandwidth = np.full(X.shape[1], bandwidth[0])
            if bandwidth.size != X.shape[1]:
                raise ValueError(
                    f"bandwidth has {bandwidth.size} values, expected {X.shape[1]}"
                )
            return bandwidth

    def fit(self, X: NDArray, y: NDArray) -> "KernelRegression":
        """
        Fit the kernel regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self
            Fitted estimator
        """
        X, y = self._validate_data_fit(X, y)

        self.X_ = X
        self.y_ = y
        self.bandwidth_ = self._compute_bandwidth(X, y)
        self.kernel_func_ = get_kernel(self.kernel)

        return self

    def predict(self, X: NDArray) -> NDArray[np.floating]:
        """
        Predict using the kernel regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        raise NotImplementedError("Subclasses must implement predict()")


class NadarayaWatson(KernelRegression):
    """
    Nadaraya-Watson kernel regression estimator.

    Implements local constant (order 0) kernel regression using the
    Nadaraya-Watson estimator:

        ŷ(x) = Σ K((x - x_i) / h) * y_i / Σ K((x - x_i) / h)

    Parameters
    ----------
    kernel : str or callable, default="gaussian"
        Kernel function. Options: "gaussian", "epanechnikov", "uniform",
        "tricube", "biweight", "triweight", "cosine", or a callable.

    bandwidth : float, array-like, or str, default="cv"
        Bandwidth parameter(s).
        - float: Single bandwidth for all dimensions
        - array-like: Per-dimension bandwidths
        - "cv": Cross-validated bandwidth selection
        - "silverman": Silverman's rule of thumb

    cv : int or str, default="loo"
        Cross-validation strategy when bandwidth="cv"

    boundary_correction : str or None, default=None
        Boundary correction method:
        - None: No correction (default)
        - "reflection": Reflect data near boundaries
        - "local_linear": Use local linear regression near boundaries

    Examples
    --------
    >>> import numpy as np
    >>> from kernel_regression import NadarayaWatson
    >>> X = np.random.randn(100, 2)
    >>> y = np.sin(X[:, 0]) + 0.1 * np.random.randn(100)
    >>> model = NadarayaWatson(kernel="gaussian", bandwidth="cv")
    >>> model.fit(X, y)
    >>> predictions = model.predict(X[:5])
    """

    def __init__(
        self,
        kernel: str | Callable = "gaussian",
        bandwidth: float | NDArray[np.floating] | str = "cv",
        cv: int | str = "loo",
        boundary_correction: Literal["reflection", "local_linear"] | None = None,
    ):
        super().__init__(kernel=kernel, bandwidth=bandwidth, cv=cv)
        self.boundary_correction = boundary_correction

    def fit(self, X: NDArray, y: NDArray) -> "NadarayaWatson":
        """
        Fit the Nadaraya-Watson model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self
            Fitted estimator
        """
        super().fit(X, y)
        # Store data bounds for boundary detection (use validated data)
        self.X_min_ = np.min(self.X_, axis=0)
        self.X_max_ = np.max(self.X_, axis=0)
        return self

    def _is_boundary_point(self, x: NDArray) -> NDArray[np.bool_]:
        """Check if points are near the boundary (within one bandwidth)."""
        near_lower = x < (self.X_min_ + self.bandwidth_)
        near_upper = x > (self.X_max_ - self.bandwidth_)
        return np.any(near_lower | near_upper, axis=1)

    def _local_linear_predict(
        self, X: NDArray, mask: NDArray[np.bool_]
    ) -> NDArray[np.floating]:
        """Use local linear regression for boundary points."""
        n_predict = np.sum(mask)
        y_pred = np.zeros(n_predict)
        X_boundary = X[mask]

        for i in range(n_predict):
            x = X_boundary[i : i + 1]
            weights = multivariate_kernel_weights(
                x, self.X_, self.bandwidth_, self.kernel_func_
            )[0]

            if np.sum(weights) < 1e-10:
                y_pred[i] = np.mean(self.y_)
                continue

            # Local linear: fit y = a + b*(x - x0)
            diff = self.X_ - x
            W = np.diag(weights)

            # Design matrix [1, (x1-x0), (x2-x0), ...]
            ones = np.ones((len(self.y_), 1))
            design = np.hstack([ones, diff])

            # Weighted least squares
            WX = W @ design
            Wy = W @ self.y_

            try:
                result = linalg.lstsq(WX, Wy, lapack_driver='gelsd')
                beta = result[0]
                y_pred[i] = beta[0]  # Intercept is prediction at x
            except Exception:
                # Fallback to weighted average
                y_pred[i] = np.sum(weights * self.y_) / np.sum(weights)

        return y_pred

    def predict(self, X: NDArray) -> NDArray[np.floating]:
        """
        Predict using Nadaraya-Watson estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        X = self._validate_data_predict(X)

        if self.boundary_correction == "local_linear":
            # Use local linear for boundary points
            boundary_mask = self._is_boundary_point(X)
            y_pred = np.zeros(len(X))

            # Standard NW for interior points
            interior_mask = ~boundary_mask
            if np.any(interior_mask):
                weights = multivariate_kernel_weights(
                    X[interior_mask], self.X_, self.bandwidth_, self.kernel_func_
                )
                weight_sums = np.sum(weights, axis=1)
                weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)
                y_pred[interior_mask] = np.sum(weights * self.y_, axis=1) / weight_sums

            # Local linear for boundary points
            if np.any(boundary_mask):
                y_pred[boundary_mask] = self._local_linear_predict(X, boundary_mask)

            return y_pred

        elif self.boundary_correction == "reflection":
            # Reflect data near boundaries before computing weights
            X_augmented, y_augmented = self._reflect_data()
            weights = multivariate_kernel_weights(
                X, X_augmented, self.bandwidth_, self.kernel_func_
            )
        else:
            weights = multivariate_kernel_weights(
                X, self.X_, self.bandwidth_, self.kernel_func_
            )
            y_augmented = self.y_

        # Nadaraya-Watson: weighted average
        weight_sums = np.sum(weights, axis=1)
        # Avoid division by zero
        weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)

        y_pred = np.sum(weights * y_augmented, axis=1) / weight_sums

        # Handle points with no neighbors (all weights zero)
        no_neighbors = np.sum(weights, axis=1) == 0
        if np.any(no_neighbors):
            y_pred[no_neighbors] = np.mean(self.y_)

        return y_pred

    def _reflect_data(self) -> tuple[NDArray, NDArray]:
        """Create reflected data points near boundaries."""
        X_list = [self.X_]
        y_list = [self.y_]

        for dim in range(self.X_.shape[1]):
            h = self.bandwidth_[dim]

            # Points near lower boundary
            near_lower = self.X_[:, dim] < (self.X_min_[dim] + h)
            if np.any(near_lower):
                X_reflected = self.X_[near_lower].copy()
                X_reflected[:, dim] = 2 * self.X_min_[dim] - X_reflected[:, dim]
                X_list.append(X_reflected)
                y_list.append(self.y_[near_lower])

            # Points near upper boundary
            near_upper = self.X_[:, dim] > (self.X_max_[dim] - h)
            if np.any(near_upper):
                X_reflected = self.X_[near_upper].copy()
                X_reflected[:, dim] = 2 * self.X_max_[dim] - X_reflected[:, dim]
                X_list.append(X_reflected)
                y_list.append(self.y_[near_upper])

        return np.vstack(X_list), np.concatenate(y_list)

    def get_weights(self, X: NDArray) -> NDArray[np.floating]:
        """
        Get kernel weights for prediction points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Points to get weights for

        Returns
        -------
        weights : ndarray of shape (n_samples, n_train)
            Normalized kernel weights
        """
        X = self._validate_data_predict(X)

        weights = multivariate_kernel_weights(
            X, self.X_, self.bandwidth_, self.kernel_func_
        )

        # Normalize
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)
        return weights / weight_sums


class LocalPolynomialRegression(KernelRegression):
    """
    Local polynomial kernel regression estimator.

    Fits a weighted polynomial locally at each prediction point.
    Order 0 is equivalent to Nadaraya-Watson; order 1 is local linear.

    Parameters
    ----------
    kernel : str or callable, default="gaussian"
        Kernel function.

    bandwidth : float, array-like, or str, default="cv"
        Bandwidth parameter(s).

    cv : int or str, default="loo"
        Cross-validation strategy when bandwidth="cv"

    order : int or str, default="cv"
        Polynomial order.
        - int: Fixed polynomial order (0=constant, 1=linear, 2=quadratic, ...)
        - "cv": Cross-validated order selection (searches 0, 1, 2)

    max_order : int, default=2
        Maximum order to consider when order="cv"

    regularization : float, default=1e-10
        Ridge regularization for numerical stability

    Examples
    --------
    >>> import numpy as np
    >>> from kernel_regression import LocalPolynomialRegression
    >>> X = np.random.randn(100, 2)
    >>> y = X[:, 0]**2 + 0.1 * np.random.randn(100)
    >>> model = LocalPolynomialRegression(order=2, bandwidth="cv")
    >>> model.fit(X, y)
    >>> predictions = model.predict(X[:5])
    """

    def __init__(
        self,
        kernel: str | Callable = "gaussian",
        bandwidth: float | NDArray[np.floating] | str = "cv",
        cv: int | str = "loo",
        order: int | str = 1,
        max_order: int = 2,
        regularization: float = 1e-10,
    ):
        super().__init__(kernel=kernel, bandwidth=bandwidth, cv=cv)
        self.order = order
        self.max_order = max_order
        self.regularization = regularization

    def fit(self, X: NDArray, y: NDArray) -> "LocalPolynomialRegression":
        """
        Fit the local polynomial regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self
            Fitted estimator
        """
        X, y = self._validate_data_fit(X, y)

        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        self.kernel_func_ = get_kernel(self.kernel)

        # Select polynomial order
        if isinstance(self.order, str) and self.order == "cv":
            self.order_ = self._select_order_cv(X, y)
        else:
            self.order_ = int(self.order)

        # Select bandwidth (after order is determined)
        self.bandwidth_ = self._compute_bandwidth(X, y, self.order_)

        return self

    def _select_order_cv(
        self, X: NDArray[np.floating], y: NDArray[np.floating]
    ) -> int:
        """Select polynomial order via cross-validation."""
        best_order = 0
        best_error = np.inf

        for order in range(self.max_order + 1):
            # Get bandwidth for this order
            selector = CrossValidatedBandwidth(
                kernel=self.kernel if isinstance(self.kernel, str) else "gaussian",
                cv=self.cv,
                polynomial_order=order,
            )
            bandwidth = selector(X, y)

            # Compute CV error
            error = self._compute_cv_error(X, y, bandwidth, order)

            if error < best_error:
                best_error = error
                best_order = order

        return best_order

    def _compute_cv_error(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        bandwidth: NDArray[np.floating],
        order: int,
    ) -> float:
        """Compute leave-one-out CV error for given order and bandwidth."""
        n_samples = X.shape[0]
        errors = np.zeros(n_samples)

        for i in range(n_samples):
            mask = np.ones(n_samples, dtype=bool)
            mask[i] = False

            y_pred = self._predict_point(
                X[i : i + 1], X[mask], y[mask], bandwidth, order
            )
            errors[i] = (y[i] - y_pred[0]) ** 2

        return float(np.mean(errors))

    def _build_design_matrix(
        self,
        diff: NDArray[np.floating],
        order: int,
    ) -> NDArray[np.floating]:
        """
        Build polynomial design matrix.

        For multivariate case, includes all polynomial terms up to given order.
        """
        n_samples, n_features = diff.shape

        # Start with intercept
        columns = [np.ones(n_samples)]

        if order >= 1:
            # Linear terms
            for j in range(n_features):
                columns.append(diff[:, j])

        if order >= 2:
            # Quadratic terms: x_j^2 and cross terms x_j * x_k
            for j in range(n_features):
                columns.append(diff[:, j] ** 2)
            for j in range(n_features):
                for k in range(j + 1, n_features):
                    columns.append(diff[:, j] * diff[:, k])

        if order >= 3:
            # Cubic terms
            for j in range(n_features):
                columns.append(diff[:, j] ** 3)

        return np.column_stack(columns)

    def _predict_point(
        self,
        x: NDArray[np.floating],
        X_train: NDArray[np.floating],
        y_train: NDArray[np.floating],
        bandwidth: NDArray[np.floating],
        order: int,
    ) -> NDArray[np.floating]:
        """Predict at point(s) using local polynomial.

        Uses scipy.linalg.lstsq for numerical stability instead of
        direct matrix inversion. Handles boundary regions where
        fewer neighbors are available.

        Args:
            x: Query points of shape (n_pred, n_features).
            X_train: Training features of shape (n_train, n_features).
            y_train: Training targets of shape (n_train,).
            bandwidth: Bandwidth per feature.
            order: Polynomial order.

        Returns:
            Predictions at query points.
        """
        x = np.atleast_2d(x)
        n_pred = x.shape[0]

        y_pred = np.zeros(n_pred)

        for i in range(n_pred):
            weights = multivariate_kernel_weights(
                x[i : i + 1], X_train, bandwidth, self.kernel_func_
            ).flatten()

            # Build design matrix
            diff = X_train - x[i]
            design = self._build_design_matrix(diff, order)

            # Weighted design matrix (more efficient than W @ design)
            sqrt_weights = np.sqrt(weights)
            W_design = design * sqrt_weights[:, np.newaxis]
            W_y = y_train * sqrt_weights

            # Add regularization
            n_cols = design.shape[1]
            reg_matrix = np.sqrt(self.regularization) * np.eye(n_cols)

            # Augment for regularized least squares
            A = np.vstack([W_design, reg_matrix])
            b = np.concatenate([W_y, np.zeros(n_cols)])

            # Use scipy.linalg.lstsq for numerical stability
            # This handles rank-deficient matrices better than solve
            # Only extract beta (index 0); other return values may be None/empty
            result = linalg.lstsq(A, b, lapack_driver='gelsd')
            beta = result[0]

            # Check if solution is valid
            if np.isfinite(beta[0]):
                y_pred[i] = beta[0]  # Intercept = prediction at x[i]
            else:
                # Fallback: weighted average (Nadaraya-Watson)
                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    y_pred[i] = np.sum(weights * y_train) / weight_sum
                else:
                    y_pred[i] = np.mean(y_train)

        return y_pred

    def predict(self, X: NDArray) -> NDArray[np.floating]:
        """
        Predict using local polynomial regression.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        X = self._validate_data_predict(X)
        return self._predict_point(X, self.X_, self.y_, self.bandwidth_, self.order_)

    def predict_with_derivatives(
        self, X: NDArray
    ) -> tuple[NDArray[np.floating], NDArray[np.floating] | None]:
        """
        Predict with gradient estimates.

        Only available when order >= 1.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        gradients : ndarray of shape (n_samples, n_features) or None
            Estimated gradients (None if order=0)
        """
        X = self._validate_data_predict(X)

        if self.order_ == 0:
            return self.predict(X), None

        n_pred = X.shape[0]
        n_features = self.n_features_in_

        y_pred = np.zeros(n_pred)
        gradients = np.zeros((n_pred, n_features))

        for i in range(n_pred):
            weights = multivariate_kernel_weights(
                X[i : i + 1], self.X_, self.bandwidth_, self.kernel_func_
            ).flatten()

            diff = self.X_ - X[i]
            design = self._build_design_matrix(diff, self.order_)

            W = np.diag(weights)
            XtW = design.T @ W
            XtWX = XtW @ design
            XtWy = XtW @ self.y_

            reg = self.regularization * np.eye(XtWX.shape[0])

            try:
                beta = np.linalg.solve(XtWX + reg, XtWy)
                y_pred[i] = beta[0]
                # Gradient is the linear coefficients (indices 1 to n_features)
                gradients[i] = beta[1 : n_features + 1]
            except np.linalg.LinAlgError:
                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    y_pred[i] = np.sum(weights * self.y_) / weight_sum
                else:
                    y_pred[i] = np.mean(self.y_)
                gradients[i] = 0.0

        return y_pred, gradients
