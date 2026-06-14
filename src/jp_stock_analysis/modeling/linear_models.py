"""Deterministic linear model baselines: Ridge and a real Elastic Net.

Both are pure-numpy, deterministic, and offline. They share a small fit/predict
API and the same preprocessing (training-set median imputation + optional
standardization). Degenerate synthetic cases return an explicit ``status`` and
``warnings`` instead of raising.

The Elastic Net is a **real coordinate-descent** implementation of

    f(beta) = 1/(2n) * ||y - X beta||^2
              + alpha * l1_ratio * ||beta||_1
              + 0.5 * alpha * (1 - l1_ratio) * ||beta||_2^2

with soft-thresholding for the L1 term and an unpenalized intercept. It is not a
placeholder, not a documented fallback, and never delegates to sklearn.

Research diagnostics only — no predictive or trading claim, no buy/sell signal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

RIDGE_VERSION = "ridge_v1"
ELASTIC_NET_VERSION = "elastic_net_coordinate_descent_v1"

STATUS_FITTED = "fitted"
STATUS_NOT_CONVERGED = "not_converged"
STATUS_CONSTANT_TARGET = "constant_target"
STATUS_INSUFFICIENT_ROWS = "insufficient_rows"
STATUS_NON_FINITE = "non_finite"
STATUS_UNFITTED = "unfitted"

_MIN_ROWS = 2


def _to_matrix(X: Sequence[Sequence[float | None]]) -> np.ndarray:
    """Float matrix with ``None`` -> ``nan`` (no fabrication; imputed later)."""
    rows = [[np.nan if v is None else float(v) for v in row] for row in X]
    return np.asarray(rows, dtype=float) if rows else np.empty((0, 0))


def _column_medians(matrix: np.ndarray) -> np.ndarray:
    """Per-column median over present values; all-missing column -> 0.0."""
    medians = np.zeros(matrix.shape[1])
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        present = col[~np.isnan(col)]
        medians[j] = float(np.median(present)) if present.size else 0.0
    return medians


def _impute(matrix: np.ndarray, medians: np.ndarray) -> np.ndarray:
    out = matrix.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        col[np.isnan(col)] = medians[j]
    return out


def _soft_threshold(z: float, gamma: float) -> float:
    """soft_threshold(z, gamma) = sign(z) * max(|z| - gamma, 0)."""
    if z > gamma:
        return z - gamma
    if z < -gamma:
        return z + gamma
    return 0.0


@dataclass
class _BaseLinearRanker:
    """Shared preprocessing, prediction, and metadata for linear rankers."""

    fit_intercept: bool = True
    standardize: bool = True

    feature_names: list[str] = field(default_factory=list)
    status: str = STATUS_UNFITTED
    warnings: list[str] = field(default_factory=list)
    model_version: str = "linear_v1"

    def __post_init__(self) -> None:
        self._imputation: np.ndarray | None = None
        self._feature_means: np.ndarray | None = None
        self._feature_scales: np.ndarray | None = None
        self._target_mean: float = 0.0
        self._beta_scaled: np.ndarray | None = None
        self._beta_unscaled: np.ndarray | None = None
        self._intercept: float = 0.0
        self._solver_meta: dict[str, Any] = {}
        self._n_samples: int = 0

    # ----- subclasses implement the solve on standardized, centered data ----- #
    def _solve(self, xs: np.ndarray, yc: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError

    def fit(
        self,
        X: Sequence[Sequence[float | None]],
        y: Sequence[float],
        feature_names: Sequence[str] | None = None,
        *,
        sample_weight: Sequence[float] | None = None,
    ) -> _BaseLinearRanker:
        matrix = _to_matrix(X)
        target = np.asarray([float(v) for v in y], dtype=float)
        n_samples, n_features = (matrix.shape if matrix.ndim == 2 else (0, 0))
        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"f{i}" for i in range(n_features)]
        )
        self.warnings = []
        self._n_samples = n_samples
        self._reset_coefficients(n_features)

        if n_samples < _MIN_ROWS or n_features == 0:
            self.status = STATUS_INSUFFICIENT_ROWS
            self.warnings.append(f"insufficient rows/features: {n_samples}x{n_features}")
            self._target_mean = float(target.mean()) if target.size else 0.0
            self._intercept = self._target_mean
            return self

        medians = _column_medians(matrix)
        self._imputation = medians
        imputed = _impute(matrix, medians)
        for j in range(n_features):
            if np.all(np.isnan(matrix[:, j])):
                self.warnings.append(f"feature {self.feature_names[j]!r} all-missing -> 0.0")

        if not np.all(np.isfinite(imputed)) or not np.all(np.isfinite(target)):
            self.status = STATUS_NON_FINITE
            self.warnings.append("non-finite values remain after imputation")
            self._target_mean = float(target.mean())
            self._intercept = self._target_mean
            return self

        weight = self._normalized_weights(sample_weight, n_samples)
        means = self._weighted_mean(imputed, weight) if self.fit_intercept else np.zeros(n_features)
        scales = self._scales(imputed, means)
        for j in range(n_features):
            if scales[j] == 0.0:
                self.warnings.append(f"feature {self.feature_names[j]!r} zero-variance")
        self._feature_means = means
        self._feature_scales = scales

        target_mean = float(np.average(target, weights=weight)) if self.fit_intercept else 0.0
        self._target_mean = target_mean
        if self.fit_intercept and float(np.std(target)) == 0.0:
            self.status = STATUS_CONSTANT_TARGET
            self.warnings.append("constant target: coefficients are zero")
            self._intercept = target_mean
            return self

        scale_eff = np.where(scales > 0, scales, 1.0)
        xs = (imputed - means) / scale_eff
        yc = target - target_mean
        # weighted least squares via sqrt(w) row scaling (penalty unaffected)
        root_w = np.sqrt(weight)[:, None]
        beta_scaled, solver_meta = self._solve(xs * root_w, (yc * np.sqrt(weight)))
        beta_scaled = np.where(scales > 0, beta_scaled, 0.0)

        self._beta_scaled = beta_scaled
        self._beta_unscaled = beta_scaled / scale_eff
        self._intercept = (
            target_mean - float(np.dot(self._beta_unscaled, means))
            if self.fit_intercept
            else 0.0
        )
        self._solver_meta = solver_meta
        self.status = solver_meta.get("status", STATUS_FITTED)
        if solver_meta.get("warning"):
            self.warnings.append(solver_meta["warning"])
        return self

    def _reset_coefficients(self, n_features: int) -> None:
        self._beta_scaled = np.zeros(n_features)
        self._beta_unscaled = np.zeros(n_features)
        self._intercept = 0.0
        self._feature_means = np.zeros(n_features)
        self._feature_scales = np.ones(n_features)
        self._imputation = np.zeros(n_features)

    @staticmethod
    def _normalized_weights(sample_weight: Sequence[float] | None, n: int) -> np.ndarray:
        if sample_weight is None:
            return np.ones(n)
        w = np.asarray([float(v) for v in sample_weight], dtype=float)
        if w.shape != (n,) or np.any(w < 0) or w.sum() <= 0:
            return np.ones(n)
        return w * (n / w.sum())  # mean weight 1, so unweighted is the default

    @staticmethod
    def _weighted_mean(matrix: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return np.average(matrix, axis=0, weights=weight)

    def _scales(self, matrix: np.ndarray, means: np.ndarray) -> np.ndarray:
        if not self.standardize:
            return np.ones(matrix.shape[1])
        centered = matrix - means
        return np.sqrt(np.mean(centered**2, axis=0))

    def predict(self, X: Sequence[Sequence[float | None]]) -> list[float]:
        if self._beta_unscaled is None or self.status == STATUS_UNFITTED:
            raise ValueError("model is not fitted")
        matrix = _to_matrix(X)
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"predict expects {len(self.feature_names)} features, got {matrix.shape[1]}"
            )
        imputed = _impute(matrix, self._imputation)
        return list(imputed @ self._beta_unscaled + self._intercept)

    def fit_predict(
        self,
        X: Sequence[Sequence[float | None]],
        y: Sequence[float],
        feature_names: Sequence[str] | None = None,
        *,
        sample_weight: Sequence[float] | None = None,
    ) -> list[float]:
        self.fit(X, y, feature_names, sample_weight=sample_weight)
        return self.predict(X)

    @property
    def coefficients(self) -> dict[str, float]:
        if self._beta_unscaled is None:
            return {}
        return {
            name: float(coef)
            for name, coef in zip(self.feature_names, self._beta_unscaled, strict=False)
        }

    @property
    def intercept(self) -> float:
        return float(self._intercept)

    @property
    def sparsity(self) -> float:
        if self._beta_scaled is None or self._beta_scaled.size == 0:
            return 0.0
        return float(np.mean(self._beta_scaled == 0.0))

    @property
    def selected_features(self) -> list[str]:
        if self._beta_scaled is None:
            return []
        return [
            name
            for name, coef in zip(self.feature_names, self._beta_scaled, strict=False)
            if coef != 0.0
        ]

    @property
    def model_metadata(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "status": self.status,
            "warnings": list(self.warnings),
            "fit_intercept": self.fit_intercept,
            "standardize": self.standardize,
            "n_samples": self._n_samples,
            "n_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "intercept": self.intercept,
            "coefficients": self.coefficients,
            "scaled_coefficients": (
                dict(zip(self.feature_names, map(float, self._beta_scaled), strict=False))
                if self._beta_scaled is not None
                else {}
            ),
            "feature_means": (
                list(map(float, self._feature_means)) if self._feature_means is not None else []
            ),
            "feature_scales": (
                list(map(float, self._feature_scales)) if self._feature_scales is not None else []
            ),
            "target_mean": self._target_mean,
            "imputation_values": (
                dict(zip(self.feature_names, map(float, self._imputation), strict=False))
                if self._imputation is not None
                else {}
            ),
            "sparsity": self.sparsity,
            "selected_features": self.selected_features,
            **self._solver_meta,
        }


@dataclass
class RidgeRanker(_BaseLinearRanker):
    """Deterministic closed-form ridge regression (research baseline)."""

    alpha: float = 1.0
    model_version: str = RIDGE_VERSION

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")

    def _solve(self, xs: np.ndarray, yc: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        n_features = xs.shape[1]
        gram = xs.T @ xs
        rhs = xs.T @ yc
        regularized = gram + self.alpha * np.eye(n_features)
        meta: dict[str, Any] = {"alpha": self.alpha, "status": STATUS_FITTED}
        try:
            beta = np.linalg.solve(regularized, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(regularized) @ rhs
            meta["warning"] = "singular matrix: used pseudo-inverse"
        return beta, meta


@dataclass
class ElasticNetRanker(_BaseLinearRanker):
    """Real coordinate-descent Elastic Net with L1 soft-thresholding."""

    alpha: float = 0.1
    l1_ratio: float = 0.5
    max_iter: int = 1000
    tol: float = 1e-6
    model_version: str = ELASTIC_NET_VERSION

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not 0.0 <= self.l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be in [0, 1]")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self.tol <= 0:
            raise ValueError("tol must be positive")

    def _objective(self, residual: np.ndarray, beta: np.ndarray, n: int) -> float:
        loss = 0.5 / n * float(residual @ residual)
        l1 = self.alpha * self.l1_ratio * float(np.sum(np.abs(beta)))
        l2 = 0.5 * self.alpha * (1.0 - self.l1_ratio) * float(beta @ beta)
        return loss + l1 + l2

    def _solve(self, xs: np.ndarray, yc: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        n, p = xs.shape
        beta = np.zeros(p)
        residual = yc.copy()  # residual = yc - xs @ beta, beta starts at 0
        norms = np.mean(xs**2, axis=0)  # (1/n) sum x_j^2
        gamma = self.alpha * self.l1_ratio
        l2_denom = self.alpha * (1.0 - self.l1_ratio)

        objective_history = [self._objective(residual, beta, n)]
        converged = False
        max_change = 0.0
        n_iter = 0
        while n_iter < self.max_iter:
            n_iter += 1
            max_change = 0.0
            for j in range(p):
                denom = norms[j] + l2_denom
                if denom == 0.0:  # zero-variance & no L2: coefficient stays 0
                    continue
                old = beta[j]
                # rho_j = (1/n) x_j . (residual + beta_j x_j)
                rho = float(np.mean(xs[:, j] * residual)) + old * norms[j]
                new = _soft_threshold(rho, gamma) / denom
                if new != old:
                    residual += (old - new) * xs[:, j]
                    beta[j] = new
                    max_change = max(max_change, abs(new - old))
            objective_history.append(self._objective(residual, beta, n))
            if max_change < self.tol:
                converged = True
                break

        status = STATUS_FITTED if converged else STATUS_NOT_CONVERGED
        meta: dict[str, Any] = {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "n_iter": n_iter,
            "converged": converged,
            "max_coefficient_change": max_change,
            "final_objective": objective_history[-1],
            "objective_history": objective_history,
            "status": status,
        }
        if not converged:
            meta["warning"] = (
                f"coordinate descent did not converge in {self.max_iter} iters "
                f"(max change {max_change:.3e} >= tol {self.tol:.3e})"
            )
        return beta, meta


__all__ = [
    "ELASTIC_NET_VERSION",
    "RIDGE_VERSION",
    "STATUS_CONSTANT_TARGET",
    "STATUS_FITTED",
    "STATUS_INSUFFICIENT_ROWS",
    "STATUS_NON_FINITE",
    "STATUS_NOT_CONVERGED",
    "STATUS_UNFITTED",
    "ElasticNetRanker",
    "RidgeRanker",
]
