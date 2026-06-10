"""Exception types for the Japanese stock analysis engine."""

from __future__ import annotations


class JPStockAnalysisError(Exception):
    """Base exception for all engine errors."""


class DataValidationError(JPStockAnalysisError):
    """Raised when input data fails structural validation (e.g. malformed CSV)."""


class ProviderError(JPStockAnalysisError):
    """Raised when a data provider cannot fulfil a request."""


class InsufficientDataError(JPStockAnalysisError):
    """Raised when an operation cannot proceed at all due to missing data.

    Routine missing financial data must NOT raise this error; it should be
    reported through ``warnings`` and reduced ``confidence_score`` on result
    objects instead.
    """
