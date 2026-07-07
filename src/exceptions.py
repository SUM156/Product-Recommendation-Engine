"""
exceptions.py
=============
Custom exception hierarchy for the recommendation engine.
"""


class RecommenderError(Exception):
    """Base class for every error raised by this application."""


class InvalidDataError(RecommenderError):
    """Raised when ratings or product data fails schema validation."""


class UserNotFoundError(RecommenderError):
    """Raised when a requested user ID has no data in the current dataset."""


class ProductNotFoundError(RecommenderError):
    """Raised when a requested product ID has no data in the current catalog."""