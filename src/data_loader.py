"""
data_loader.py
===============
Loads and validates the two datasets this engine needs: user-product
ratings, and the product catalog (used for content-based filtering
and for turning product IDs into human-readable names in results).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from src.exceptions import InvalidDataError

RATINGS_REQUIRED_COLUMNS = {"user_id", "product_id", "rating"}
PRODUCTS_REQUIRED_COLUMNS = {"product_id", "name", "category", "description"}

MIN_RATING = 1
MAX_RATING = 5


def load_ratings(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate the user-product ratings dataset.

    Args:
        file_path: Path to a CSV with columns: user_id, product_id, rating.

    Returns:
        A validated DataFrame.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        InvalidDataError: If required columns are missing, the file
            is empty, or any rating falls outside the valid 1-5 range.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Ratings file not found: {file_path}")

    ratings = pd.read_csv(path)

    missing = RATINGS_REQUIRED_COLUMNS - set(ratings.columns)
    if missing:
        raise InvalidDataError(f"Ratings data is missing required column(s): {sorted(missing)}.")

    if ratings.empty:
        raise InvalidDataError("Ratings file contains no rows.")

    out_of_range = ratings[(ratings["rating"] < MIN_RATING) | (ratings["rating"] > MAX_RATING)]
    if not out_of_range.empty:
        raise InvalidDataError(
            f"Found {len(out_of_range)} rating(s) outside the valid "
            f"{MIN_RATING}-{MAX_RATING} range."
        )

    return ratings


def load_products(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate the product catalog.

    Args:
        file_path: Path to a CSV with columns: product_id, name,
            category, description (price is optional).

    Returns:
        A validated DataFrame.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        InvalidDataError: If required columns are missing or the file
            is empty.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Products file not found: {file_path}")

    products = pd.read_csv(path)

    missing = PRODUCTS_REQUIRED_COLUMNS - set(products.columns)
    if missing:
        raise InvalidDataError(f"Products data is missing required column(s): {sorted(missing)}.")

    if products.empty:
        raise InvalidDataError("Products file contains no rows.")

    return products