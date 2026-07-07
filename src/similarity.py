"""
similarity.py
=============
Builds the user-item rating matrix and computes cosine similarity
between users (for user-based CF) and between items (for item-based
CF). This is the mathematical core of collaborative filtering --
everything else in this package consumes the outputs of this module.

Why cosine similarity specifically? It measures the angle between two
rating vectors, not their magnitude -- so a user who rates everything
5/5 and a user who rates everything 4/5 (but in the exact same
relative pattern) are considered similar, rather than "different"
just because one rates more generously overall. This is the standard
choice for collaborative filtering on rating data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format ratings into a wide user x item matrix.

    Args:
        ratings: A DataFrame with columns [user_id, product_id, rating].

    Returns:
        A DataFrame indexed by user_id, columned by product_id, with
        rating values. Missing (user, product) pairs -- the vast
        majority of the matrix, since no user rates every product --
        are filled with 0. Using 0 (rather than NaN) is deliberate:
        cosine similarity treats a 0 as "no signal" for that
        dimension, which is exactly the right interpretation for "this
        user never rated this product" (as opposed to imputing a
        NaN-derived guess).
    """
    return ratings.pivot_table(
        index="user_id", columns="product_id", values="rating", fill_value=0
    )


def compute_user_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise cosine similarity between all users.

    Args:
        user_item_matrix: Output of `build_user_item_matrix`.

    Returns:
        A square DataFrame (user_id x user_id) of similarity scores,
        0 (completely dissimilar) to 1 (identical rating pattern).
    """
    similarity_scores = cosine_similarity(user_item_matrix.values)
    return pd.DataFrame(
        similarity_scores, index=user_item_matrix.index, columns=user_item_matrix.index
    )


def compute_item_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise cosine similarity between all items.

    Args:
        user_item_matrix: Output of `build_user_item_matrix`.

    Returns:
        A square DataFrame (product_id x product_id) of similarity
        scores. Computed by transposing the user-item matrix (so rows
        become items, columns become users) before running cosine
        similarity -- the same math as `compute_user_similarity`, just
        applied to the other axis.
    """
    item_matrix = user_item_matrix.T
    similarity_scores = cosine_similarity(item_matrix.values)
    return pd.DataFrame(similarity_scores, index=item_matrix.index, columns=item_matrix.index)