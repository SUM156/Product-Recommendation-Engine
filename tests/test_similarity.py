"""Unit tests for src/similarity.py -- hand-verified cosine similarity math."""

import pandas as pd
import pytest

from src.similarity import build_user_item_matrix, compute_item_similarity, compute_user_similarity


@pytest.fixture
def sample_ratings():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "product_id": [10, 20, 10, 20, 10, 30],
            "rating": [5, 3, 5, 3, 1, 5],
        }
    )


def test_build_user_item_matrix_shape(sample_ratings):
    matrix = build_user_item_matrix(sample_ratings)
    assert set(matrix.index) == {1, 2, 3}
    assert set(matrix.columns) == {10, 20, 30}


def test_build_user_item_matrix_fills_missing_with_zero(sample_ratings):
    matrix = build_user_item_matrix(sample_ratings)
    # User 1 never rated product 30
    assert matrix.loc[1, 30] == 0


def test_build_user_item_matrix_preserves_actual_ratings(sample_ratings):
    matrix = build_user_item_matrix(sample_ratings)
    assert matrix.loc[1, 10] == 5
    assert matrix.loc[3, 30] == 5


def test_identical_users_have_similarity_one():
    """Users 1 and 2 rated product 10 and 20 IDENTICALLY (5, 3) --
    cosine similarity between identical vectors must be exactly 1.0.
    """
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "product_id": [10, 20, 10, 20],
            "rating": [5, 3, 5, 3],
        }
    )
    matrix = build_user_item_matrix(ratings)
    similarity = compute_user_similarity(matrix)

    assert similarity.loc[1, 2] == pytest.approx(1.0, abs=1e-9)


def test_orthogonal_users_have_similarity_zero():
    """User 1 only rated product 10; user 2 only rated product 20 --
    completely disjoint rating vectors, cosine similarity = 0.
    """
    ratings = pd.DataFrame(
        {
            "user_id": [1, 2],
            "product_id": [10, 20],
            "rating": [5, 5],
        }
    )
    matrix = build_user_item_matrix(ratings)
    similarity = compute_user_similarity(matrix)

    assert similarity.loc[1, 2] == pytest.approx(0.0, abs=1e-9)


def test_user_similarity_matrix_is_symmetric(sample_ratings):
    matrix = build_user_item_matrix(sample_ratings)
    similarity = compute_user_similarity(matrix)

    assert similarity.loc[1, 2] == pytest.approx(similarity.loc[2, 1])


def test_user_similarity_diagonal_is_one(sample_ratings):
    """A user compared to themselves must have similarity exactly 1.0."""
    matrix = build_user_item_matrix(sample_ratings)
    similarity = compute_user_similarity(matrix)

    for user_id in matrix.index:
        assert similarity.loc[user_id, user_id] == pytest.approx(1.0)


def test_item_similarity_matrix_shape(sample_ratings):
    matrix = build_user_item_matrix(sample_ratings)
    item_similarity = compute_item_similarity(matrix)

    assert set(item_similarity.index) == {10, 20, 30}
    assert set(item_similarity.columns) == {10, 20, 30}


def test_item_similarity_diagonal_is_one(sample_ratings):
    matrix = build_user_item_matrix(sample_ratings)
    item_similarity = compute_item_similarity(matrix)

    for product_id in matrix.columns:
        assert item_similarity.loc[product_id, product_id] == pytest.approx(1.0)