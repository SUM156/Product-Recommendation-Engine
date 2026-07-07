"""Unit tests for src/recommender.py -- user-based CF, item-based CF,
popularity fallback, and the hybrid orchestration logic. Predictions
are hand-verified against manually computed expected values, not just
checked for "doesn't crash".
"""

import pandas as pd
import pytest

from src.exceptions import ProductNotFoundError, UserNotFoundError
from src.recommender import (
    item_based_recommendations,
    popularity_based_recommendations,
    recommend_for_user,
    user_based_recommendations,
)
from src.similarity import build_user_item_matrix, compute_item_similarity, compute_user_similarity


# ---------------------------------------------------------------------
# user_based_recommendations
# ---------------------------------------------------------------------


def test_user_based_single_neighbor_predicts_exact_neighbor_rating():
    """With exactly one (positively-similar) neighbor, the weighted
    average collapses to exactly that neighbor's rating -- a clean,
    hand-verifiable case regardless of the exact similarity weight.
    """
    ratings = pd.DataFrame(
        {
            "user_id": [1, 2, 2],
            "product_id": [10, 10, 20],
            "rating": [4, 4, 5],
        }
    )
    matrix = build_user_item_matrix(ratings)
    similarity = compute_user_similarity(matrix)

    recommendations = user_based_recommendations(1, matrix, similarity, top_n=5, k_neighbors=5)

    assert len(recommendations) == 1
    assert recommendations[0].product_id == 20
    assert recommendations[0].score == pytest.approx(5.0)


def test_user_based_recommendations_returns_empty_for_isolated_user():
    """User 1 and user 2 have ZERO overlapping rated products --
    cosine similarity is 0, so there's no CF signal at all. Must
    return an empty list, not an error or a nonsensical prediction.
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

    recommendations = user_based_recommendations(1, matrix, similarity)
    assert recommendations == []


def test_user_based_recommendations_unknown_user_raises():
    ratings = pd.DataFrame({"user_id": [1], "product_id": [10], "rating": [5]})
    matrix = build_user_item_matrix(ratings)
    similarity = compute_user_similarity(matrix)

    with pytest.raises(UserNotFoundError):
        user_based_recommendations(999, matrix, similarity)


def test_user_based_recommendations_excludes_already_rated_products():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "product_id": [10, 20, 10, 20],
            "rating": [5, 3, 5, 3],
        }
    )
    matrix = build_user_item_matrix(ratings)
    similarity = compute_user_similarity(matrix)

    recommendations = user_based_recommendations(1, matrix, similarity)
    recommended_ids = [r.product_id for r in recommendations]
    assert 10 not in recommended_ids
    assert 20 not in recommended_ids


def test_user_based_recommendations_respects_top_n():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 2, 2, 2, 2],
            "product_id": [5, 5, 10, 20, 30],
            "rating": [5, 5, 4, 3, 2],
        }
    )
    matrix = build_user_item_matrix(ratings)
    similarity = compute_user_similarity(matrix)

    recommendations = user_based_recommendations(1, matrix, similarity, top_n=2)
    assert len(recommendations) == 2


# ---------------------------------------------------------------------
# item_based_recommendations
# ---------------------------------------------------------------------


def test_item_based_recommendations_excludes_self():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "product_id": [10, 20, 10, 20],
            "rating": [5, 4, 5, 4],
        }
    )
    matrix = build_user_item_matrix(ratings)
    item_similarity = compute_item_similarity(matrix)

    recommendations = item_based_recommendations(10, item_similarity)
    recommended_ids = [r.product_id for r in recommendations]
    assert 10 not in recommended_ids


def test_item_based_recommendations_unknown_product_raises():
    ratings = pd.DataFrame({"user_id": [1], "product_id": [10], "rating": [5]})
    matrix = build_user_item_matrix(ratings)
    item_similarity = compute_item_similarity(matrix)

    with pytest.raises(ProductNotFoundError):
        item_based_recommendations(999, item_similarity)


# ---------------------------------------------------------------------
# popularity_based_recommendations
# ---------------------------------------------------------------------


def test_popularity_ranks_more_ratings_above_fewer_at_same_mean():
    """Product 1 (3 ratings of 5) must rank ABOVE product 2 (1 rating
    of 5) despite an identical raw average -- this is exactly the
    Bayesian-smoothing behavior that prevents a single 5-star review
    from outranking a product with a proven track record.
    """
    ratings = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "product_id": [1, 1, 1, 2, 3, 3, 3, 3, 3],
            "rating": [5, 5, 5, 5, 1, 1, 1, 1, 1],
        }
    )
    recommendations = popularity_based_recommendations(ratings, top_n=3)
    ranked_ids = [r.product_id for r in recommendations]

    assert ranked_ids.index(1) < ranked_ids.index(2) < ranked_ids.index(3)


def test_popularity_recommendations_respects_top_n():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "product_id": [1, 2, 3],
            "rating": [5, 4, 3],
        }
    )
    recommendations = popularity_based_recommendations(ratings, top_n=2)
    assert len(recommendations) == 2


# ---------------------------------------------------------------------
# recommend_for_user (hybrid orchestration)
# ---------------------------------------------------------------------


@pytest.fixture
def hybrid_test_data():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2],
            "product_id": [10, 20, 10, 20, 30],
            "rating": [5, 4, 5, 4, 3],
        }
    )
    matrix = build_user_item_matrix(ratings)
    similarity = compute_user_similarity(matrix)
    return ratings, matrix, similarity


def test_recommend_for_user_uses_collaborative_filtering_when_available(hybrid_test_data):
    ratings, matrix, similarity = hybrid_test_data
    result = recommend_for_user(1, ratings, matrix, similarity)

    assert result.method_used == "collaborative_filtering"
    assert len(result.recommendations) > 0


def test_recommend_for_user_falls_back_for_brand_new_user_not_in_matrix(hybrid_test_data):
    """A user who has NEVER rated anything isn't even a row in the
    matrix -- must fall back to popularity, not raise a KeyError.
    """
    ratings, matrix, similarity = hybrid_test_data
    result = recommend_for_user(999, ratings, matrix, similarity)

    assert result.method_used == "popularity_cold_start"
    assert len(result.recommendations) > 0


def test_recommend_for_user_falls_back_when_no_similar_users_exist():
    """User has rating history, but shares ZERO products with anyone
    else -- CF has no signal, so this must fall back to popularity
    rather than returning an empty result.
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

    result = recommend_for_user(1, ratings, matrix, similarity)
    assert result.method_used == "popularity_cold_start"