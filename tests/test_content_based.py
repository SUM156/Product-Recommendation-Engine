"""Unit tests for src/content_based.py."""

import pandas as pd
import pytest

from src.content_based import build_content_similarity_matrix, recommend_similar_products
from src.exceptions import ProductNotFoundError


@pytest.fixture
def sample_products():
    return pd.DataFrame(
        {
            "product_id": [1, 2, 3],
            "name": ["Wireless Mouse", "Wireless Keyboard", "Yoga Mat"],
            "category": ["Electronics", "Electronics", "Sports"],
            "description": [
                "Wireless ergonomic mouse for office use",
                "Wireless mechanical keyboard for gaming and office",
                "Non-slip yoga mat for home workouts",
            ],
        }
    )


def test_build_content_similarity_matrix_shape(sample_products):
    similarity = build_content_similarity_matrix(sample_products)
    assert set(similarity.index) == {1, 2, 3}
    assert set(similarity.columns) == {1, 2, 3}


def test_same_category_products_more_similar_than_different_category(sample_products):
    """Product 1 and 2 are both Electronics with overlapping words
    ('wireless', 'office'); product 3 is a completely different
    category (Sports) with no word overlap. Similarity(1,2) must
    exceed similarity(1,3).
    """
    similarity = build_content_similarity_matrix(sample_products)
    assert similarity.loc[1, 2] > similarity.loc[1, 3]


def test_content_similarity_diagonal_is_one(sample_products):
    similarity = build_content_similarity_matrix(sample_products)
    for product_id in sample_products["product_id"]:
        assert similarity.loc[product_id, product_id] == pytest.approx(1.0)


def test_recommend_similar_products_excludes_self(sample_products):
    similarity = build_content_similarity_matrix(sample_products)
    results = recommend_similar_products(1, similarity, top_n=2)

    product_ids = [pid for pid, _ in results]
    assert 1 not in product_ids


def test_recommend_similar_products_ranks_same_category_first(sample_products):
    similarity = build_content_similarity_matrix(sample_products)
    results = recommend_similar_products(1, similarity, top_n=2)

    top_match_id = results[0][0]
    assert top_match_id == 2  # the other Electronics product


def test_recommend_similar_products_unknown_product_raises(sample_products):
    similarity = build_content_similarity_matrix(sample_products)
    with pytest.raises(ProductNotFoundError):
        recommend_similar_products(999, similarity)


def test_recommend_similar_products_respects_top_n(sample_products):
    similarity = build_content_similarity_matrix(sample_products)
    results = recommend_similar_products(1, similarity, top_n=1)
    assert len(results) == 1