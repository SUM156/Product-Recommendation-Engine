"""
content_based.py
=================
Content-based filtering using TF-IDF over product text (category +
description). This is the cold-start fallback: collaborative
filtering needs rating HISTORY to work at all -- a brand-new user with
zero ratings, or a brand-new product with zero purchases, has no
signal for cosine similarity on the user-item matrix to use. Content
similarity needs no interaction history whatsoever, just the product's
own text, which is why it's the standard industry answer to the
"cold start problem".
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.exceptions import ProductNotFoundError


def build_content_similarity_matrix(products: pd.DataFrame) -> pd.DataFrame:
    """Build a product-product similarity matrix from category + description text.

    Args:
        products: The product catalog DataFrame (must have 'category'
            and 'description' columns).

    Returns:
        A square DataFrame (product_id x product_id) of cosine
        similarity scores over TF-IDF vectors. Category is included
        alongside the free-text description (and effectively repeated
        as its own "term") so products in the same category get a
        similarity boost even when their descriptions share few words.
    """
    combined_text = products["category"] + " " + products["category"] + " " + products["description"]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(combined_text)

    similarity_scores = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(
        similarity_scores, index=products["product_id"], columns=products["product_id"]
    )


def recommend_similar_products(
    product_id: int, content_similarity: pd.DataFrame, top_n: int = 5
) -> list[tuple[int, float]]:
    """Recommend products most similar in content to a given product.

    Args:
        product_id: The product to find similar items for.
        content_similarity: Output of `build_content_similarity_matrix`.
        top_n: How many similar products to return.

    Returns:
        A list of (product_id, similarity_score) tuples, most similar
        first, EXCLUDING the product itself.

    Raises:
        ProductNotFoundError: If `product_id` isn't in the similarity matrix.
    """
    if product_id not in content_similarity.index:
        raise ProductNotFoundError(f"No content data for product_id {product_id}.")

    scores = content_similarity.loc[product_id].drop(product_id)
    top_matches = scores.sort_values(ascending=False).head(top_n)
    return list(top_matches.items())