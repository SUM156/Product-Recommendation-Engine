"""
recommender.py
==============
Orchestrates the three recommendation strategies into one coherent
system:

1. USER-BASED COLLABORATIVE FILTERING -- "users like you also liked
   these" -- for users with enough rating history for similarity to
   be meaningful.
2. ITEM-BASED SIMILARITY -- "customers who bought X also bought Y" --
   for a specific product page, independent of any one user.
3. POPULARITY-BASED COLD START -- for brand-new users with zero (or
   too few) ratings, where collaborative filtering has no signal to
   work with at all.

`recommend_for_user` is the single entry point that decides which
strategy applies and falls back gracefully -- this is the "hybrid"
part: never return an empty result just because a user is new.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.exceptions import ProductNotFoundError, UserNotFoundError

DEFAULT_TOP_N = 5
DEFAULT_K_NEIGHBORS = 5
MIN_RATINGS_FOR_COLLABORATIVE_FILTERING = 1

# Bayesian-average smoothing constant for popularity ranking -- see
# `popularity_based_recommendations` docstring for why this matters.
POPULARITY_MIN_RATINGS_PRIOR = 3


@dataclass(frozen=True)
class Recommendation:
    """A single recommended product with its predicted/relevance score."""

    product_id: int
    score: float


@dataclass(frozen=True)
class RecommendationResult:
    """The full result of a recommendation request.

    Attributes:
        recommendations: Ranked list of recommended products.
        method_used: Which strategy actually produced these results --
            'collaborative_filtering', 'item_similarity', or
            'popularity_cold_start'. Surfacing this explicitly (rather
            than hiding it) matters for a real product: a "customers
            also bought" widget and a "because you have no history
            yet, here's what's popular" widget should look and read
            differently to the end user.
    """

    recommendations: list[Recommendation] = field(default_factory=list)
    method_used: str = ""


def user_based_recommendations(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    user_similarity: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
) -> list[Recommendation]:
    """Recommend products via user-based collaborative filtering.

    For each product the target user HASN'T rated, predicts a score as
    the similarity-weighted average of the k most similar users'
    ratings on that product.

    Args:
        user_id: The user to generate recommendations for.
        user_item_matrix: Output of `similarity.build_user_item_matrix`.
        user_similarity: Output of `similarity.compute_user_similarity`.
        top_n: How many recommendations to return.
        k_neighbors: How many most-similar users to draw predictions from.

    Returns:
        Ranked recommendations. Returns an EMPTY list (not an error)
        if the user has no similar users with overlapping rated
        products -- this is a normal outcome for a sparse dataset, and
        the caller (`recommend_for_user`) treats an empty result as a
        signal to fall back to popularity-based recommendations.

    Raises:
        UserNotFoundError: If `user_id` isn't in the rating matrix at all.
    """
    if user_id not in user_item_matrix.index:
        raise UserNotFoundError(f"No rating data for user_id {user_id}.")

    similar_users = user_similarity.loc[user_id].drop(user_id).sort_values(ascending=False)
    top_neighbors = similar_users.head(k_neighbors)
    top_neighbors = top_neighbors[top_neighbors > 0]  # zero similarity = no useful signal

    if top_neighbors.empty:
        return []

    user_ratings = user_item_matrix.loc[user_id]
    unrated_products = user_ratings[user_ratings == 0].index

    predictions = []
    for product_id in unrated_products:
        neighbor_ratings = user_item_matrix.loc[top_neighbors.index, product_id]
        rated_mask = neighbor_ratings > 0
        if not rated_mask.any():
            continue  # none of our neighbors rated this product either

        relevant_weights = top_neighbors[rated_mask]
        relevant_ratings = neighbor_ratings[rated_mask]
        predicted_score = (relevant_weights * relevant_ratings).sum() / relevant_weights.sum()
        predictions.append(Recommendation(product_id=int(product_id), score=round(predicted_score, 3)))

    predictions.sort(key=lambda r: r.score, reverse=True)
    return predictions[:top_n]


def item_based_recommendations(
    product_id: int, item_similarity: pd.DataFrame, top_n: int = DEFAULT_TOP_N
) -> list[Recommendation]:
    """Recommend products similar to a given product -- "customers who
    bought X also bought Y", independent of any specific user.

    Args:
        product_id: The reference product (e.g. the one being viewed).
        item_similarity: Output of `similarity.compute_item_similarity`.
        top_n: How many similar products to return.

    Returns:
        Ranked similar products, excluding the product itself.

    Raises:
        ProductNotFoundError: If `product_id` has no rating data.
    """
    if product_id not in item_similarity.index:
        raise ProductNotFoundError(f"No rating data for product_id {product_id}.")

    scores = item_similarity.loc[product_id].drop(product_id)
    top_matches = scores[scores > 0].sort_values(ascending=False).head(top_n)
    return [Recommendation(product_id=int(pid), score=round(score, 3)) for pid, score in top_matches.items()]


def popularity_based_recommendations(
    ratings: pd.DataFrame, top_n: int = DEFAULT_TOP_N
) -> list[Recommendation]:
    """Rank products by a Bayesian-smoothed average rating -- the
    cold-start fallback for users with no (or too little) history.

    Why not just sort by raw average rating? A product with a single
    5-star rating would outrank a product with 200 ratings averaging
    4.8 -- clearly the wrong answer. The Bayesian average pulls
    low-count products' scores toward the GLOBAL mean rating,
    proportional to how few ratings they have, which is the standard
    fix (the same approach IMDB uses for its "Top 250" ranking).

    Args:
        ratings: The full ratings DataFrame.
        top_n: How many popular products to return.

    Returns:
        Ranked products by smoothed popularity score.
    """
    stats = ratings.groupby("product_id")["rating"].agg(["mean", "count"])
    global_mean = ratings["rating"].mean()
    prior_count = POPULARITY_MIN_RATINGS_PRIOR

    stats["bayesian_score"] = (
        stats["count"] * stats["mean"] + prior_count * global_mean
    ) / (stats["count"] + prior_count)

    top_products = stats.sort_values("bayesian_score", ascending=False).head(top_n)
    return [
        Recommendation(product_id=int(pid), score=round(row["bayesian_score"], 3))
        for pid, row in top_products.iterrows()
    ]


def recommend_for_user(
    user_id: int,
    ratings: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    user_similarity: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
) -> RecommendationResult:
    """The main hybrid entry point: recommend products for a user,
    automatically falling back through strategies so a result is
    (almost) always returned.

    Decision logic:
    1. If the user has NO rating history at all -> popularity fallback
       immediately (there's no collaborative signal whatsoever).
    2. If the user has history but user-based CF finds no similar
       users with overlapping products -> popularity fallback.
    3. Otherwise -> user-based collaborative filtering.

    Args:
        user_id: The user to recommend for.
        ratings: The full ratings DataFrame (used for the popularity fallback).
        user_item_matrix: Output of `similarity.build_user_item_matrix`.
        user_similarity: Output of `similarity.compute_user_similarity`.
        top_n: How many recommendations to return.
        k_neighbors: How many similar users to consider for CF.

    Returns:
        A `RecommendationResult` with `method_used` telling the caller
        which strategy actually produced the result.
    """
    is_known_user = user_id in user_item_matrix.index
    has_rating_history = is_known_user and (user_item_matrix.loc[user_id] > 0).any()

    if not has_rating_history:
        popular = popularity_based_recommendations(ratings, top_n=top_n)
        return RecommendationResult(recommendations=popular, method_used="popularity_cold_start")

    cf_recommendations = user_based_recommendations(
        user_id, user_item_matrix, user_similarity, top_n=top_n, k_neighbors=k_neighbors
    )
    if cf_recommendations:
        return RecommendationResult(
            recommendations=cf_recommendations, method_used="collaborative_filtering"
        )

    # User has ratings, but no similar users overlap with their taste
    # (can happen in a very sparse or very small dataset) -- fall back
    # rather than returning nothing.
    popular = popularity_based_recommendations(ratings, top_n=top_n)
    return RecommendationResult(recommendations=popular, method_used="popularity_cold_start")