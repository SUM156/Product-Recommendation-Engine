"""
api.py
======
Flask REST API exposing the recommendation engine over HTTP -- the
"API endpoint with Flask" deliverable. Uses an application factory
(`create_app`) rather than a bare module-level `app = Flask(__name__)`
so tests can spin up multiple independent app instances (e.g. pointed
at different test datasets) without any global state leaking between
tests.
"""

from __future__ import annotations

import logging

from flask import Flask, jsonify, request

from src.content_based import build_content_similarity_matrix, recommend_similar_products
from src.data_loader import load_products, load_ratings
from src.exceptions import ProductNotFoundError, RecommenderError, UserNotFoundError
from src.recommender import (
    Recommendation,
    item_based_recommendations,
    popularity_based_recommendations,
    recommend_for_user,
)
from src.similarity import build_user_item_matrix, compute_item_similarity, compute_user_similarity

logger = logging.getLogger(__name__)


def _recommendation_to_dict(recommendation: Recommendation, products: dict) -> dict:
    """Attach a human-readable product name to a raw recommendation,
    since an API consumer wants "Wireless Earbuds Pro", not just
    product_id=1 -- a product_id-only response would force every
    client to make a second lookup call just to render a result.
    """
    product_info = products.get(recommendation.product_id, {})
    return {
        "product_id": recommendation.product_id,
        "name": product_info.get("name", "Unknown Product"),
        "category": product_info.get("category", ""),
        "score": recommendation.score,
    }


def create_app(ratings_path: str = "data/ratings.csv", products_path: str = "data/products.csv") -> Flask:
    """Application factory: loads data, precomputes similarity
    matrices, and wires up routes.

    Precomputing the user-item matrix and both similarity matrices at
    startup (rather than on every request) is deliberate -- cosine
    similarity over the full matrix is the expensive part, and it
    doesn't change between requests unless the underlying rating data
    changes, so paying that cost once per app startup is the right
    trade-off for a read-heavy recommendation API.

    Args:
        ratings_path: Path to the ratings CSV.
        products_path: Path to the products CSV.

    Returns:
        A configured Flask app, ready to run or test.
    """
    app = Flask(__name__)

    ratings = load_ratings(ratings_path)
    products = load_products(products_path)

    user_item_matrix = build_user_item_matrix(ratings)
    user_similarity = compute_user_similarity(user_item_matrix)
    item_similarity = compute_item_similarity(user_item_matrix)
    content_similarity = build_content_similarity_matrix(products)

    products_by_id = products.set_index("product_id").to_dict(orient="index")
    for product_id, info in products_by_id.items():
        info["name"] = info.get("name")  # ensure key exists for the helper above

    app.config["ratings"] = ratings
    app.config["products_by_id"] = products_by_id
    app.config["user_item_matrix"] = user_item_matrix
    app.config["user_similarity"] = user_similarity
    app.config["item_similarity"] = item_similarity
    app.config["content_similarity"] = content_similarity

    @app.errorhandler(RecommenderError)
    def handle_recommender_error(error: RecommenderError):
        """Convert any domain error into a clean 404 JSON response,
        rather than letting Flask's default 500 HTML error page leak
        internal exception details to an API consumer.
        """
        return jsonify({"error": str(error)}), 404

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Simple liveness check for load balancers/monitoring."""
        return jsonify({"status": "ok", "products_loaded": len(products_by_id)})

    @app.route("/api/recommend/user/<int:user_id>", methods=["GET"])
    def recommend_for_user_endpoint(user_id: int):
        """Hybrid recommendations for a user: collaborative filtering
        if they have rating history, popularity-based cold start if not.
        """
        top_n = request.args.get("top_n", default=5, type=int)

        if user_id not in app.config["user_item_matrix"].index:
            raise UserNotFoundError(f"No rating data for user_id {user_id}.")

        result = recommend_for_user(
            user_id,
            app.config["ratings"],
            app.config["user_item_matrix"],
            app.config["user_similarity"],
            top_n=top_n,
        )
        return jsonify(
            {
                "user_id": user_id,
                "method": result.method_used,
                "recommendations": [
                    _recommendation_to_dict(r, app.config["products_by_id"])
                    for r in result.recommendations
                ],
            }
        )

    @app.route("/api/recommend/product/<int:product_id>", methods=["GET"])
    def recommend_similar_endpoint(product_id: int):
        """'Customers who bought this also bought' -- item-based
        similarity, falling back to content-based similarity if the
        product has no rating history yet (a brand-new listing).
        """
        top_n = request.args.get("top_n", default=5, type=int)

        if product_id not in app.config["products_by_id"]:
            raise ProductNotFoundError(f"No product found with id {product_id}.")

        try:
            recommendations = item_based_recommendations(
                product_id, app.config["item_similarity"], top_n=top_n
            )
            method = "item_similarity"
        except ProductNotFoundError:
            recommendations = []
            method = "item_similarity"

        if not recommendations:
            recommendations = recommend_similar_products(
                product_id, app.config["content_similarity"], top_n=top_n
            )
            recommendations = [Recommendation(product_id=pid, score=score) for pid, score in recommendations]
            method = "content_based_cold_start"

        return jsonify(
            {
                "product_id": product_id,
                "method": method,
                "recommendations": [
                    _recommendation_to_dict(r, app.config["products_by_id"]) for r in recommendations
                ],
            }
        )

    @app.route("/api/popular", methods=["GET"])
    def popular_endpoint():
        """Globally popular products, ranked by Bayesian-smoothed rating."""
        top_n = request.args.get("top_n", default=5, type=int)
        recommendations = popularity_based_recommendations(app.config["ratings"], top_n=top_n)
        return jsonify(
            {
                "method": "popularity",
                "recommendations": [
                    _recommendation_to_dict(r, app.config["products_by_id"]) for r in recommendations
                ],
            }
        )

    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(debug=False, host="0.0.0.0", port=5000)
    