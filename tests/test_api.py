"""Integration tests for src/api.py -- exercises the full Flask app
with a small, isolated test dataset (not the bundled production data).
"""

import pytest

from src.api import create_app


@pytest.fixture
def client(tmp_path):
    ratings_content = (
        "user_id,product_id,rating\n"
        "1,10,5\n1,20,4\n"
        "2,10,5\n2,20,4\n2,30,3\n"
        "3,10,1\n3,40,5\n"
    )
    products_content = (
        "product_id,name,category,description\n"
        "10,Wireless Mouse,Electronics,Ergonomic wireless mouse\n"
        "20,Wireless Keyboard,Electronics,Mechanical wireless keyboard\n"
        "30,Yoga Mat,Sports,Non-slip yoga mat\n"
        "40,Running Shoes,Sports,Lightweight running shoes\n"
    )
    ratings_path = tmp_path / "ratings.csv"
    products_path = tmp_path / "products.csv"
    ratings_path.write_text(ratings_content)
    products_path.write_text(products_content)

    app = create_app(ratings_path=str(ratings_path), products_path=str(products_path))
    app.config["TESTING"] = True
    return app.test_client()


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["products_loaded"] == 4


def test_recommend_for_user_returns_recommendations(client):
    response = client.get("/api/recommend/user/1")
    assert response.status_code == 200
    data = response.get_json()

    assert data["user_id"] == 1
    assert data["method"] in ("collaborative_filtering", "popularity_cold_start")
    assert isinstance(data["recommendations"], list)


def test_recommend_for_user_includes_product_names(client):
    response = client.get("/api/recommend/user/1")
    data = response.get_json()

    if data["recommendations"]:
        assert "name" in data["recommendations"][0]
        assert data["recommendations"][0]["name"] != "Unknown Product"


def test_recommend_for_unknown_user_returns_404(client):
    response = client.get("/api/recommend/user/9999")
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_recommend_for_user_respects_top_n(client):
    response = client.get("/api/recommend/user/1?top_n=1")
    data = response.get_json()
    assert len(data["recommendations"]) <= 1


def test_recommend_similar_products(client):
    response = client.get("/api/recommend/product/10")
    assert response.status_code == 200
    data = response.get_json()

    assert data["product_id"] == 10
    assert data["method"] in ("item_similarity", "content_based_cold_start")


def test_recommend_similar_for_unknown_product_returns_404(client):
    response = client.get("/api/recommend/product/9999")
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_popular_endpoint_returns_recommendations(client):
    response = client.get("/api/popular")
    assert response.status_code == 200
    data = response.get_json()

    assert data["method"] == "popularity"
    assert len(data["recommendations"]) > 0


def test_popular_endpoint_respects_top_n(client):
    response = client.get("/api/popular?top_n=2")
    data = response.get_json()
    assert len(data["recommendations"]) <= 2