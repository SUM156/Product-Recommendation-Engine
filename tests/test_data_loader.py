"""Unit tests for src/data_loader.py."""

import pytest

from src.data_loader import load_products, load_ratings
from src.exceptions import InvalidDataError


def _write_csv(tmp_path, content, filename):
    file_path = tmp_path / filename
    file_path.write_text(content)
    return str(file_path)


# ---------------------------------------------------------------------
# load_ratings
# ---------------------------------------------------------------------


def test_load_ratings_valid_file(tmp_path):
    content = "user_id,product_id,rating\n1,1,5\n1,2,3\n"
    path = _write_csv(tmp_path, content, "ratings.csv")

    ratings = load_ratings(path)
    assert len(ratings) == 2


def test_load_ratings_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ratings(str(tmp_path / "missing.csv"))


def test_load_ratings_missing_column_raises(tmp_path):
    content = "user_id,product_id\n1,1\n"
    path = _write_csv(tmp_path, content, "ratings.csv")

    with pytest.raises(InvalidDataError):
        load_ratings(path)


def test_load_ratings_empty_file_raises(tmp_path):
    content = "user_id,product_id,rating\n"
    path = _write_csv(tmp_path, content, "ratings.csv")

    with pytest.raises(InvalidDataError):
        load_ratings(path)


def test_load_ratings_out_of_range_rating_raises(tmp_path):
    content = "user_id,product_id,rating\n1,1,6\n"
    path = _write_csv(tmp_path, content, "ratings.csv")

    with pytest.raises(InvalidDataError):
        load_ratings(path)


def test_load_ratings_zero_rating_raises(tmp_path):
    content = "user_id,product_id,rating\n1,1,0\n"
    path = _write_csv(tmp_path, content, "ratings.csv")

    with pytest.raises(InvalidDataError):
        load_ratings(path)


# ---------------------------------------------------------------------
# load_products
# ---------------------------------------------------------------------


def test_load_products_valid_file(tmp_path):
    content = "product_id,name,category,description\n1,Widget,Tools,A useful widget\n"
    path = _write_csv(tmp_path, content, "products.csv")

    products = load_products(path)
    assert len(products) == 1
    assert products.iloc[0]["name"] == "Widget"


def test_load_products_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_products(str(tmp_path / "missing.csv"))


def test_load_products_missing_column_raises(tmp_path):
    content = "product_id,name\n1,Widget\n"
    path = _write_csv(tmp_path, content, "products.csv")

    with pytest.raises(InvalidDataError):
        load_products(path)


def test_load_products_empty_file_raises(tmp_path):
    content = "product_id,name,category,description\n"
    path = _write_csv(tmp_path, content, "products.csv")

    with pytest.raises(InvalidDataError):
        load_products(path)