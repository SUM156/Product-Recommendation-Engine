# 🛒 Product Recommendation Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/tests-47%20passed-4CAF50?style=flat-square&logo=pytest&logoColor=white" alt="47 tests passed">
  <img src="https://img.shields.io/badge/API-Flask-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask API">
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="MIT License">
</p>

<p align="center">
  <b>A hybrid recommendation system built from scratch — user-based collaborative filtering, item-item similarity, and content-based cold-start handling — the same category of system that drives ~35% of Amazon's revenue via "Customers also bought."</b>
</p>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [The Three Recommendation Strategies](#-the-three-recommendation-strategies)
- [Technology Stack](#-technology-stack)
- [Architecture](#-architecture)
- [Folder Structure](#-folder-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Testing](#-testing)
- [Demo](#-demo)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

This project implements a recommendation engine the way it actually works under the hood — no black-box `recommend()` library call. The math (cosine similarity over a user-item ratings matrix, TF-IDF over product text) is implemented directly on top of pandas/scikit-learn primitives, fully visible and fully tested, then exposed as a real Flask REST API.

## ❓ Problem Statement

Every e-commerce platform faces the same core challenge: **most users have rated only a tiny fraction of the catalog** (this dataset is 72.8% sparse — realistic for real platforms), and **new users/products have no history at all**. A recommendation engine has to (1) find meaningful patterns in extremely sparse data, and (2) gracefully handle the "cold start" case where collaborative filtering has nothing to work with. This project solves both.

## ✨ Features

- 👥 **User-based collaborative filtering** — "users like you also liked these," via cosine similarity between user rating vectors.
- 🔗 **Item-item similarity matrix** — "customers who bought X also bought Y," independent of any specific user.
- 📝 **Content-based filtering (TF-IDF)** — similarity from product category + description text, for products/users with no rating history.
- 🥶 **Automatic cold-start handling** — brand-new users transparently get Bayesian-smoothed popularity recommendations instead of an empty result.
- 🌐 **Flask REST API** — three endpoints, JSON responses, clean 404 error handling for unknown users/products.
- ⚖️ **Bayesian-smoothed popularity ranking** — prevents a single 5-star review from outranking a product with hundreds of consistently good ratings (the same fix IMDB uses for its Top 250).
- ✅ **47 automated tests**, including hand-verified similarity math (e.g. "identical rating vectors → similarity exactly 1.0") and full Flask API integration tests.

## 🧠 The Three Recommendation Strategies

| Strategy | Question it answers | Needs rating history? |
|---|---|---|
| **User-based CF** | "What would THIS user like, based on similar users?" | Yes — for the target user |
| **Item-based similarity** | "What else do people who buy THIS product also buy?" | Yes — for the product |
| **Content-based (TF-IDF)** | "What's textually similar to THIS product?" | No — works from day one |
| **Popularity (Bayesian)** | "What's broadly well-liked across everyone?" | No — the cold-start fallback |

`recommend_for_user` automatically picks the right strategy: collaborative filtering when there's enough signal, popularity-based cold start when there isn't — a user is **never** shown an empty result just because they're new.

## 🛠️ Technology Stack

| Layer | Technology | Why |
|---|---|---|
| Similarity math | scikit-learn (`cosine_similarity`, `TfidfVectorizer`) | Battle-tested, vectorized implementations |
| Data processing | Pandas | User-item matrix construction, aggregation |
| API | Flask | Lightweight REST API, application factory pattern |
| Testing | `pytest` + Flask test client | Hand-verified math + full HTTP integration tests |

## 🏗️ Architecture

```
ratings.csv + products.csv
        ↓
data_loader.py         ← Load, validate ratings + product catalog
        ↓
similarity.py             ← Build user-item matrix, user & item cosine similarity
        ↓
content_based.py             ← TF-IDF product-text similarity (cold-start)
        ↓
recommender.py                   ← Hybrid orchestration: CF → item-sim → popularity fallback
        ↓
api.py                               ← Flask REST API (application factory)
```

**Key design decision — similarity math has zero fallback logic:** `similarity.py` and `content_based.py` compute pure math and nothing else. `recommender.py` is the ONLY module that decides "if collaborative filtering returns nothing, fall back to popularity." This separation means the similarity math is testable with tiny hand-built matrices where the correct cosine similarity is calculable by hand (e.g. identical vectors → exactly 1.0, orthogonal vectors → exactly 0.0), completely independent of the fallback decision logic.

**Key design decision — Bayesian-smoothed popularity, not raw averages:** A product with one 5-star rating and a product with 200 ratings averaging 4.8 are NOT equally "popular" — the smoothing formula pulls low-count products toward the global mean rating, proportional to how few ratings they have. Verified with a hand-built test: 3 ratings of 5 outranks 1 rating of 5, despite an identical raw average.

## 📁 Folder Structure

```
day25_recommendation_engine/
├── main.py                       # Entry point — starts the Flask API server
├── requirements.txt
├── README.md
├── GUIDE.txt                       # Roman Urdu setup guide
├── data/
│   ├── ratings.csv                   # 653 ratings, 60 users, 72.8% sparse
│   └── products.csv                    # 40 products across 4 categories
├── src/
│   ├── __init__.py
│   ├── exceptions.py
│   ├── data_loader.py                    # CSV loading + validation
│   ├── similarity.py                        # User-item matrix + cosine similarity
│   ├── content_based.py                        # TF-IDF content similarity
│   ├── recommender.py                              # Hybrid orchestration
│   └── api.py                                          # Flask REST API
└── tests/
    ├── test_data_loader.py
    ├── test_similarity.py                             # Hand-verified cosine similarity math
    ├── test_content_based.py
    ├── test_recommender.py                                 # Hand-verified CF predictions
    └── test_api.py                                            # Full Flask integration tests
```

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/product-recommendation-engine.git
cd product-recommendation-engine
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Usage

```bash
# Start the API server
python main.py

# In another terminal:
curl http://localhost:5000/api/recommend/user/1
curl http://localhost:5000/api/recommend/product/5
curl http://localhost:5000/api/popular
```

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Liveness check, returns loaded product count |
| `/api/recommend/user/<user_id>?top_n=5` | GET | Hybrid recommendations for a user (CF or cold-start popularity) |
| `/api/recommend/product/<product_id>?top_n=5` | GET | "Customers also bought" (item similarity, content-based fallback) |
| `/api/popular?top_n=5` | GET | Globally popular products (Bayesian-smoothed ranking) |

Every response includes a `"method"` field telling you exactly which strategy produced the result — `collaborative_filtering`, `item_similarity`, `content_based_cold_start`, or `popularity_cold_start`.

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

**Result: 47/47 tests passing** — including hand-verified cosine similarity (identical vectors → 1.0, orthogonal vectors → 0.0), hand-verified CF predictions (single-neighbor case collapses to an exact expected value), and full Flask API integration tests.

## 🎬 Demo

Real output from the bundled 60-user, 40-product dataset:

```
=== User with rich history (user_id=1) ===
Method: collaborative_filtering
  USB-C Hub 7-in-1 (Electronics) - score 5.0
  Atomic Habits (Books) - score 5.0
  The Lean Startup (Books) - score 5.0

=== TRUE cold-start user (user_id=59, ZERO ratings ever) ===
Method: popularity_cold_start
  Smart Watch Series 5 - Bayesian score 4.234
  Mechanical Keyboard - Bayesian score 4.226
  Sapiens - Bayesian score 4.168

=== "Customers who bought Mechanical Keyboard also bought" ===
  Webcam 1080p (Electronics) - similarity 0.546
  Smart Watch Series 5 (Electronics) - similarity 0.528
  Wireless Charging Pad (Electronics) - similarity 0.509
```

The cold-start user correctly triggered the popularity fallback with zero errors — the hybrid design working exactly as intended, on a genuinely never-rated user.

<img width="1149" height="633" alt="Screenshot 2026-07-07 102335" src="https://github.com/user-attachments/assets/d69d6303-4c3e-40a0-b04f-b439b6c6582b" />
<img width="1015" height="665" alt="Screenshot 2026-07-07 102515" src="https://github.com/user-attachments/assets/098c38c5-34bb-404d-838e-6800866e0c0b" />

## 🗺️ Future Roadmap

- [ ] Matrix factorization (SVD/ALS) for better sparse-data performance at scale
- [ ] A/B testing framework to compare strategies' real click-through rates
- [ ] Implicit feedback support (views/clicks, not just explicit ratings)
- [ ] Redis caching layer for the similarity matrices in production
- [ ] Diversity/serendipity re-ranking (avoid recommending 5 near-identical products)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for any new logic (hand-verify math where possible)
4. Ensure `pytest tests/` passes before opening a PR

## 📄 License

MIT License — free to use, modify, and distribute.
