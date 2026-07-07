#!/usr/bin/env python3
"""
main.py
=======
Top-level entry point. Starts the Flask API server.
Run: python main.py
Then try: curl http://localhost:5000/api/recommend/user/1
"""

from src.api import create_app

if __name__ == "__main__":
    app = create_app()
    print("🛒 Product Recommendation Engine API running at http://localhost:5000")
    print("   Try: GET /api/recommend/user/<id>, /api/recommend/product/<id>, /api/popular")
    app.run(debug=False, host="0.0.0.0", port=5000)