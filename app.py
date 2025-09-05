import sys
from typing import List, Dict
import sqlite3
from core.recommender import Recommender


# Function to compute/build embeddings and index
def initialize_recommender():
    rec = Recommender()
    rec.build()  # compute embeddings and build HNSW index
    return rec

# Function to handle user input and give recommendations
def query_recommendations(rec, movie_id=None)  -> List[Dict]:
    if not movie_id:
        movie_id = "tt1000092"  # default movie ID

    # print(f"\n Recommendations for movie_id: {movie_id}\n")
    try:
        recommendations = rec.recommend(movie_id, top_k=5)
        for r in recommendations:
            print(f"- {r['title']} ({r['movie_id']}) | Score: {r['score']:.4f}")
        return recommendations
    except ValueError as e:
        print(f"{e}")
        return []

    # # Optional loop for additional queries
    # while True:
    #     movie_id = input("\nEnter another movie ID (or 'exit' to quit): ").strip()
    #     if movie_id.lower() == "exit":
    #         break
    #     try:
    #         recommendations = rec.recommend(movie_id, top_k=5)
    #         print(f"\n🔍 Recommendations for movie_id: {movie_id}\n")
    #         for r in recommendations:
    #             print(f"- {r['title']} ({r['movie_id']}) | Score: {r['score']:.4f}")
    #     except ValueError as e:
    #         print(f"{e}")



# # Main function
# def main():
#     # Step 1: Compute/build embeddings and index once
#     rec = initialize_recommender()

#     # Step 2: Take initial movie ID from command line
#     movie_id = sys.argv[1] if len(sys.argv) > 1 else None
#     query_recommendations(rec, movie_id)

# # fetch all movies title and movie-id

DB_PATH = "./db/movies.db"  # Update with your database path

def fetch_all_titles():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT title, movie_id FROM movies")
    rows = cur.fetchall()

    conn.close()

    # Convert rows to list of dicts
    result = [{"title": row["title"], "movie_id": row["movie_id"]} for row in rows]
    return result


def fetch_movies_by_ids(movie_ids):
    if not movie_ids:
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    placeholders = ",".join("?" * len(movie_ids))
    query = f"SELECT * FROM movies WHERE movie_id IN ({placeholders})"
    cur.execute(query, movie_ids)
    rows = cur.fetchall()
    conn.close()

    return [dict(row) for row in rows]
