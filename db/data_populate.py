# ###
# make a connection with db
# open a json file and sotred all data in sqlite
# with description senTran emdedding
# ###

import json
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "./db/movies.db"
JSON_PATH = "./db/movies_cleaned.json"

def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY,
            title TEXT,
            genres TEXT,
            director TEXT,
            description TEXT,
            movie_id TEXT,
            embedding BLOB
        )
    """)
    conn.commit()

def insert_movie(conn, movie, embedding):
    conn.execute("""
        INSERT INTO movies (id, title, genres, director, description, movie_id, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        movie["id"],
        movie["title"],
        json.dumps(movie["genres"]),
        json.dumps(movie["director"]),
        movie["description"],
        movie["movie_id"],
        embedding.astype(np.float32).tobytes()
    ))

def main():
    # load model once
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # read json file
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        movies = json.load(f)

    # connect DB_PATH
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    # Process movie (create a embedding) and insert in DB_PATH
    for movie in movies:
        # Weighted embedding
        title_emb = model.encode(movie["title"], normalize_embeddings=True, convert_to_numpy=True)
        desc_emb = model.encode(movie["description"], normalize_embeddings=True, convert_to_numpy=True)
        genre_emb = model.encode(" ".join(movie["genres"]), normalize_embeddings=True, convert_to_numpy=True)
        director_emb = model.encode(" ".join(movie["director"]), normalize_embeddings=True, convert_to_numpy=True)

        embedding = (
            0.5 * title_emb +
            0.3 * desc_emb +
            0.1 * genre_emb +
            0.1 * director_emb
        )
        embedding = embedding / np.linalg.norm(embedding)
        insert_movie(conn, movie, embedding)

    conn.commit()
    conn.close()
    print(f" inserted {len(movies)} movies into db {DB_PATH}")

if __name__ == "__main__":
    main()
