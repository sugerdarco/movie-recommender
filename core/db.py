import sqlite3
import json
import numpy as np

DB_PATH = "./db/movies.db"

def get_connection(db_path=DB_PATH):
    return sqlite3.connect(db_path)

def fetch_all_movies(db_path=DB_PATH):
    conn = get_connection(db_path)
    cursor = conn.execute("""
        SELECT id, title, genres, director, description, movie_id, embedding
        FROM movies
    """)

    movies, embeddings = [] , []
    for row in cursor.fetchall():
        movie = {
            "id": row[0],
            "title": row[1],
            "genres": json.loads(row[2]),
            "director": json.loads(row[3]),
            "description": row[4],
            "movie_id": row[5],
        }
        embedding = np.frombuffer(row[6], dtype=np.float32)
        movies.append(movie)
        embeddings.append(embedding)

    conn.close()
    return movies, np.vstack(embeddings) if embeddings else np.array([], dtype=np.float32)

def fetch_movies_by_id(movie_id, db_path=DB_PATH):
    conn = get_connection(db_path)
    cursor = conn.execute("""
        SELECT id, title, genres, director, description, movie_id, embedding
        FROM movies
        WHERE movie_id = ?
        """, (movie_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "id": row[0],
            "title": row[1],
            "genres": json.loads(row[2]),
            "director": json.loads(row[3]),
            "description": row[4],
            "movie_id": row[5],
            "embedding": np.frombuffer(row[6], dtype=np.float32),
        }
    return None

def insert_movie(db_path, movie, embedding):
    conn = get_connection(db_path)
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
    conn.commit()
    conn.close()
