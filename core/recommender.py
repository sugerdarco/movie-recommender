import numpy as np
from core import db
from core.ann_index import ANNIndex
from sentence_transformers import SentenceTransformer

DB_PATH = "./db/movies.db"

class Recommender:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.movies = []
        self.embeddings = None
        self.index = None
        self.movie_id_map = {}

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def build(self):
        # load data from db
        self.movies, self.embeddings = db.fetch_all_movies(self.db_path)
        if self.embeddings.size == 0:
            raise RuntimeError("No embeddings found in database")
        dim = self.embeddings.shape[1]

        # build ANN index
        self.index = ANNIndex(dim=dim, space="cosine")
        self.index.build(self.embeddings)

        # Map movie_id with row index
        self.movie_id_map = {m["movie_id"]: i for i, m in enumerate(self.movies)}

    def add_movie(self, movie: dict):
        if self.index is None:
            raise RuntimeError("Index not initialized. Call build() first.")

        title_emb = self.model.encode(movie["title"], normalize_embeddings=True, convert_to_numpy=True)
        desc_emb = self.model.encode(movie["description"], normalize_embeddings=True, convert_to_numpy=True)
        genre_emb = self.model.encode(" ".join(movie["genres"]), normalize_embeddings=True, convert_to_numpy=True)
        director_emb = self.model.encode(" ".join(movie["director"]), normalize_embeddings=True, convert_to_numpy=True)

        embedding = (
            0.5 * title_emb +
            0.3 * desc_emb +
            0.1 * genre_emb +
            0.1 * director_emb
        )
        embedding = embedding / np.linalg.norm(embedding)

        # save to db
        db.insert_movie(self.db_path, movie, embedding)

        # update in-memory
        idx = len(self.movies)
        self.movies.append(movie)

        if self.embeddings is not None and self.embeddings.size > 0:
            self.embeddings = np.vstack([self.embeddings, embedding])
        else:
            self.embeddings = embedding.reshape(1, -1);

        # add to ann index
        self.index.add(embedding.reshape(1, -1), [idx])

        # update ID map
        self.movie_id_map[movie["movie_id"]] = idx

    def recommend(self, movie_id: str, top_k: int=5):
        if self.embeddings is None or self.index is None:
            raise RuntimeError("Recommender not built. Call build() first.")

        if movie_id not in self.movie_id_map:
            raise ValueError(f"Movie ID {movie_id} not found in database")

        idx = self.movie_id_map[movie_id]
        query_vector = self.embeddings[idx]

        labels, distances = self.index.query(query_vector, k=top_k + 1) # +1 to skip itself
        labels, distances = labels[1:], distances[1:]

        results = []
        for i, dist in zip(labels, distances):
            movie = self.movies[i]
            results.append({
                "movie_id": movie["movie_id"],
                "title": movie["title"],
                "genres": movie["genres"],
                "director": movie["director"],
                "score": float(1 - dist)  # cosine similarity (1 - distance)
            })

        return results
