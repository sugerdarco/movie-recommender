import numpy as np
import hnswlib
from typing import Literal, Union

class ANNIndex:
    def __init__(self, dim, space: Union[Literal["l2", "ip", "cosine"], str] ="cosine"):
        self.index: hnswlib.Index | None = None
        self.built = False
        self.max_elements = 0
        self.num_elements = 0
        self.dim = dim
        self.space = space
        self.M = 16
        self.ef_construction = 200
        self.ef_runtime = 50

    def build(self, vectors: np.ndarray, ef_construction=200, M=16, buffer=1000):
        num_elements = vectors.shape[0]
        self.max_elements = num_elements + buffer
        self.num_elements = num_elements
        self.M = M
        self.ef_construction = ef_construction

        # init index
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(
            max_elements=self.max_elements,
            ef_construction=ef_construction,
            M=M
        )
        self.index.add_items(vectors, list(range(num_elements)))
        self.index.set_ef(self.ef_runtime)
        self.built = True

    # rebuild with larger capacity
    def _rebuild_with_buffer(self, extra=1000):
        if self.index is None:
            raise RuntimeError("Index not initialized.")

        print(f"Rebuilding ANN index with extra capacity (+{extra})")

        # fetch all current items
        ids = list(range(self.num_elements))
        vectors = self.index.get_items(ids)

        # new capacity
        self.max_elements = self.num_elements + extra

        # rebuild
        new_index = hnswlib.Index(space=self.space, dim=self.dim)
        new_index.init_index(
            max_elements=self.max_elements,
            ef_construction=self.ef_construction,
            M=self.M
        )
        new_index.add_items(vectors, ids)
        new_index.set_ef(self.ef_runtime)

        self.index = new_index

    def add(self, vectors: np.ndarray, ids: list[int]):
        if not self.built or self.index is None:
            raise RuntimeError("Index not built yet.")

        # If adding more than capacity, rebuild with a bigger cap
        if self.num_elements + len(ids) > self.max_elements:
            self._rebuild_with_buffer(extra=max(1000, len(ids)))

        self.index.add_items(vectors, ids)
        self.num_elements += len(ids)

    def query(self, vector, k=5):
        if not self.built or self.index is None:
            raise RuntimeError("Index not built yet.")
        labels, distances = self.index.knn_query(vector.reshape(1, -1), k=k)
        return labels[0], distances[0]
