import numpy as np  # open source library having support for large arrays and matrices and has collection of high level maths functions.
import hnswlib  # hnswlib - hierarchical navigable small world graphs - a fast library for ann(approximate nearest neighbour)
from typing import Literal, Union # literal- restricts the allowed constant values , union - union[A,B] means either A or B

class ANNIndex:  #defines a class that wraps hnswlib into a neat and easy to use interface
    def __init__(self, dim, space: Union[Literal["l2", "ip", "cosine"], str] ="cosine"): #constructor
        #dim-dimension-total numbers in each vector
        #l2-squared euclidean distance,ip-similarity based on inner product(dot product),cosine=1-cos(angle between 2 vectors)-default
        self.index: hnswlib.Index | None = None #it will later hold an hnswlib index or none
        self.built = False #boolean flag indicating that index is not fully constructed yet,set to true at end of build method
        self.max_elements = 0 #maximum number of vectors that index can hold currently
        self.num_elements = 0 #how many vectors are actually stroed right now
        self.dim = dim #saves the input dim on object so that other methods can use them
        self.space = space #saves the input space on object so that other methods can use them
        self.M = 16 #default graph parameter which controls maximum number of connections each node can have
        self.ef_construction = 200 #default build time search parameter,if it's higher it means more accurate graph but slower build
        self.ef_runtime = 50 #default query time parameter,if it's higher it means better recall but slower queries

    def build(self, vectors: np.ndarray, ef_construction=200, M=16, buffer=1000):
        #method to create an index and fill it with initital batch of vectors
        #vectors-a numpy array of shape (N,dim) where N is number of vectors and dim is their dimension
        #buufer=extra capacity the index can hold more than initial vectors
        num_elements = vectors.shape[0] #shape gives (rows,columns) of the matrix,shape[0] gives number of rows(=number of vectors)
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
