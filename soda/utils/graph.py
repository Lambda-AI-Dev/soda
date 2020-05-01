import numpy as np
from scipy import sparse


class _IdentityMap:
    """
    _IdentityMap()[x] returns exactly x
    """
    def __getitem__(self, item):
        return item


class BipartiteGraph:
    def __init__(self):
        self.U_neighbors = {}  # u -> [v1, v2, ...]
        self.V_neighbors = {}  # v -> [u1, u2, ...]
        self.edges = {}  # (u, v) -> e
        # assume that we always know the query order

    def add_edge(self, u, v, e):
        if u not in self.U_neighbors:
            self.U_neighbors[u] = []
        if v not in self.V_neighbors:
            self.V_neighbors[v] = []
        self.U_neighbors[u].append(v)
        self.V_neighbors[v].append(u)
        self.edges[(u, v)] = e

    def add_edges(self, iterable, rename_u=None, rename_v=None):
        if rename_u is None:
            rename_u = _IdentityMap()
        if rename_v is None:
            rename_v = _IdentityMap()
        for u, v, e in iterable:
            self.add_edge(rename_u[u], rename_v[v], e)

    def add_edges_t(self, U, V, E, rename_u=None, rename_v=None):
        self.add_edges(zip(U, V, E), rename_u=rename_u, rename_v=rename_v)
        return self

    def add_edges_sparse(self, sp_matrix, rename_u=None, rename_v=None):
        """
        convert a sparse matrix into a bipartite graph:
        U: row indices
        V: column indices
        E: nonzero entries in the matrix

        Note that this only applies to 2d matrices, so 3d array of probability vectors
        (in the context of crowd estimator) doesn't follow this format
        """
        U, V, E = sparse.find(sp_matrix)
        self.add_edges_t(U, V, E, rename_u=rename_u, rename_v=rename_v)
        return self

    def get_u_neighbors(self, u):
        return self.U_neighbors[u]

    def get_v_neighbors(self, v):
        return self.V_neighbors[v]

    def agg_u(self, func):
        """
        func takes in an iterable of (u, v, e) pairs
        the aggregate output is of the form u -> result
        """
        ret = {}
        for u in self.U_neighbors:
            ret[u] = func((u, v, self.edges[(u, v)]) for v in self.U_neighbors[u])
        return ret

    def agg_v(self, func):
        ret = {}
        for v in self.V_neighbors:
            ret[v] = func((u, v, self.edges[(u, v)]) for u in self.V_neighbors[v])
        return ret

    def peak_edge_isscalar(self):
        for x in self.edges:
            return np.isscalar(self.edges[x])
        return False  # if there's no edge in the bipartite graph, treat as if not a scalar
