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
        self.U_neighbors[u].append((v, e))
        self.V_neighbors[v].append((u, e))

    def add_edges(self, iterable):
        for u, v, e in iterable:
            self.add_edge(u, v, e)

    def add_edges_t(self, U, V, E):
        self.add_edges(zip(U, V, E))

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
