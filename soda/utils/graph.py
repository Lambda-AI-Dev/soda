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

    def get_u_neighbors(self, u):
        return self.U_neighbors[u]

    def get_v_neighbors(self, v):
        return self.V_neighbors[v]


