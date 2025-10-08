Example graph

Graph with 3 vertices: A, B, C

Edges:

A → B, weight 5

A → C, weight 2

B → C, weight 1

1. Adjacency list
E = {
    'A': [('B', 5), ('C', 2)],
    'B': [('C', 1)],
    'C': []
}


Here, for each vertex, we only store the neighbors that exist and their weights.

for v, w in E[u]: will only loop over existing edges.

2. Adjacency matrix
V = ['A', 'B', 'C']
A = [
    [0, 5, 2],    # A → A=0, A → B=5, A → C=2
    [float('inf'), 0, 1],  # B → A=∞, B → B=0, B → C=1
    [float('inf'), float('inf'), 0]  # C → no outgoing edges
]


Here, each row corresponds to a vertex, and each column represents a possible edge to every vertex.

You would loop over all columns:

for v in range(len(V)):
    if A[u][v] != float('inf'):
        # relax edge


Even if most entries are ∞ (no edge), you still check every vertex.

Key difference:

Aspect	Adjacency List	Adjacency Matrix
Storage	Only store existing edges	Store all possible edges
Looping	Only neighbors of u	All vertices v = 0..V-1
Sparse graph	Very efficient	Inefficient (many ∞ entries)
