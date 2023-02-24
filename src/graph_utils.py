import torch as t

def adj_to_edge_index(adj):
    return adj.nonzero().T.contiguous()

def edge_index_to_adj(edge_index):
    n = edge_index.max() + 1
    tensor = t.zeros((n, n), dtype=t.long)
    tensor[edge_index[0], edge_index[1]] = 1
    return tensor