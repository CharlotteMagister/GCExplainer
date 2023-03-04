import random
import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
from gcf import Graph
import torch as t
import torch_geometric as pyg

def house():
    edges = [
        [0, 1],
        [0, 3],
        [0, 4],
        [1, 4],
        [1, 2],
        [1, 0],
        [2, 1],
        [2, 3],
        [3, 2],
        [3, 0],
        [4, 0],
        [4, 1]
    ]
    labels = [1, 1, 2, 2, 3]
    return Graph((labels, edges))

def triangle():
    # create networkx triangle graph:
    return Graph(
        pyg.data.Data(
            x = t.tensor([0, 1, 2]),
            edge_index = t.tensor([[0, 1, 2, 0, 1, 2], [1, 2, 0, 2, 0, 1]])
        )
    )


def ring(n):
    x = t.ones((n, 1))
    edge_index = t.stack([t.arange(n), t.arange(n) + 1], dim=0)
    edge_index[1][-1] = 0
    edge_index_flipped = t.cat([edge_index[1].unsqueeze(dim=0), edge_index[0].unsqueeze(dim=0)], dim=0)
    edge_index = t.cat([edge_index, edge_index_flipped], dim=1)
    return Graph(pyg.data.Data(x, edge_index))


def ring_with_attachments(num_nodes, shape_gen, num_shapes):
    r = ring(num_nodes)
    connecting_nodes = np.random.permutation(num_nodes)[:num_shapes]
    for i in range(num_shapes):
        prev_nodes = r.num_nodes()
        shape_graph = shape_gen()
        # merge the networkx graphs ring and shape_graph,
        # connecting them at the node connecting_nodes[i]:
        r = r.disjoint_union(shape_graph)
        r.add_edge(
            connecting_nodes[i],
            prev_nodes + 1
        )
    return r