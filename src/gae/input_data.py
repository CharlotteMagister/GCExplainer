import random
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ["x", "tx", "allx", "graph"]
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset)
    )
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1
        )
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def house():
    edges = np.array(
        [
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
            [4, 1],
        ]
    )
    labels = [1, 1, 2, 2, 3]
    return edges, labels


def gen_ring(len):
    sources = np.arange(0, len)
    targets = np.concatenate((np.arange(1, len), np.array([0])))

    print(sources.shape)
    print(targets.shape)

    edges = np.vstack(
        (
            np.concatenate([sources, targets]),
            np.concatenate([targets, sources]),
        )
    ).transpose()

    labels = np.zeros(len)

    G = nx.Graph()
    G.add_edges_from(edges)
    # nx.draw(G)
    return G


# Based on pytorch geometric implementation
def bhshapes():
    # Build the Barabasi-Albert graph:
    num_nodes = FLAGS.num_bh_nodes
    random.seed(1234)
    np.random.seed(1234)

    if FLAGS.graph_type == "bashapes":
        ba_graph = nx.barabasi_albert_graph(num_nodes, 1)
    elif FLAGS.graph_type == "ring":
        ba_graph = gen_ring(num_nodes)
    else:
        print("NO GRAPH TYPE SPECIFIED")
        exit(1)
    node_labels = [0 for _ in range(num_nodes)]

    # Select nodes to connect shapes:
    num_houses = FLAGS.num_bh_houses
    connecting_nodes = np.random.permutation(num_nodes)[:num_houses]

    # Connect houses to Barabasi-Albert graph:
    for i in range(num_houses):
        house_edges, house_labels = house()
        house_edges += num_nodes
        house_edges = [(e[0], e[1]) for e in house_edges]
        print(f"Connection house to node {int(connecting_nodes[i])}")
        ba_graph.add_edges_from(
            house_edges
            + [
                (int(connecting_nodes[i]), num_nodes),
                (num_nodes, int(connecting_nodes[i])),
            ]
        )
        num_nodes += 5
        node_labels += house_labels

    # nx.draw(ba_graph)
    # plt.show()

    nx.write_adjlist(ba_graph, f"{FLAGS.save_prefix}.adjlist")
    np.save(f"{FLAGS.save_prefix}.labels", np.array(node_labels))

    # Identity is possibly a bad choice for feature here... TODO
    if FLAGS.bh_features == "ones":
        return nx.adjacency_matrix(ba_graph), sp.csc_matrix(
            np.ones((num_nodes, 1))
        )
    elif FLAGS.bh_features == "identity":
        return nx.adjacency_matrix(ba_graph), sp.identity(num_nodes)
    else:
        print(f"ERROR: invalid feature option: {FLAGS.bh_features}")
        exit(1)
