import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import networkx as nx
import sys

from sklearn import tree, linear_model
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from sklearn.neighbors import NearestCentroid
import scipy.cluster.hierarchy as hierarchy

import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, DenseDataLoader

from models import *

def load_syn_data(dataset_str):
    if dataset_str == "BA_Shapes":
        G = nx.readwrite.read_gpickle("../../data/BA_Houses/graph_ba_300_80.gpickel")
        role_ids = np.load("../../data/BA_Houses/role_ids_ba_300_80.npy")

    elif dataset_str == "BA_Grid":
        G = nx.readwrite.read_gpickle("../../data/BA_Grid/graph_ba_300_80.gpickel")
        role_ids = np.load("../../data/BA_Grid/role_ids_ba_300_80.npy")

    elif dataset_str == "BA_Community":
        G = nx.readwrite.read_gpickle("../../data/BA_Community/graph_ba_350_100_2comm.gpickel")
        role_ids = np.load("../../data/BA_Community/role_ids_ba_350_100_2comm.npy")

    elif dataset_str == "Tree_Cycle":
        G = nx.readwrite.read_gpickle("../../data/Tree_Cycle/graph_tree_8_60.gpickel")
        role_ids = np.load("../../data/Tree_Cycle/role_ids_tree_8_60.npy")

    elif dataset_str == "Tree_Grid":
        G = nx.readwrite.read_gpickle("../../data/Tree_Grid/graph_tree_8_80.gpickel")
        role_ids = np.load("../../data/Tree_Grid/role_ids_tree_8_80.npy")


    else:
        raise Exception("Invalid Syn Dataset Name")

    return G, role_ids


def load_real_data(dataset_str):
    if dataset_str == "Mutagenicity":
        graphs = TUDataset(root='.', name='Mutagenicity')

    elif dataset_str == "Reddit_Binary":
        graphs = TUDataset(root='.', name='REDDIT-BINARY', transform=torch_geometric.transforms.Constant())
    else:
        raise Exception("Invalid Real Dataset Name")

    print()
    print(f'Dataset: {graphs}:')
    print('====================')
    print(f'Number of graphs: {len(graphs)}')
    print(f'Number of features: {graphs.num_features}')
    print(f'Number of classes: {graphs.num_classes}')

    data = graphs[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    return graphs


def prepare_syn_data(G, labels, train_split, if_adj=False):
    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].size
    features = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        features[i, :] = G.nodes[u]["feat"]

    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels)

    edge_list = torch.from_numpy(np.array(G.edges))

    edges = edge_list.transpose(0, 1)
    if if_adj:
        edges = torch.tensor(nx.to_numpy_matrix(G), requires_grad=True, dtype=torch.float)

    train_mask = np.random.rand(len(features)) < train_split
    test_mask = ~train_mask

    print("Task: Node Classification")
    print("Number of features: ", len(features))
    print("Number of labels: ", len(labels))
    print("Number of classes: ", len(set(labels)))
    print("Number of edges: ", len(edges))

    return {"x": features, "y": labels, "edges": edges, "edge_list": edge_list, "train_mask": train_mask, "test_mask": test_mask}

def prepare_real_data(graphs, train_split, batch_size, dataset_str):
    graphs = graphs.shuffle()

    train_idx = int(len(graphs) * train_split)
    train_set = graphs[:train_idx]
    test_set = graphs[train_idx:]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    if dataset_str == "Mutagenicity":
        full_loader = DataLoader(test_set, batch_size=int(len(test_set)), shuffle=True)
        small_loader = DataLoader(test_set, batch_size=int(len(test_set) * 0.1))

    elif dataset_str == "Reddit_Binary":
        full_loader = DataLoader(test_set, batch_size=int(len(test_set) * 0.1), shuffle=True)
        small_loader = DataLoader(test_set, batch_size=int(len(test_set) * 0.005))

    train_zeros = 0
    train_ones = 0
    for data in train_set:
        train_ones += np.sum(data.y.detach().numpy())
        train_zeros += len(data.y.detach().numpy()) - np.sum(data.y.detach().numpy())

    test_zeros = 0
    test_ones = 0
    for data in test_set:
        test_ones += np.sum(data.y.detach().numpy())
        test_zeros += len(data.y.detach().numpy()) - np.sum(data.y.detach().numpy())

    print()
    print(f"Class split - Training 0: {train_zeros} 1:{train_ones}, Test 0: {test_zeros} 1: {test_ones}")


    return train_loader, test_loader, full_loader, small_loader

def set_rc_params():
    small = 14
    medium = 20
    large = 28

    plt.rc('figure', autolayout=True, figsize=(10, 6))
    plt.rc('font', size=medium)
    plt.rc('axes', titlesize=medium, labelsize=small, grid=True)
    plt.rc('xtick', labelsize=small)
    plt.rc('ytick', labelsize=small)
    plt.rc('legend', fontsize=small)
    plt.rc('figure', titlesize=large, facecolor='white')
    plt.rc('legend', loc='upper left')


def plot_activation_space(data, labels, activation_type, layer_num, path, note="", naming_help=""):
    rows = len(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"{activation_type} Activations of Layer {layer_num} {note}")

    scatter = ax.scatter(data[:,0], data[:,1], c=labels, cmap='rainbow')
    ax.legend(handles=scatter.legend_elements()[0], labels=list(np.unique(labels)), bbox_to_anchor=(1.05, 1))

    plt.savefig(os.path.join(path, f"{layer_num}_layer{naming_help}.png"))
    plt.show()


# def plot_clusters(data, labels, clustering_type, k, layer_num, path, data_type, reduction_type="", note=""):
#     fig, ax = plt.subplots(figsize=(10, 6))
#     fig.suptitle(f'{clustering_type} Clustered {data_type} Activations of Layer {layer_num} {note}')
#
#     for i in range(k):
#         scatter = ax.scatter(data[labels == i,0], data[labels == i,1], label=f'Cluster {i}')
#
#     ax.legend(bbox_to_anchor=(1.05, 1))
#     plt.savefig(os.path.join(path, f"{layer_num}layer_{data_type}{reduction_type}.png"))
#     plt.show()


def get_top_subgraphs(top_indices, y, edges, num_expansions, graph_data=None, graph_name=None):
    graphs = []
    color_maps = []
    labels = []
    node_labels = []

    df = pd.DataFrame(edges)

    for idx in top_indices:
        # get neighbours
        neighbours = list()
        neighbours.append(idx)

        for i in range(0, num_expansions):
            new_neighbours = list()
            for e in edges:
                if (e[0] in neighbours) or (e[1] in neighbours):
                    new_neighbours.append(e[0])
                    new_neighbours.append(e[1])

            neighbours = neighbours + new_neighbours
            neighbours = list(set(neighbours))

        new_G = nx.Graph()
        df_neighbours = df[(df[0].isin(neighbours)) & (df[1].isin(neighbours))]
        remaining_edges = df_neighbours.to_numpy()
        new_G.add_edges_from(remaining_edges)

        color_map = []
        node_label = {}
        if graph_data is None:
            for node in new_G:
                if node in top_indices:
                    color_map.append('green')
                else:
                    color_map.append('pink')
        else:
            if graph_name == "Mutagenicity":
                ids = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]
            elif graph_name == "REDDIT-BINARY":
                ids = []

            for node in zip(new_G):
                node = node[0]
                color_idx = graph_data[node]
                color_map.append(color_idx)
                node_label[node] = f"{ids[color_idx]}"

        color_maps.append(color_map)
        graphs.append(new_G)
        labels.append(y[idx])
        node_labels.append(node_label)


    return graphs, color_maps, labels, node_labels

def get_node_distances(clustering_model, data):
    if isinstance(clustering_model, AgglomerativeClustering) or isinstance(clustering_model, DBSCAN):
        x, y_predict = data
        clf = NearestCentroid()
        clf.fit(x, y_predict)
        centroids = clf.centroids_
        res = pairwise_distances(centroids, x)
        res_sorted = np.argsort(res, axis=-1)
    elif isinstance(clustering_model, KMeans):
        res_sorted = clustering_model.transform(data)

    return res_sorted


def plot_clusters(data, labels, clustering_type, k, layer_num, path, data_type, reduction_type="", note=""):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'{clustering_type} Clustered {data_type} Activations of Layer {layer_num} {note}')

    for i in range(k):
        scatter = ax.scatter(data[labels == i,0], data[labels == i,1], label=f'Cluster {i}')

    ncol = 1
    if k > 20:
        ncol = int(k / 20) + 1
    ax.legend(bbox_to_anchor=(1.05, 1), ncol=ncol)
    plt.savefig(os.path.join(path, f"{layer_num}layer_{data_type}{reduction_type}.png"))
    plt.show()


def plot_samples(clustering_model, data, y, layer_num, k, clustering_type, reduction_type, num_nodes_view, edges, num_expansions, path, graph_data=None, graph_name=None):
    res_sorted = get_node_distances(clustering_model, data)

    if isinstance(num_nodes_view, int):
        num_nodes_view = [num_nodes_view]
    col = sum([abs(number) for number in num_nodes_view])

    fig, axes = plt.subplots(k, col, figsize=(18, 3 * k + 2))
    fig.suptitle(f'Nearest Instances to {clustering_type} Cluster Centroid for {reduction_type} Activations of Layer {layer_num}', y=1.005)

    if graph_data is not None:
        fig2, axes2 = plt.subplots(k, col, figsize=(18, 3 * k + 2))
        fig2.suptitle(f'Nearest Instances to {clustering_type} Cluster Centroid for {reduction_type} Activations of Layer {layer_num} (by node index)', y=1.005)

    l = list(range(0, k))
    sample_graphs = []
    sample_feat = []

    for i, ax_list in zip(l, axes):
        if isinstance(clustering_model, AgglomerativeClustering) or isinstance(clustering_model, DBSCAN):
            distances = res_sorted[i]
        elif isinstance(clustering_model, KMeans):
            distances = res_sorted[:, i]

        top_graphs, color_maps = [], []
        for view in num_nodes_view:
            if view < 0:
                top_indices = np.argsort(distances)[::][view:]
            else:
                top_indices = np.argsort(distances)[::][:view]

            tg, cm, labels, node_labels = get_top_subgraphs(top_indices, y, edges, num_expansions, graph_data, graph_name)
            top_graphs = top_graphs + tg
            color_maps = color_maps + cm

        if graph_data is None:
            for ax, new_G, color_map, g_label in zip(ax_list, top_graphs, color_maps, labels):
                nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax)
                ax.set_title(f"label {g_label}", fontsize=14)
        else:
            for ax, new_G, color_map, g_label, n_labels in zip(ax_list, top_graphs, color_maps, labels, node_labels):
                nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax, labels=n_labels)
                ax.set_title(f"label {g_label}", fontsize=14)

            for ax, new_G, color_map, g_label, n_labels in zip(axes2[i], top_graphs, color_maps, labels, node_labels):
                nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax)
                ax.set_title(f"label {g_label}", fontsize=14)

        sample_graphs.append((top_graphs[0], top_indices[0]))
        sample_feat.append(color_maps[0])

    views = ''.join((str(i) + "_") for i in num_nodes_view)
    if isinstance(clustering_model, AgglomerativeClustering):
        fig.savefig(os.path.join(path, f"{k}h_{layer_num}layer_{clustering_type}_{views}view.png"))
    else:
        fig.savefig(os.path.join(path, f"{layer_num}layer_{clustering_type}_{reduction_type}_{views}view.png"))

    if graph_data is not None:
        if isinstance(clustering_model, AgglomerativeClustering):
            fig2.savefig(os.path.join(path, f"{k}h_{layer_num}layer_{clustering_type}_{views}view_by_node.png"))
        else:
            fig2.savefig(os.path.join(path, f"{layer_num}layer_{clustering_type}_{reduction_type}_{views}view_by_node.png"))

    plt.show()

    return sample_graphs, sample_feat


# def plot_samples(clustering_model, data, y, layer_num, k, clustering_type, reduction_type, num_nodes_view, edges, num_expansions, path, graph_data=None):
#     res_sorted = get_node_distances(clustering_model, data)
#
#     if isinstance(num_nodes_view, int):
#         num_nodes_view = [num_nodes_view]
#     col = sum([abs(number) for number in num_nodes_view])
#
#     fig, axes = plt.subplots(k, col, figsize=(18, 3 * k))
#     fig.suptitle(f'Nearest Instances to {clustering_type} Cluster Centroid for {reduction_type} Activations of Layer {layer_num}')
#
#     l = list(range(0, k))
#     sample_graphs = []
#     sample_feat = []
#
#     for i, ax_list in zip(l, axes):
#         if isinstance(clustering_model, AgglomerativeClustering) or isinstance(clustering_model, DBSCAN):
#             distances = res_sorted[i]
#         elif isinstance(clustering_model, KMeans):
#             distances = res_sorted[:, i]
#
#         top_graphs, color_maps = [], []
#         for view in num_nodes_view:
#             if view < 0:
#                 top_indices = np.argsort(distances)[::][view:]
#             else:
#                 top_indices = np.argsort(distances)[::][:view]
#
#             tg, cm, labels = get_top_subgraphs(top_indices, y, edges, num_expansions, graph_data)
#             top_graphs = top_graphs + tg
#             color_maps = color_maps + cm
#
#         for ax, new_G, color_map, g_label in zip(ax_list, top_graphs, color_maps, labels):
#             nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax)
#             ax.set_title(f"label {g_label}", fontsize=14)
#
#         sample_graphs.append((top_graphs[0], top_indices[0]))
#         sample_feat.append(color_maps[0])
#
#     views = ''.join((str(i) + "_") for i in num_nodes_view)
#     if isinstance(clustering_model, AgglomerativeClustering):
#         plt.savefig(os.path.join(path, f"{k}h_{layer_num}layer_{clustering_type}_{views}view.png"))
#     else:
#         plt.savefig(os.path.join(path, f"{layer_num}layer_{clustering_type}_{views}view.png"))
#     plt.show()
#
#     return sample_graphs, sample_feat


def plot_dendrogram(data, reduction_type, layer_num, path):
    """Learned from: https://medium.com/@sametgirgin/hierarchical-clustering-model-in-5-steps-with-python-6c45087d4318 """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'HC Dendrogram of {reduction_type} Activation Space of Layer {layer_num}')

    dendrogram = hierarchy.dendrogram(hierarchy.linkage(data, method='average'), truncate_mode="lastp", ax=ax, leaf_rotation=90.0, leaf_font_size=14)
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Euclidean Distances")

    plt.savefig(os.path.join(path, f"hc_dendrograms_{reduction_type}.png"))
    plt.show()


def plot_completeness_table(model_type, calc_type, data, path):
    fig, ax = plt.subplots(figsize=(10, 2 * len(data)))
    headings = ["Data", "Layer", "Completeness Score"]

    ax.set_title(f"Completeness Score (Task Accuracy) for {model_type} Models using {calc_type}")
    ax.axis('off')
    ax.table(cellText=data, colLabels=headings, loc="center", rowLoc="center", cellLoc="center", colLoc="center", fontsize=18)

    calc_type = calc_type.replace(" ", "")
    plt.savefig(os.path.join(path, f"{model_type}_{calc_type}_completeness.png"))
    plt.show()


def calc_graph_similarity(top_graphs, max_nodes, num_nodes_view):
    top_G = top_graphs[0]

    print("Nodes ", top_G.number_of_nodes(), " Graphs ", len(top_graphs))

    if top_G.number_of_nodes() > max_nodes:
        return "skipping (too many nodes)"

    if_iso = True

    for G in top_graphs[1:]:
        if not nx.is_isomorphic(top_G, G):
            if_iso = False
            break

    if if_iso:
        return 0

    total_score = 0
    for G in top_graphs[1:]:

        if G.number_of_nodes() > max_nodes:
            return "skipping (too many nodes)"

        total_score += min(list(nx.optimize_graph_edit_distance(top_G, G)))

    return total_score / (len(top_graphs) - 1)


def plot_graph_similarity_table(model_type, data, path):
    fig, ax = plt.subplots(figsize=(10, 0.25 * len(data)))
    headings = ["Model", "Data", "Layer", "Concept/Cluster", "Graph Similarity Score"]

    ax.set_title(f"Graph Similarity for Concepts extracted using {model_type}")
    ax.axis('off')
    ax.table(cellText=data, colLabels=headings, loc="center", rowLoc="center", cellLoc="center", colLoc="center", fontsize=18)

    plt.savefig(os.path.join(path, f"{model_type}_graph_similarity.png"))
    plt.show()


def prepare_output_paths(dataset_name, k):
    path = f"output/{dataset_name}/"
    path_tsne = os.path.join(path, "TSNE")
    path_pca = os.path.join(path, "PCA")
    path_umap = os.path.join(path, "UMAP")
    path_kmeans = os.path.join(path, f"{k}_KMeans")
    path_hc = os.path.join(path, f"HC")
    path_ward = os.path.join(path, f"WARD")
    path_dbscan = os.path.join(path, f"DBSCAN")
    os.makedirs(path, exist_ok=True)
    os.makedirs(path_tsne, exist_ok=True)
    os.makedirs(path_pca, exist_ok=True)
    os.makedirs(path_umap, exist_ok=True)
    os.makedirs(path_kmeans, exist_ok=True)
    os.makedirs(path_hc, exist_ok=True)
    os.makedirs(path_ward, exist_ok=True)
    os.makedirs(path_dbscan, exist_ok=True)

    return {"base": path, "TSNE": path_tsne, "PCA": path_pca, "UMAP": path_umap, "KMeans": path_kmeans, "HC": path_hc, "Ward": path_ward, "DBSCAN": path_dbscan}
