#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import networkx as nx

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, convert
from torch_geometric.data import DataLoader, DenseDataLoader
from torch_geometric.nn import GNNExplainer, GCNConv
from torch_geometric.nn import global_mean_pool, GlobalAttention

from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn import tree, linear_model

import scipy.cluster.hierarchy as hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestCentroid

from torch_geometric.nn import GNNExplainer

from utilities import *
from heuristics import *
from activation_classifier import *

set_rc_params()

np.random.seed(3)
torch.manual_seed(3)


# In[2]:


k = 10

path = "output/mutag/"
path_tsne = os.path.join(path, "TSNE")
path_pca = os.path.join(path, "PCA")
path_kmeans = os.path.join(path, f"{k}_KMeans")
path_hc = os.path.join(path, f"HC")
os.makedirs(path, exist_ok=True)
os.makedirs(path_tsne, exist_ok=True)
os.makedirs(path_pca, exist_ok=True)
os.makedirs(path_kmeans, exist_ok=True)
os.makedirs(path_hc, exist_ok=True)


graphs = load_real_data("Mutagenicity")
train_loader, test_loader, full_loader = prepare_real_data(graphs, 0.8, 32)
labels = next(iter(full_loader)).y

model = Mutag_GCN(graphs.num_node_features, graphs.num_classes)

load_pretrained = False

if load_pretrained:
    model.load_state_dict(torch.load("models/mutagenicity_model.pkl"))
    model.eval()

    with open("models/mutagenicity_activation.txt", 'rb') as file:
        activation_list = pickle.loads(file.read())

else:
    model.apply(weights_init)
    train_mutag(model, train_loader, test_loader, full_loader, 1000, 0.001, path)


# # TSNE

# In[ ]:


# TSNE conversion
tsne_models = []
tsne_data = []
# for layer_num, key in enumerate(activation_list):
#     activation = torch.squeeze(activation_list[key]).detach().numpy()
#     tsne_model = TSNE(n_components=2)
#     d = tsne_model.fit_transform(activation)
#     plot_activation_space(d, labels, "TSNE-Reduced", layer_num, path_tsne, "(coloured by labels)")
    
#     tsne_models.append(tsne_model)
#     tsne_data.append(d)
    
key = "linear"

activation = torch.squeeze(activation_list[key]).detach().numpy()
tsne_model = TSNE(n_components=2)
d = tsne_model.fit_transform(activation)
plot_activation_space(d, labels, "TSNE-Reduced", key, path_tsne, "(coloured by labels)")

tsne_models.append(tsne_model)
tsne_data.append(d)


# # PCA

# In[ ]:


# PCA conversion
pca_models = []
pca_data = []
# for layer_num, key in enumerate(activation_list):
#     activation = torch.squeeze(activation_list[key]).detach().numpy()
#     pca_model = PCA(n_components=2)
#     d = pca_model.fit_transform(activation)
#     plot_activation_space(d, labels, "PCA-Reduced", layer_num, path_pca, "(coloured by labels)")
    
#     pca_models.append(pca_model)
#     pca_data.append(d)
    
activation = torch.squeeze(activation_list[key]).detach().numpy()
pca_model = PCA(n_components=2)
d = pca_model.fit_transform(activation)
plot_activation_space(d, labels, "PCA-Reduced", key, path_pca, "(coloured by labels)")

pca_models.append(pca_model)
pca_data.append(d)


# # Perform KMeans Clustering

# In[ ]:


print(activation_list)
activation_list.pop('pool3')
activation_list.pop('pool4')
activation_list.pop('pool5')
print(activation_list)


# In[ ]:


num_nodes_view = 5
num_expansions = 2

dataset_data = next(iter(full_loader))
edges = dataset_data.edge_index.transpose(0, 1).detach().numpy()
y = dataset_data.y

raw_sample_graphs = []
raw_sample_feat = []
raw_kmeans_models = []
for layer_num, key in enumerate(activation_list):
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    kmeans_model = KMeans(n_clusters=k, random_state=0)
    kmeans_model = kmeans_model.fit(activation)
    pred_labels = kmeans_model.predict(activation)
        
    plot_clusters(tsne_data[layer_num], pred_labels, "KMeans", k, key, path_kmeans, "Raw", "_TSNE", "(TSNE Reduced)")
    plot_clusters(pca_data[layer_num], pred_labels, "KMeans", k, key, path_kmeans, "Raw", "_PCA", "(PCA Reduced)")
    sample_graphs, sample_feat = plot_samples(kmeans_model, activation, y, key, k, "KMeans-Raw", num_nodes_view, edges, num_expansions, path_kmeans, dataset_data)
    
    raw_sample_graphs.append(sample_graphs)
    raw_sample_feat.append(sample_feat)
    raw_kmeans_models.append(kmeans_model)
    


# In[ ]:


tsne_sample_graphs = []
tsne_sample_feat = []
tsne_kmeans_models = []
for layer_num, (key, item) in enumerate(zip(activation_list, tsne_data)):
    kmeans_model = KMeans(n_clusters=k, random_state=0)
    kmeans_model = kmeans_model.fit(item)
    pred_labels = kmeans_model.predict(item)
        
    plot_clusters(item, pred_labels, "KMeans", k, key, path_kmeans, "TSNE")
    sample_graphs, sample_feat = plot_samples(kmeans_model, item, y, key, k, "KMeans-TSNE", num_nodes_view, edges, num_expansions, path_kmeans, dataset_data)
    
    tsne_sample_graphs.append(sample_graphs)
    tsne_sample_feat.append(sample_feat)
    tsne_kmeans_models.append(kmeans_model)
    
pca_sample_graphs = []
pca_sample_feat = []
pca_kmeans_models = []
for key, item in zip(activation_list, pca_data):
    kmeans_model = KMeans(n_clusters=k, random_state=0)
    kmeans_model = kmeans_model.fit(item)
    pred_labels = kmeans_model.predict(item)
        
    plot_clusters(item, pred_labels, "KMeans", k, key, path_kmeans, "PCA")
    sample_graphs, sample_feat = plot_samples(kmeans_model, item, y, key, k, "KMeans-PCA", num_nodes_view, edges, num_expansions, path_kmeans, dataset_data)
    
    pca_sample_graphs.append(sample_graphs)
    pca_sample_feat.append(sample_feat)
    pca_kmeans_models.append(kmeans_model)


# In[ ]:


mutag_heuristics = Mutag_Heuristics()

for layer_num, (key, sample, feat) in enumerate(zip(activation_list, raw_sample_graphs, raw_sample_feat)):
    mutag_heuristics.plot_heuristics_table(sample, feat, key, "KMeans-Raw", path_kmeans)
    
for layer_num, (key, sample, feat) in enumerate(zip(activation_list, tsne_sample_graphs, tsne_sample_feat)):
    mutag_heuristics.plot_heuristics_table(sample, feat, key, "KMeans-TSNE", path_kmeans)
    
for layer_num, (key, sample, feat) in enumerate(zip(activation_list, pca_sample_graphs, pca_sample_feat)):
    mutag_heuristics.plot_heuristics_table(sample, feat, key, "KMeans-PCA", path_kmeans)
    


# # Perform Hierarchical Clustering

# In[ ]:


for layer_num, key in enumerate(activation_list):
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    plot_dendrogram(activation, "Raw", layer_num, path_hc)


# In[ ]:


raw_n_clusters = [3, 3, 18, 20]

raw_sample_graphs = []
raw_sample_feat = []
raw_hc_models = []
for layer_num, (key, n) in enumerate(zip(activation_list, raw_n_clusters)):
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    hc = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='average')
    pred_labels = hc.fit_predict(activation)
    
    d = (activation, pred_labels)
    plot_clusters(tsne_data[layer_num], pred_labels, "HC", n, layer_num, path_hc, "Raw", "_TSNE", "(TSNE Reduced)")
    plot_clusters(pca_data[layer_num], pred_labels, "HC", n, layer_num, path_hc, "Raw", "_PCA", "(PCA Reduced)")
    sample_graphs, sample_feat = plot_samples(hc, d, y, layer_num, n, "HC-Raw", num_nodes_view, edges, num_expansions, path_hc, dataset_data)
    
    raw_sample_graphs.append(sample_graphs)
    raw_sample_feat.append(sample_feat)
    raw_hc_models.append(hc)


# In[ ]:


for layer_num, item in enumerate(tsne_data):
    plot_dendrogram(item, "TSNE", layer_num, path_hc)


# In[ ]:


tsne_n_clusters = [7, 12, 20, 20]

tsne_sample_graphs = []
tsne_sample_feat = []
tsne_hc_models = []
for layer_num, (item, n) in enumerate(zip(tsne_data, tsne_n_clusters)):
    hc = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='average')
    pred_labels = hc.fit_predict(item)
    
    d = (item, pred_labels)
    plot_clusters(item, pred_labels, "HC", n, layer_num, path_hc, "TSNE")
    sample_graphs, sample_feat = plot_samples(hc, d, y, layer_num, n, "HC-TSNE", num_nodes_view, edges, num_expansions, path_hc, dataset_data)
    
    tsne_sample_graphs.append(sample_graphs)
    tsne_sample_feat.append(sample_feat)
    tsne_hc_models.append(hc)


# In[ ]:


for layer_num, item in enumerate(pca_data):
    plot_dendrogram(item, "PCA", layer_num, path_hc)


# In[ ]:


pca_n_clusters = [7, 12, 20, 20]

pca_sample_graphs = []
pca_sample_feat = []
pca_hc_models = []
for layer_num, (item, n) in enumerate(zip(pca_data, pca_n_clusters)):
    hc = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='average')
    pred_labels = hc.fit_predict(item)
    
    d = (item, pred_labels)
    plot_clusters(item, pred_labels, "HC", n, layer_num, path_hc, "PCA")
    sample_graphs, sample_feat = plot_samples(hc, d, y, layer_num, n, "HC-PCA", num_nodes_view, edges, num_expansions, path_hc, dataset_data)
    
    pca_sample_graphs.append(sample_graphs)
    pca_sample_feat.append(sample_feat)
    pca_hc_models.append(hc)


# In[ ]:


for layer_num, (sample, feat) in enumerate(zip(raw_sample_graphs, raw_sample_feat)):
    mutag_heuristics.plot_heuristics_table(sample, feat, layer_num, "HC-RAW", path_kmeans)
    
for layer_num, (sample, feat) in enumerate(zip(tsne_sample_graphs, tsne_sample_feat)):
    mutag_heuristics.plot_heuristics_table(sample, feat, layer_num, "HC-TSNE", path_hc)
    
for layer_num, (sample, feat) in enumerate(zip(pca_sample_graphs, pca_sample_feat)):
    mutag_heuristics.plot_heuristics_table(sample, feat, layer_num, "HC-PCA", path_hc)


# # Activation to Concept to Class

# In[ ]:


classifier_str = "decision_tree"

completeness_scores = []

for i, key in enumerate(activation_list):
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    activation_cls = ActivationClassifier(activation, raw_kmeans_models[i], classifier_str, graphs, dataset_data.y, edges, i, True)
    
    d = ["Kmeans", "Raw", str(activation_cls.get_classifier_accuracy())]
    completeness_scores.append(d)
    activation_cls.plot(path_kmeans, i, k, "Raw")
    
for i, item in enumerate(tsne_data):
    activation_cls = ActivationClassifier(item, tsne_kmeans_models[i], classifier_str, graphs, dataset_data.y, edges, i, True)
    
    d = ["Kmeans", "TSNE-Reduced", str(activation_cls.get_classifier_accuracy())]
    completeness_scores.append(d)
    activation_cls.plot(path_kmeans, i, k, "TSNE")
    
for i, item in enumerate(pca_data):
    activation_cls = ActivationClassifier(item, pca_kmeans_models[i], classifier_str, graphs, dataset_data.y, edges, i, True)
    
    d = ["Kmeans", "PCA-Reduced", str(activation_cls.get_classifier_accuracy())]
    completeness_scores.append(d)
    activation_cls.plot(path_kmeans, i, k, "PCA")
    
plot_completeness_table("Kmeans", "Decision Tree", completeness_scores, path_kmeans)
    


# In[ ]:


classifier_str = "logistic_regression"

completeness_scores = []

for i, key in enumerate(activation_list):
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    activation_cls = ActivationClassifier(activation, raw_kmeans_models[i], classifier_str, graphs, dataset_data.y, edges, i, True)
    
    d = ["Kmeans", "Raw", str(activation_cls.get_classifier_accuracy())]
    completeness_scores.append(d)
    activation_cls.plot(path_kmeans, i, k, "Raw")
    
for i, item in enumerate(tsne_data):
    activation_cls = ActivationClassifier(item, tsne_kmeans_models[i], classifier_str, graphs, dataset_data.y, edges, i, True)
    
    d = ["Kmeans", "TSNE-Reduced", str(activation_cls.get_classifier_accuracy())]
    completeness_scores.append(d)
    activation_cls.plot(path_kmeans, i, k, "TSNE")
    
for i, item in enumerate(pca_data):
    activation_cls = ActivationClassifier(item, pca_kmeans_models[i], classifier_str, graphs, dataset_data.y, edges, i, True)
    
    d = ["Kmeans", "PCA-Reduced", str(activation_cls.get_classifier_accuracy())]
    completeness_scores.append(d)
    activation_cls.plot(path_kmeans, i, k, "PCA")
    
plot_completeness_table("Kmeans", "Logistic Regression", completeness_scores, path_kmeans)
    


# In[ ]:


classifier_str = "decision_tree"

completeness_scores = []

for i, (key, n) in enumerate(zip(activation_list, raw_n_clusters)):
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    activation_cls = ActivationClassifier(activation, raw_hc_models[i], classifier_str, graphs, dataset_data.y, edges, i, True)
    
    d = ["HC", "Raw", str(activation_cls.get_classifier_accuracy())]
    completeness_scores.append(d)
    activation_cls.plot(path_hc, i, n, "Raw")
    
for i, (item, n) in enumerate(zip(tsne_data, tsne_n_clusters)):
    activation_cls = ActivationClassifier(item, tsne_hc_models[i], classifier_str, graphs, dataset_data.y, edges, i, True)
    
    d = ["HC", "TSNE-Reduced", str(activation_cls.get_classifier_accuracy())]
    completeness_scores.append(d)
    activation_cls.plot(path_hc, i, n, "PCA")
    
for i, (item, n) in enumerate(zip(pca_data, pca_n_clusters)):
    activation_cls = ActivationClassifier(item, pca_hc_models[i], classifier_str, graphs, dataset_data.y, edges, i, True)
    
    d = ["HC", "PCA-Reduced", str(activation_cls.get_classifier_accuracy())]
    completeness_scores.append(d)
    activation_cls.plot(path_hc, i, n, "PCA")

plot_completeness_table("HC", "Decision Tree", completeness_scores, path_hc)


# In[ ]:


classifier_str = "logistic_regression"

completeness_scores = []

for i, key in enumerate(activation_list):
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    activation_cls = ActivationClassifier(activation, raw_hc_models[i], classifier_str, graphs, dataset_data.y, edges, i, True)
    
    d = ["HC", "Raw", str(activation_cls.get_classifier_accuracy())]
    completeness_scores.append(d)
    activation_cls.plot(path_hc, i, n, "Raw")
    
for i, item in enumerate(tsne_data):
    activation_cls = ActivationClassifier(item, tsne_hc_models[i], classifier_str, graphs, dataset_data.y, edges, i, True)
    
    d = ["HC", "TSNE-Reduced", str(activation_cls.get_classifier_accuracy())]
    completeness_scores.append(d)
    activation_cls.plot(path_hc, i, n, "TSNE")
    
for i, item in enumerate(pca_data):
    activation_cls = ActivationClassifier(item, pca_hc_models[i], classifier_str, graphs, dataset_data.y, edges, i, True)
    
    d = ["HC", "PCA-Reduced", str(activation_cls.get_classifier_accuracy())]
    completeness_scores.append(d)
    activation_cls.plot(path_hc, i, n, "PCA")

plot_completeness_table("HC", "Logistic Regression", completeness_scores, path_hc)
    


# # Graph Similarity Score

# In[ ]:


graph_scores = []
view = 3
max_nodes = 15

for i, key in enumerate(activation_list):
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    distances = get_node_distances(raw_kmeans_models[i], activation)
    
    for k_idx in range(k):        
        top_indices = np.argsort(distances[:, k_idx])[::][:view]
        top_graphs, _, _ = get_top_subgraphs(top_indices, dataset_data.y, edges, num_expansions)
        
        score = calc_graph_similarity(top_graphs, max_nodes, view)
        print(score)
        
        d = ["KMeans", "Raw", str(i), str(k_idx), str(score)]
        graph_scores.append(d)
        
    plot_samples(raw_kmeans_models[i], activation, y, i, n, "KMeans-Raw", view, edges, num_expansions, path_hc, dataset_data)
        
        
for i, item in enumerate(tsne_data): 
    distances = get_node_distances(tsne_kmeans_models[i], item)
    
    for k_idx in range(k):
        top_indices = np.argsort(distances[:, k_idx])[::][:view]
        top_graphs, _, _ = get_top_subgraphs(top_indices, dataset_data.y, edges, num_expansions)
        
        score = calc_graph_similarity(top_graphs, max_nodes, view)
        print(score)
        
        d = ["KMeans", "TSNE", str(i), str(k_idx), str(score)]
        graph_scores.append(d)
        
    plot_samples(tsne_kmeans_models[i], item, dataset_data.y, i, n, "KMeans-TSNE", view, edges, num_expansions, path_hc, dataset_data)
        

        
for i, item in enumerate(pca_data):
    distances = get_node_distances(pca_kmeans_models[i], item)

    for k_idx in range(k):
        top_indices = np.argsort(distances[:, k_idx])[::][:view]
        top_graphs, _, _ = get_top_subgraphs(top_indices, dataset_data.y, edges, num_expansions)
        
        score = calc_graph_similarity(top_graphs, max_nodes, view)
        print(score)
        
        d = ["KMeans", "PCA", str(i), str(k_idx), str(score)]
        graph_scores.append(d)
        
    plot_samples(pca_kmeans_models[i], item, dataset_data.y, i, n, "KMeans-PCA", view, edges, num_expansions, path_hc, dataset_data)
        
    
plot_graph_similarity_table("HC", graph_scores, path_hc)
    


# In[ ]:


graph_scores = []
view = 3
max_nodes = 15

for i, (key, n) in enumerate(zip(activation_list, raw_n_clusters)):
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    pred_labels = raw_hc_models[i].fit_predict(activation)
    d_item = (activation, pred_labels)
    distances = get_node_distances(raw_hc_models[i], d_item)
    
    for k_idx in range(n):        
        top_indices = np.argsort(distances[k_idx])[::][:view]
        top_graphs, _, _ = get_top_subgraphs(top_indices, dataset_data.y, edges, num_expansions)
        
        score = calc_graph_similarity(top_graphs, max_nodes, view)
        print(score)
        
        d = ["HC", "Raw", str(i), str(k_idx), str(score)]
        graph_scores.append(d)
        
    plot_samples(raw_hc_models[i], d_item, y, i, n, "HC-Raw", view, edges, num_expansions, path_hc, dataset_data)
        
        
for i, (item, n) in enumerate(zip(tsne_data, tsne_n_clusters)): 
    pred_labels = tsne_hc_models[i].fit_predict(item)
    d_item = (item, pred_labels)
    distances = get_node_distances(tsne_hc_models[i], d_item)
    
    for k_idx in range(n):
        top_indices = np.argsort(distances[k_idx])[::][:view]
        top_graphs, _, _ = get_top_subgraphs(top_indices, dataset_data.y, edges, num_expansions)
        
        score = calc_graph_similarity(top_graphs, max_nodes, view)
        print(score)
        
        d = ["HC", "TSNE", str(i), str(k_idx), str(score)]
        graph_scores.append(d)
        
    plot_samples(tsne_hc_models[i], d_item, y, i, n, "HC-TSNE", view, edges, num_expansions, path_hc, dataset_data)
        

        
for i, (item, n) in enumerate(zip(pca_data, pca_n_clusters)):
    pred_labels = pca_hc_models[i].fit_predict(item)
    d_item = (item, pred_labels)
    distances = get_node_distances(pca_hc_models[i], d_item)

    for k_idx in range(n):
        top_indices = np.argsort(distances[k_idx])[::][:view]
        top_graphs, _, _ = get_top_subgraphs(top_indices, dataset_data.y, edges, num_expansions)
        
        score = calc_graph_similarity(top_graphs, max_nodes, view)
        print(score)
        
        d = ["HC", "PCA", str(i), str(k_idx), str(score)]
        graph_scores.append(d)
        
    plot_samples(pca_hc_models[i], d_item, y, i, n, "HC-PCA", view, edges, num_expansions, path_hc, dataset_data)
        
    
plot_graph_similarity_table("HC", graph_scores, path_hc)
    


# In[ ]:


color_map = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

g1 = nx.Graph()
g1.add_edge(0, 1)
g1.add_edge(1, 2)
g1.add_edge(2, 3)
g1.add_edge(3, 4)
g1.add_edge(4, 5)
g1.add_edge(5, 6)
g1.add_edge(6, 7)
g1.add_edge(7, 8)
g1.add_edge(8, 9)
g1.add_edge(9, 10)
g1.add_edge(10, 11)
g1.add_edge(11, 12)
g1.add_edge(12, 13)

nx.draw(g1, node_color=color_map, with_labels=True)
plt.show()


# # GNNExplainer

# In[ ]:


# node_idx = 572

# # convert to edge format
# edges = edge_list.transpose(0, 1).t().contiguous()

# explainer = GNNExplainer2(model, epochs=200, return_type='log_prob', log=True)
# node_feat_mask, edge_mask = explainer.explain_node(node_idx, node_data_x, edges)


# In[ ]:


# ax, G = explainer.visualize_subgraph(node_idx, edges, edge_mask, y=node_data_y, threshold=0.8)
# plt.show()


# In[ ]:


# # 1) get 3 closest nodes of a cluster
# dataset_data = next(iter(full_loader))
# edges = dataset_data.edge_index.transpose(0, 1).detach().numpy()

# def get_top_graphs(graphs, top_indices):    
#     top_graphs = []
#     color_maps = []
#     graph_labels = []
        
#     for idx in top_indices:
#         graph_data = graphs[int(idx)]
#         new_G = nx.Graph()
#         new_G.add_edges_from(graph_data.edge_index.transpose(0, 1).numpy())
#         top_graphs.append(new_G)
        
#         color_map = []
#         for node, attribute in zip(new_G, graph_data.x.numpy()):
#             color_idx = np.argmax(attribute, axis=0)
#             color_map.append(color_idx)
            
#         color_maps.append(color_map)
        
#         graph_labels.append(graph_data.y)
            
#     return top_graphs, color_maps, graph_labels


# def plot_samples(graphs, clustering_model, layer, data, clustering_type, output):
#     num_nodes_view = 5
    
#     fig, axes = plt.subplots(k, num_nodes_view, figsize=(30,30))
#     fig.suptitle(f'Nearest to {clustering_type} Centroid for Layer {layer}', fontsize=40)

#     l = list(range(0, k))

#     for i, ax_list in zip(l, axes):        
#         # get top graphs
#         distances = clustering_model.transform(data)[:, i]
#         top_indices = np.argsort(distances)[::][:num_nodes_view]
#         top_graphs, color_maps, graph_labels = get_top_graphs(graphs, top_indices)
        
#         for ax, new_G, color_map, g_label in zip(ax_list, top_graphs, color_maps, graph_labels):
#             nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax)
#             ax.set_title(f"label {g_label}", fontsize=14)
            
#     plt.savefig(os.path.join(path, f"{output}.png"))
#     plt.show()


# In[ ]:


# class ActivationClassifier:
#     def __init__(self, tsne_data, clustering_model, classifier_type, x, y, edge_list, layer):
#         self.tsne_data = tsne_data
#         self.clustering_model = clustering_model
#         self.classifier_type = classifier_type
#         self.x = x.detach().numpy()
#         self.y = y.detach().numpy()
#         self.edge_list = edge_list
#         self.layer = layer
        
#         self.classifier, self.accuracy = self._train_classifier()
        
        
#     def _train_classifier(self):
#         concepts = []
#         for node_idx in range(len(node_data_x)):
#             concepts.append(self.activation_to_concept(node_idx))
          
#         if self.classifier_type == 'decision_tree':
#             classifier = tree.DecisionTreeClassifier()
#             classifier = classifier.fit(concepts, self.y)
#         elif self.classifier_type == 'linear_regression':
#             classifier = linear_model.LinearRegression()
#             classifier = classifier.fit(concepts, self.y)
        
#         # decision tree accuracy
#         accuracy = classifier.score(concepts, self.y)

#         return classifier, accuracy
    
    
#     def get_classifier_accuracy(self):
#         return self.accuracy
    

#     def _activation_to_cluster(self, node):
#         # apply tsne
#         if isinstance(self.clustering_model, KMeans):
#             activation = tsne_data[self.layer][node]
#             activation = activation.reshape((1, 2))
#             cluster = self.clustering_model.predict(activation)
            
#         elif isinstance(self.clustering_model, AgglomerativeClustering):
#             cluster = np.array([y_hc[node]])

#         return cluster

    
#     def _cluster_to_concept(self, cluster):
#         concept = cluster

#         return concept


#     def activation_to_concept(self, node):
#         # get cluster for node
#         cluster = self._activation_to_cluster(node)

#         # return cluster number as substitute of concept
#         concept = self._cluster_to_concept(cluster)

#         return concept

    
#     def concept_to_class(self, concept):
#         concept = concept.reshape(1, -1)
#         pred = self.classifier.predict(concept)

#         return pred


# In[ ]:


# # get data
# # node_data_x = torch.from_numpy(features).float()
# # node_data_y = torch.from_numpy(labels)
# # edge_list = torch.from_numpy(edges).transpose(0, 1)

# temp = next(iter(full_loader))

# node_data_x = temp.x
# node_data_y = temp.y
# edge_list = temp.edge_index.transpose(0, 1)

# # vars
# chosen_layer = len(activation_list) - 1
# KmeansActivationCls = ActivationClassifier(tsne_data, kmeans_models[chosen_layer], 'decision_tree', node_data_x, node_data_y, edge_list, chosen_layer)

# print("Decision Tree Accuracy: ", KmeansActivationCls.get_decision_tree_accuracy())

# # activation to concept
# node_idx = 224
# concept = KmeansActivationCls.activation_to_concept(node_idx)
# print("Predicted concept is: ", concept)

# cls = KmeansActivationCls.concept_to_class(concept)
# print("Predicted class is: ", cls ," where real one is: ", node_data_y[node_idx])

