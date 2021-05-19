#!/usr/bin/env python
# coding: utf-8

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
from torch_geometric.utils import add_self_loops, degree

from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min

from torch_geometric.nn import GCNConv

from sklearn import tree, linear_model

import scipy.cluster.hierarchy as hierarchy
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestCentroid
import umap

from torch_geometric.nn import GNNExplainer

from utilities import *
from activation_classifier import *

set_rc_params()

# general parameters
dataset_name = "BA_Houses"

model_type = BA_Shapes_GCN
load_pretrained = True

# hyperparameters
k = 10

# other parameters
train_test_split = 0.8
num_hidden_units = 20
num_classes = 4

epochs = 3000
lr = 0.001


paths = prepare_output_paths(dataset_name, k)

G, labels = load_syn_data(dataset_name)
data = prepare_syn_data(G, labels, train_test_split)
model = model_type(data["x"].shape[1], num_hidden_units, num_classes, "BA-Houses")

if load_pretrained:
    print("Loading pretrained model...")
    model.load_state_dict(torch.load(os.path.join(paths['base'], "model.pkl")))
    model.eval()
    
    with open(os.path.join(paths['base'], "activations.txt"), 'rb') as file:
        activation_list = pickle.loads(file.read())
        
else:
    model.apply(weights_init)
    train(model, data, epochs, lr, paths['base'])

activation_list = {'conv3': activation_list['conv3']}

# TSNE conversion
tsne_models = []
tsne_data = []
for layer_num, key in enumerate(activation_list):
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    tsne_model = TSNE(n_components=2)
    d = tsne_model.fit_transform(activation)
    plot_activation_space(d, labels, "TSNE-Reduced", layer_num, paths['TSNE'], "(coloured by labels)")
    
    tsne_models.append(tsne_model)
    tsne_data.append(d)


# PCA conversion
pca_models = []
pca_data = []
for layer_num, key in enumerate(activation_list):
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    pca_model = PCA(n_components=2)
    d = pca_model.fit_transform(activation)
    plot_activation_space(d, labels, "PCA-Reduced", layer_num, paths['PCA'], "(coloured by labels)")

    pca_models.append(pca_model)
    pca_data.append(d)


num_nodes_view = 5
num_expansions = 2
edges = data['edge_list'].numpy()
print(edges)

raw_sample_graphs = []
raw_kmeans_models = []
for layer_num, key in enumerate(activation_list):
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    kmeans_model = KMeans(n_clusters=k, random_state=0)
    kmeans_model = kmeans_model.fit(activation)
    pred_labels = kmeans_model.predict(activation)
        
    plot_clusters(tsne_data[layer_num], pred_labels, "KMeans", k, layer_num, paths['KMeans'], "Raw", "_TSNE", "(TSNE Reduced)")
    plot_clusters(pca_data[layer_num], pred_labels, "KMeans", k, layer_num, paths['KMeans'], "Raw", "_PCA", "(PCA Reduced)")
    sample_graphs, sample_feat = plot_samples(kmeans_model, activation, data["y"], layer_num, k, "KMeans-Raw", num_nodes_view, edges, num_expansions, paths['KMeans'])

    raw_sample_graphs.append(sample_graphs)
    raw_kmeans_models.append(kmeans_model)

classifier_str = "decision_tree"

completeness_scores = []

for i, key in enumerate(activation_list):
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    activation_cls = ActivationClassifier(activation, raw_kmeans_models[i], classifier_str, data["x"], data["y"], data["test_mask"], data["test_mask"], data["edges"], i)
    
    d = ["Kmeans", "Raw", str(activation_cls.get_classifier_accuracy())]
    completeness_scores.append(d)
    print(d)
#     activation_cls.plot(paths['KMeans'], i, k, "Raw")


# x = data["x"]
# edges = data["edges"]
# y = data["y"]
# train_mask = data["train_mask"]
# test_mask = data["test_mask"]
# print(test(model, x, y, edges, test_mask))

# activation = torch.squeeze(activation_list['conv2']).detach().numpy()
# kmeans_model = KMeans(n_clusters=9, random_state=0)
# kmeans_model = kmeans_model.fit(activation)

# testing = ActivationClassifier(activation, kmeans_model, classifier_str, data["x"], data["y"], data["x"], data["y"], data["edges"], i)
# print(testing.accuracy)