import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import networkx as nx

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
import sklearn.metrics as metrics
import seaborn as sn

class ActivationClassifier():
    def __init__(self, data, clustering_model, classifier_type, x, y, edge_list, layer, if_graph_class=False):
        self.data = data
        self.clustering_model = clustering_model
        self.classifier_type = classifier_type

        if if_graph_class:
            self.x = x
        else:
            self.x = x.detach().numpy()

        self.y = y.detach().numpy()
        self.edge_list = edge_list
        self.layer = layer
        self.if_graph_class = if_graph_class

        if isinstance(self.clustering_model, AgglomerativeClustering):
            self.y_hc = self.clustering_model.fit_predict(self.data)

        self.classifier, self.accuracy = self._train_classifier()


    def _train_classifier(self):
        self.concepts = []

        for node_idx in range(len(self.x)):
            self.concepts.append([self.activation_to_concept(node_idx)])

        if self.classifier_type == 'decision_tree':
            cls = tree.DecisionTreeClassifier()
            cls = cls.fit(self.concepts, self.y)
        elif self.classifier_type == 'logistic_regression':
            cls = linear_model.LogisticRegression()
            cls = cls.fit(self.concepts, self.y)

        # decision tree accuracy
        accuracy = cls.score(self.concepts, self.y)

        return cls, accuracy


    def get_classifier_accuracy(self):
        return self.accuracy


    def _activation_to_cluster(self, node):
        # apply tsne
        if isinstance(self.clustering_model, KMeans):
            cluster = self.clustering_model.predict(self.data)
            cluster = cluster[node]

        elif isinstance(self.clustering_model, AgglomerativeClustering):
            cluster = np.array(self.y_hc[node])

        return cluster


    def _cluster_to_concept(self, cluster):
        concept = cluster

        return concept


    def activation_to_concept(self, node):
        # get cluster for node
        cluster = self._activation_to_cluster(node)

        # return cluster number as substitute of concept
        concept = self._cluster_to_concept(cluster)

        return concept


    def concept_to_class(self, concept):
        concept = concept.reshape(1, -1)
        pred = self.classifier.predict(concept)

        return pred


    def plot(self, path, layer_num, k, reduction_type):
        if self.classifier_type == 'decision_tree':
            fig, ax = plt.subplots(figsize=(20, 20))
            tree.plot_tree(self.classifier, ax=ax)
            fig.suptitle(f"Decision Tree for {reduction_type} Model")

        elif self.classifier_type == 'logistic_regression':
            fig, ax = plt.subplots(figsize=(6, 6))
            pred = self.classifier.predict(self.concepts)
            ls = np.unique(self.y)
            confusion_matrix = metrics.confusion_matrix(self.y, pred, labels=ls)
            cm = pd.DataFrame(confusion_matrix, index=ls, columns=ls)

            ax = sn.heatmap(cm, annot=True, cmap="YlGnBu", ax=ax, fmt='g', )
            fig.suptitle(f"Confusion Matrix of Logistic Regression on for {reduction_type} Model")
            ax.set_xlabel("Target Class")
            ax.set_ylabel("Predicted Class")

        plt.savefig(os.path.join(path, f"{k}k_{layer_num}layer_{reduction_type}_{self.classifier_type}.png"))
        plt.show()
