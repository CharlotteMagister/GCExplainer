import torch as t
import torch_geometric as pyg
from gcf import GraphConceptFinder
from training import train_k_means_gnn
from sklearn import cluster
import numpy as np


class k_means_gnn(pyg.nn.conv.MessagePassing):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        nonlinearity=t.nn.ReLU,
        num_layers=2,
        k=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        layers = [
            pyg.nn.conv.GCNConv(in_channels, hidden_channels),
        ]

        for _ in range(num_layers - 2):
            layers.append(
                pyg.nn.conv.GCNConv(hidden_channels, hidden_channels)
            )

        layers.append(pyg.nn.conv.GCNConv(hidden_channels, out_channels))

        self.layers = t.nn.ModuleList(layers)

        self.nonlinearity = nonlinearity()
        self.k = k
        self.centres = t.rand((k, out_channels), requires_grad=False)
        # Updated when caclulate centres update called - is the counts BEFORE the
        # update is applied
        self.cluster_counts = None

    def forward(self, x, edge_index):
        # produce embeddings for each node
        for l in self.layers[:-1]:
            x = l(x, edge_index)
            x = self.nonlinearity(x)
        x = self.layers[-1](x, edge_index)
        return x

    def assign_to_clusters(self, embeddings):
        cluster_distances = t.norm(
            self.centres - t.unsqueeze(embeddings, 1), dim=2
        )
        cluster_assignments = t.argmin(cluster_distances, dim=1)
        return cluster_assignments

    def cluster_distances(self, assignments, embeddings):
        gather_indices = t.repeat_interleave(
            t.unsqueeze(assignments, 1), self.out_channels, dim=1
        )
        dist_to_assigned_centre = (
            t.gather(self.centres, 0, gather_indices) - embeddings
        )
        return dist_to_assigned_centre

    def clustering_loss(self, distances):
        k_means_residuals = t.norm(distances, dim=1)
        return t.sum(k_means_residuals)

    def calculate_centres_update(self, assignments, distances, gamma=0.01):
        num_nodes = assignments.shape[0]
        cluster_counts = t.scatter_add(
            t.zeros(self.k, requires_grad=False),
            0,
            assignments,
            t.ones(num_nodes, requires_grad=False),
        )
        self.cluster_counts = cluster_counts

        scatter_indices = t.repeat_interleave(
            t.unsqueeze(assignments, 1), self.out_channels, dim=1
        )
        sum_of_distances = t.scatter_add(
            t.zeros(self.centres.shape, requires_grad=False),
            0,
            scatter_indices,
            distances,
        )

        mask = cluster_counts != 0
        update = -(
            gamma
            / t.unsqueeze(cluster_counts[mask], dim=1)
            * sum_of_distances[mask]
        )
        return mask, update

    def perform_centres_update(self, mask, update):
        self.centres[mask] += update

    def updates_kmeans_centres(self, graphs, k):
        embeds = []
        with t.no_grad():
            for graph in graphs:
                embeds += [self.forward(graph.x, graph.edge_index).numpy()]
        embeds = np.concatenate(embeds)
        centres, assignments, inertia = cluster.k_means(
            embeds, k, n_init="auto"
        )
        self.centres = t.tensor(centres)
        self.k = k
        return centres, assignments


class KmeansConceptFinder(GraphConceptFinder):
    def __init__(
        self,
        k,
        hidden_channels=32,
        out_channels=16,
        depth=2,
        initial_epochs=200,
        initial_lr=0.01,
        initial_gamma=0.01,
        fine_epochs=100,
        fine_lr=0.001,
        fine_gamma=0.001,
    ):
        self.k = k
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth

        self.initial_epochs = initial_epochs
        self.initial_lr = initial_lr
        self.initial_gamma = initial_gamma

        self.fine_epochs = fine_epochs
        self.fine_lr = fine_lr
        self.fine_gamma = fine_gamma

    def find_concepts(self, graph, file):
        graph_data = graph.pyg_data
        in_channels = graph_data.x.shape[1]

        k_means_model = k_means_gnn(
            in_channels,
            self.hidden_channels,
            self.out_channels,
            num_layers=self.depth,
            k=self.k,
        )

        train_k_means_gnn(
            k_means_model,
            [graph_data],
            self.initial_epochs,
            lr=self.initial_lr,
            gamma=self.initial_gamma,
        )

        k_means_model.updates_kmeans_centres([graph_data], self.k)

        train_k_means_gnn(
            k_means_model,
            [graph_data],
            self.fine_epochs,
            lr=self.fine_lr,
            gamma=self.fine_gamma,
        )

        with t.no_grad():
            embeds = k_means_model.forward(graph_data.x, graph_data.edge_index)
            clusters = (
                k_means_model.assign_to_clusters(embeds).detach().numpy()
            )

        concepts = [
            list((clusters == c).nonzero()[0]) for c in np.unique(clusters)
        ]
        self.concepts = concepts
        return concepts
