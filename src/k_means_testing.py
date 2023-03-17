import graph_gen as gg
import graph
from k_means_gnn import k_means_gnn, KmeansConceptFinder
import torch as t
import numpy as np
from sklearn import cluster

bh_adj_matrix = gg.bhshapes(300, 80)
bh_Graph = graph.Graph(bh_adj_matrix)
graph_data = bh_Graph.pyg_data

concept_finder = KmeansConceptFinder(4)
concept_finder.find_concepts(bh_Graph, "testsav")
print(concept_finder.concepts)
exit(1)

k_means_model = k_means_gnn(1, 32, 16, k=5, num_layers=3)

print("CENTRES:")
print(k_means_model.centres)
print(k_means_model.centres.shape)
print()

embeds = k_means_model.forward(graph_data.x, graph_data.edge_index)
print("EMBEDS:")
print(embeds[:2])
print(embeds.shape)
print()

clusters = k_means_model.assign_to_clusters(embeds)
print("ASSIGNMENTS:")
print(clusters)
print(clusters.shape)
print()

centre_distances = k_means_model.cluster_distances(clusters, embeds)

cluster_loss = k_means_model.clustering_loss(centre_distances)
print("CLUSTER LOSS")
print(cluster_loss)
print()

# for _ in range(10):

#     mask, update = k_means_model.calculate_centres_update(
#         clusters, centre_distances
#     )
#     k_means_model.perform_centres_update(mask, update)
#     print("UPDATED CENTRES")
#     print(k_means_model.centres)
#     print(k_means_model.centres.shape)
#     print()

#     clusters = k_means_model.assign_to_clusters(embeds)
#     # print("UPDATED ASSIGNMENTS")
#     # print(clusters)
#     # print(clusters.shape)
#     # print()

#     centre_distances = k_means_model.cluster_distances(clusters, embeds)

#     cluster_loss = k_means_model.clustering_loss(centre_distances)
#     print("NEW CLUSTER LOSS:")
#     print(cluster_loss)


def train_k_means_gnn(model, graphs, epochs, lr=0.01, gamma=0.001):
    model.train()
    losses = []
    opt = t.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for graph in graphs:
            opt.zero_grad()

            embeds = model(graph.x, graph.edge_index)

            # Update assignments
            clusters = model.assign_to_clusters(embeds)
            distances = model.cluster_distances(clusters, embeds)

            # Calculate centre update
            with t.no_grad():
                mask, update = model.calculate_centres_update(
                    clusters, distances
                )

            # Update GNN layers
            loss = model.clustering_loss(distances)
            losses.append(loss.detach().numpy())
            loss.backward()
            opt.step()

            # Update centres
            model.perform_centres_update(mask, update)

            print(
                f"Epoch {epoch}: loss = {loss}, cluster_counts={model.cluster_counts}"
            )

    return losses


train_k_means_gnn(k_means_model, [graph_data], 500, lr=0.01, gamma=0.01)
print("PRE UPDATE:")
print(k_means_model.centres)
print()


def updates_kmeans_centres(model, graphs, k):
    embeds = []
    with t.no_grad():
        for graph in graphs:
            embeds += [model(graph.x, graph.edge_index).numpy()]
    embeds = np.concatenate(embeds)
    centres, assignments, inertia = cluster.k_means(embeds, k, n_init="auto")
    model.centres = t.tensor(centres)
    model.k = k
    return centres, assignments


new_centres, new_assignments = updates_kmeans_centres(
    k_means_model, [graph_data], 6
)
print("AFTER UPDATE")
print(new_centres)
print(np.unique(new_assignments, return_counts=True))
print()

print("Fine tuning:")
print(k_means_model.centres)
train_k_means_gnn(k_means_model, [graph_data], 100, lr=0.001, gamma=0.001)

with t.no_grad():
    embeds = k_means_model.forward(graph_data.x, graph_data.edge_index)
    clusters = k_means_model.assign_to_clusters(embeds)
    np.save("training_runs/testing.embeds", embeds.numpy())
    np.save("training_runs/testing.clusters", clusters.numpy())
    np.save("training_runs/testing.centres", k_means_model.centres.numpy())
