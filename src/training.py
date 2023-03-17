import torch as t
import torch_geometric as pyg
from graph_utils import edge_index_to_adj
from tqdm import tqdm


def train_edge_predictor(
    model, graphs, epochs, adj_loss=t.nn.MSELoss(), lr=0.01
):
    losses = []
    opt = t.optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
        for graph in graphs:
            model.train()
            edges_predicted = model(graph.x, graph.edge_index)
            loss = adj_loss(
                edges_predicted, edge_index_to_adj(graph.edge_index).float()
            )
            losses.append(loss.detach().numpy())
            loss.backward()
            opt.step()
    return losses


def train_k_means_gnn(model, graphs, epochs, lr=0.01, gamma=0.001):
    model.train()
    losses = []
    opt = t.optim.Adam(model.parameters(), lr=lr)
    # for epoch in tqdm(range(epochs)):
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
