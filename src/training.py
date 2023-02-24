import torch as t
import torch_geometric as pyg
from graph_utils import edge_index_to_adj
from tqdm import tqdm

def train_edge_predictor(
    model,
    graphs,
    epochs,
    adj_loss=t.nn.MSELoss(),
    lr=0.01
):
    losses = []
    opt = t.optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
        for graph in graphs:
            model.train()
            edges_predicted = model(graph.x, graph.edge_index)
            loss = adj_loss(edges_predicted, edge_index_to_adj(graph.edge_index).float())
            losses.append(loss.detach().numpy())
            loss.backward()
            opt.step()
    return losses