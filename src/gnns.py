import torch as t
import torch_geometric as pyg

class GAE(pyg.nn.conv.MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        nonlinearity = t.nn.ReLU,
        gcn_layers = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gcn1 = pyg.nn.conv.GCNConv(in_channels, out_channels)
        self.gcn = pyg.nn.conv.GCNConv(out_channels, out_channels)
        self.nonlinearity = nonlinearity()
        self.gcn_layers = gcn_layers
    
    def forward(self, x, edge_index):
        z = self.gcn1(x, edge_index)
        for _ in range(self.gcn_layers - 1):
            z = self.gcn(z, edge_index)
        # computing the outer product is probably pretty inefficient;
        # could change to something sparse probably
        adj_pred = self.nonlinearity(t.einsum("ie,je->ij", z, z))
        return adj_pred

