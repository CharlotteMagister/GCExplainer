import networkx as nx
import torch as t
import torch_geometric as pyg
import gspan_mining as gspan

from graph_utils import edge_index_to_adj

def networkx_to_pyg_graph(graph):
    edge_index = t.tensor(list(graph.edges())).T
    try:
        return pyg.data.Data(
            x = t.tensor(list(graph.nodes(data = True))),
            edge_index = edge_index
        )
    except:
        # if the networkx nodes have no data, create constant 1 features:
        x = t.ones((len(graph.nodes), 1))
        return pyg.data.Data(x, edge_index)

def pyg_to_networkx_graph(graph):
    return pyg.utils.to_networkx(graph)

def adjacency_matrix_to_pyg_graph(A):
    return pyg.data.Data(
        x = t.arange(A.shape[0]),
        edge_index = t.tensor(A.nonzero())
    )

def pyg_to_adjacency_matrix(graph):
    return edge_index_to_adj(graph.edge_index)

def subgraph(graph, node_set):
    """Returns the subgraph of graph induced by node_set.
    """
    return Graph(graph.to_networkx().subgraph(node_set))

class Graph:
    def __init__(self, graph_data):
        ## figure out which format graph data is in:
        self.format = "unknown"
        if isinstance(graph_data, nx.Graph):
            self.format = "networkx"
            self.pyg_data = networkx_to_pyg_graph(graph_data)
        elif isinstance(graph_data, pyg.data.Data):
            self.format = "pyg"
            self.pyg_data = graph_data
        elif isinstance(graph_data, t.Tensor):
            self.format = "adjacency_matrix"
            self.pyg_data = adjacency_matrix_to_pyg_graph(graph_data)
        elif isinstance(graph_data, tuple):
            # assume it's a tuple of (node_labels, edges)
            self.format = "labels_and_edges"
            edge_index = t.tensor(graph_data[1])
            if edge_index.shape[0] != 2:
                if edge_index.shape[1] == 2:
                    print("WARNING: edge_index shape wrong way around")
                    edge_index = edge_index.T
                else:
                    raise Exception(f"Edge index shape {edge_index.shape} not understood, must be [2, X]")
            self.pyg_data = pyg.data.Data(
                x = t.tensor(graph_data[0]),
                edge_index = edge_index
            )
            
        else:
            raise ValueError("Unknown graph format")
        
        assert self.pyg_data.edge_index.shape[0] == 2, f"{self.pyg_data.edge_index.shape} is not a valid edge index shape, must be [2, X]"
    
    def to_networkx(self):
        return pyg_to_networkx_graph(self.pyg_data)
    def to_adjacency_matrix(self):
        return pyg_to_adjacency_matrix(self.pyg_data)
    def to_pyg(self):
        return self.pyg_data
    def to_gspan(self):
        # this format is described here: https://github.com/betterenvi/gSpan/tree/master/graphdata
        s = "t # 0\n"
        for i, node in enumerate(self.to_networkx().nodes(data = True)):
            node_label = 1 # node[1]["label"]
            # NB: node_labels are currently not used;
            #     I think they need to be discrete
            s += "v {} {}\n".format(i, node_label)
        for edge in self.to_networkx().edges(data = True):
            edge_label = 1 # edge[2]["label"]
            s += "e {} {} {}\n".format(edge[0], edge[1], edge_label)
        s += "t # -1\n"
        # ^^ required to avoid a bug according to the gSpan docs ... :) :) :)
        return s
    
    def draw(self):
        nx.draw(self.to_networkx())
    
    def add_edge(self, node_a, node_b):
        self.pyg_data.edge_index = t.cat(
            (self.pyg_data.edge_index,
             t.tensor([[node_a], [node_b]])),
            dim = 1
        )
    
    def disjoint_union(self, graph_b):
        nx_union = nx.disjoint_union(self.to_networkx(), graph_b.to_networkx())
        return Graph(networkx_to_pyg_graph(nx_union))
    
    def num_nodes(self):
        return self.pyg_data.x.shape[0]
    
    def subgraph(self, node_set):
        return subgraph(self, node_set)
        

