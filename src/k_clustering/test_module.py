import unittest
import utilities
import networkx as nx
import torch_geometric
import sklearn
import numpy as np
import activation_classifier
import torch
import heuristics


class Utilities_Tests(unittest.TestCase):

    def test_load_syn_data(self):
        """Test synthetic dataset loaded as expected."""
        G, role_ids = utilities.load_syn_data("BA_Shapes")
        self.assertIsInstance(role_ids, np.ndarray)
        self.assertIsInstance(G, nx.Graph)

    def test_load_real_data(self):
        """Test real dataset loaded as expected."""
        graphs = utilities.load_real_data("Mutagenicity")
        self.assertIsInstance(graphs, torch_geometric.datasets.TUDataset)

    def test_prepare_syn_data(self):
        """Test expected edge list."""
        G, role_ids = utilities.load_syn_data("BA_Shapes")
        data = utilities.prepare_syn_data(G, role_ids, 0.8)

        self.assertIsNotNone(data['x'])
        self.assertIsNotNone(data['y'])
        self.assertIsNotNone(data['edges'])
        self.assertIsNotNone(data['edge_list'])
        self.assertIsNotNone(data['train_mask'])
        self.assertIsNotNone(data['test_mask'])

    def test_prepare_syn_data2(self):
        """Test synthetic dataset transformed as expected."""
        G, role_ids = utilities.load_syn_data("BA_Shapes")
        data = utilities.prepare_syn_data(G, role_ids, 0.8, if_adj=True)

        self.assertIsNotNone(data['x'])
        self.assertIsNotNone(data['y'])
        self.assertIsNotNone(data['edges'])
        self.assertIsNotNone(data['edge_list'])
        self.assertIsNotNone(data['train_mask'])
        self.assertIsNotNone(data['test_mask'])

    def test_prepare_real_data(self):
        """Test real dataset transformed as expected."""
        graphs = utilities.load_real_data("Mutagenicity")
        train_loader, test_loader, full_loader, small_loader = utilities.prepare_real_data(graphs, 0.8, 20, "Mutagenicity")

        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(test_loader)
        self.assertIsNotNone(full_loader)
        self.assertIsNotNone(small_loader)
        self.assertIsInstance(train_loader, torch_geometric.data.DataLoader)

    def test_get_top_subgraphs(self):
        """Test top subgraphs retrieved appropriately."""
        y = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        edges = [(1, 2), (2, 3), (4, 5), (5, 6), (7, 8), (8, 9)]
        top_indices = [1, 4, 7]
        num_expansions = 2

        graphs, color_maps, labels, _ = utilities.get_top_subgraphs(top_indices, y, edges, num_expansions)
        self.assertEqual(len(graphs), 3)
        self.assertEqual(len(color_maps[0]), 3)
        self.assertEqual(labels, [2, 5, 8])

    def test_get_node_distances(self):
        """Test node distances sorted appropriately."""
        points = np.array([[1, 1], [2, 2], [10, 10], [11, 11]])
        kmeans = sklearn.cluster.KMeans(n_clusters=2, random_state=0).fit(points)

        res_sorted = utilities.get_node_distances(kmeans, points)
        self.assertTrue(res_sorted[0][0] <= res_sorted[0][1])

    def test_calc_graph_similarity(self):
        """Test calculation of graph similarity as expected."""
        g1 = nx.Graph()
        g1.add_edge(1, 2)

        g2 = nx.Graph()
        g2.add_edge(1, 2)

        graphs = [g1, g2]

        self.assertEqual(utilities.calc_graph_similarity(graphs, 15, 2), 0)

    def test_prepare_output_paths(self):
        """Test figure output paths as expected."""
        paths = utilities.prepare_output_paths("dataset", 10)

        self.assertEqual(paths['base'], "output/dataset/")
        self.assertEqual(paths['KMeans'], "output/dataset/10_KMeans")
        self.assertEqual(paths['UMAP'], "output/dataset/UMAP")


class Activation_Classifier_Tests(unittest.TestCase):

    def test_activation_classifier_DT(self):
        """"Test implementation of Activation Classifier using Decision Tree."""
        y = torch.from_numpy(np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1]))
        x = torch.from_numpy(np.array([[1, 1], [2, 2], [10, 10], [11, 11], [3, 3], [0,0], [12, 12], [12, 12], [13, 13], [12, 12]]))
        train_mask = [True, True, True, True, True, True, True, True, False, False]
        test_mask = [False, False, False, False, False, False, False, False, True, True]
        kmeans = sklearn.cluster.KMeans(n_clusters=2, random_state=0).fit(x)
        cls = activation_classifier.ActivationClassifier(x, kmeans, 'decision_tree', x, y, train_mask, test_mask)

        self.assertEqual(cls.get_classifier_accuracy(), 1)

    def test_activation_classifier_LR(self):
        """Test implementation of Activation Classifier using Logistic Regression."""
        y = torch.from_numpy(np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1]))
        x = torch.from_numpy(np.array([[1, 1], [2, 2], [10, 10], [11, 11], [3, 3], [0,0], [12, 12], [12, 12], [13, 13], [12, 12]]))
        train_mask = [True, True, True, True, True, True, True, True, False, False]
        test_mask = [False, False, False, False, False, False, False, False, True, True]
        kmeans = sklearn.cluster.KMeans(n_clusters=2, random_state=0).fit(x)
        cls = activation_classifier.ActivationClassifier(x, kmeans, 'logistic_regression', x, y, train_mask, test_mask)

        self.assertEqual(cls.get_classifier_accuracy(), 1)


class Heuristic_Tests(unittest.TestCase):

    def test_BA_Shapes_heuristic1(self):
        """Assert heuristic for top node in house structure."""
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 3)
        g.add_edge(2, 4)
        g.add_edge(3, 5)
        g.add_edge(4, 5)

        h = heuristics.BA_Shapes_Heuristics()
        h_num, h_str, if_true = h.eval_heuristic1(g, 1)
        self.assertEqual(h_num, 1)
        self.assertEqual(h_str, "Top Node")
        self.assertEqual(if_true, True)

        h_num, h_str, if_true = h.eval_heuristic1(g, 2)
        self.assertEqual(h_num, 1)
        self.assertEqual(h_str, "Top Node")
        self.assertEqual(if_true, False)

    def test_BA_Shapes_heuristic2(self):
        """Assert heuristic for middle node in house structure."""
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 3)
        g.add_edge(2, 4)
        g.add_edge(3, 5)
        g.add_edge(4, 5)

        h = heuristics.BA_Shapes_Heuristics()
        h_num, h_str, if_true = h.eval_heuristic2(g, 2)
        self.assertEqual(h_num, 2)
        self.assertEqual(h_str, "Middle Node")
        self.assertEqual(if_true, True)

        h_num, h_str, if_true = h.eval_heuristic2(g, 4)
        self.assertEqual(h_num, 2)
        self.assertEqual(h_str, "Middle Node")
        self.assertEqual(if_true, False)

    def test_BA_Shapes_heuristic3(self):
        """Assert heuristic of bottom node in house structure."""
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 3)
        g.add_edge(2, 4)
        g.add_edge(3, 5)
        g.add_edge(4, 5)

        h = heuristics.BA_Shapes_Heuristics()
        h_num, h_str, if_true = h.eval_heuristic3(g, 4)
        self.assertEqual(h_num, 3)
        self.assertEqual(h_str, "Bottom Node")
        self.assertEqual(if_true, True)

        h_num, h_str, if_true = h.eval_heuristic3(g, 1)
        self.assertEqual(h_num, 3)
        self.assertEqual(h_str, "Bottom Node")
        self.assertEqual(if_true, False)

    def test_BA_Shapes_heuristic4(self):
        """Assert heuristic of house structure with top node and middle arm."""
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 3)
        g.add_edge(2, 4)
        g.add_edge(3, 5)
        g.add_edge(4, 5)
        g.add_edge(2, 6)

        h = heuristics.BA_Shapes_Heuristics()
        h_num, h_str, if_true = h.eval_heuristic4(g, 1)
        self.assertEqual(h_num, 4)
        self.assertEqual(h_str, "Top Node and Middle Arm")
        self.assertEqual(if_true, True)

        h_num, h_str, if_true = h.eval_heuristic4(g, 2)
        self.assertEqual(h_num, 4)
        self.assertEqual(h_str, "Top Node and Middle Arm")
        self.assertEqual(if_true, False)

    def test_BA_Shapes_heuristic5(self):
        """Assert heuristic of house structure with middle node and edge on close side."""
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 3)
        g.add_edge(2, 4)
        g.add_edge(3, 5)
        g.add_edge(4, 5)
        g.add_edge(2, 6)

        h = heuristics.BA_Shapes_Heuristics()
        h_num, h_str, if_true = h.eval_heuristic5(g, 2)
        self.assertEqual(h_num, 5)
        self.assertEqual(h_str, "Middle Node and Middle Arm Close")
        self.assertEqual(if_true, True)

        h_num, h_str, if_true = h.eval_heuristic5(g, 3)
        self.assertEqual(h_num, 5)
        self.assertEqual(h_str, "Middle Node and Middle Arm Close")
        self.assertEqual(if_true, False)

    def test_BA_Shapes_heuristic6(self):
        """Assert heuristic of house structure with middle node and edge on far side."""
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 3)
        g.add_edge(2, 4)
        g.add_edge(3, 5)
        g.add_edge(4, 5)
        g.add_edge(2, 6)

        h = heuristics.BA_Shapes_Heuristics()
        h_num, h_str, if_true = h.eval_heuristic6(g, 3)
        self.assertEqual(h_num, 6)
        self.assertEqual(h_str, "Middle Node and Middle Arm Far")
        self.assertEqual(if_true, True)

        h_num, h_str, if_true = h.eval_heuristic6(g, 1)
        self.assertEqual(h_num, 6)
        self.assertEqual(h_str, "Middle Node and Middle Arm Far")
        self.assertEqual(if_true, False)

    def test_BA_Shapes_heuristic7(self):
        """Assert heuristic of house structure with bottom node and edge on close side."""
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 3)
        g.add_edge(2, 4)
        g.add_edge(3, 5)
        g.add_edge(4, 5)
        g.add_edge(2, 6)

        h = heuristics.BA_Shapes_Heuristics()
        h_num, h_str, if_true = h.eval_heuristic7(g, 4)
        self.assertEqual(h_num, 7)
        self.assertEqual(h_str, "Bottom Node and Middle Arm Close")
        self.assertEqual(if_true, True)

        h_num, h_str, if_true = h.eval_heuristic7(g, 5)
        self.assertEqual(h_num, 7)
        self.assertEqual(h_str, "Bottom Node and Middle Arm Close")
        self.assertEqual(if_true, False)

    def test_BA_Shapes_heuristic8(self):
        """Assert heuristic of house structure with bottom node and edge on far side."""
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 3)
        g.add_edge(2, 4)
        g.add_edge(3, 5)
        g.add_edge(4, 5)
        g.add_edge(2, 6)

        h = heuristics.BA_Shapes_Heuristics()
        h_num, h_str, if_true = h.eval_heuristic8(g, 5)
        self.assertEqual(h_num, 8)
        self.assertEqual(h_str, "Bottom Node and Middle Arm Far")
        self.assertEqual(if_true, True)

        h_num, h_str, if_true = h.eval_heuristic8(g, 4)
        self.assertEqual(h_num, 8)
        self.assertEqual(h_str, "Bottom Node and Middle Arm Far")
        self.assertEqual(if_true, False)


if __name__ == '__main__':
    unittest.main()
