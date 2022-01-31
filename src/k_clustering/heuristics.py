import networkx as nx
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Mutag_Heuristics():
    def __init__(self):
        self.num_heuristics = 4

        self.ring = self._build_ring_heuristic()
        self.no2 = self._build_no2_heuristic()

        self.heuristic_functions = []
        self.heuristic_functions.append(self.eval_heuristic1)
        self.heuristic_functions.append(self.eval_heuristic2)

    def _build_ring_heuristic(self):
        """Simple Ring"""
        G = nx.Graph()
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 4)
        G.add_edge(4, 5)
        G.add_edge(5, 6)
        G.add_edge(6, 1)

        return G

    def _build_no2_heuristic(self):
        G = nx.Graph()
        G.add_edge(1, 2)
        G.add_edge(1, 3)
        return G

    def eval_heuristic1(self, G2, node, x):
        """Simple Circle"""
        return 1, "Circle", nx.is_isomorphic(self.ring, G2)

    def eval_heuristic2(self, G2, node, x):
        """NO2"""

        ret = False
        if nx.is_isomorphic(self.no2, G2):
            n = 0
            o = 0

            for feat in x:
                if np.argmax(feat, axis=0) == 4:
                    n += 1
                elif np.argmax(feat, axis=0) == 4:
                    o += 1

            if n == 1 and o == 2:
                ret = True

        return 2, "NO2", ret

    def eval(self, sample_graphs, sample_x):
        results = []
        for i, ((G, node), x) in enumerate(zip(sample_graphs, sample_x)):
            ifFound = False
            for heuristic in self.heuristic_functions:
                ret = heuristic(G, node, x)

                if ret[2]:
                    results.append([str(i), str(ret[0]), ret[1]])
                    ifFound = True
                    break

            if not ifFound:
                results.append([str(i), "No Match", "-"])

        return results

    def plot_heuristics_table(self, sample_graphs, sample_feats, layer_num, clustering_type, reduction_type, path):
        fig, ax = plt.subplots(figsize=(10, 10))
        data = self.eval(sample_graphs, sample_feats)
        headings = ["Cluster", "Heuristic", "Descritpion"]

        ax.set_title(f"Concepts Identified per {clustering_type} Cluster in {reduction_type} Activation Space of Layer {layer_num}")
        ax.axis('off')
        ax.table(cellText=data, colLabels=headings, loc="center", rowLoc="center", cellLoc="center", colLoc="center", fontsize=18)

        plt.savefig(os.path.join(path, f"{layer_num}layer_{clustering_type}_{reduction_type}_heuristics.png"))
        plt.show()


class Tree_Cycle_Heuristics():
    def __init__(self):
        self.num_heuristics = 4

        self.base = self._build_base_heuristic()
        self.base_1arm = self._build_base_1_arm_heuristic()
        self.base_1arm2 = self._build_base_1_arm2_heuristic()
        self.base_2arm = self._build_base_2_arm_heuristic()
        self.base_tree = self._build_tree_heuristic()
        self.base_tree2 = self._build_tree2_heuristic()

        self.heuristic_functions = []
        self.heuristic_functions.append(self.eval_heuristic1)
        self.heuristic_functions.append(self.eval_heuristic2)
        self.heuristic_functions.append(self.eval_heuristic3)
        self.heuristic_functions.append(self.eval_heuristic4)
        self.heuristic_functions.append(self.eval_heuristic5)
        self.heuristic_functions.append(self.eval_heuristic6)

    def _build_base_heuristic(self):
        """Simple Circle"""
        G = nx.Graph()
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 4)
        G.add_edge(4, 5)
        G.add_edge(5, 6)
        G.add_edge(6, 1)

        return G

    def _build_base_1_arm_heuristic(self):
        G = self.base.copy()
        G.add_edge(1, 7)
        return G

    def _build_base_1_arm2_heuristic(self):
        G = self.base_1arm.copy()
        G.add_edge(7, 8)
        return G

    def _build_base_2_arm_heuristic(self):
        G = self.base_1arm2.copy()
        G.add_edge(4, 9)
        return G

    def _build_tree_heuristic(self):
        G = nx.Graph()
        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(1, 4)
        G.add_edge(2, 5)
        G.add_edge(2, 6)
        return G

    def _build_tree2_heuristic(self):
        G = self.base_tree.copy()
        G.add_edge(6, 7)
        return G

    def eval_heuristic1(self, G2, node):
        """Simple Circle"""
        return 1, "Circle", nx.is_isomorphic(self.base, G2)

    def eval_heuristic2(self, G2, node):
        """Simple Circle with Arm of length 1"""
        return 2, "Circle with Arm (len 1)", nx.is_isomorphic(self.base_1arm, G2)

    def eval_heuristic3(self, G2, node):
        """Simple Circle with Arm of length 1"""
        return 3, "Circle with Arm (len 2)", nx.is_isomorphic(self.base_1arm2, G2)

    def eval_heuristic4(self, G2, node):
        """Simple Circle with 2 Arms"""
        return 4, "Circle with 2 Arms", nx.is_isomorphic(self.base_2arm, G2)

    def eval_heuristic5(self, G2, node):
        """Tree"""
        ret = False
        if nx.is_isomorphic(self.base_tree, G2):
            if len(list(G2.neighbors(node))) == 2:
                ret = True

        return 5, "Balanced Bone", ret

    def eval_heuristic6(self, G2, node):
        """Tree2"""
        ret = False
        if nx.is_isomorphic(self.base_tree2, G2):
            if len(list(G2.neighbors(node))) == 2:
                ret = True

        return 6, "Unbalanced Bone", ret


    def eval(self, sample_graphs):
        results = []
        for i, (G, node) in enumerate(sample_graphs):
            ifFound = False
            for heuristic in self.heuristic_functions:
                ret = heuristic(G, node)

                if ret[2]:
                    results.append([str(i), str(ret[0]), ret[1]])
                    ifFound = True
                    break

            if not ifFound:
                results.append([str(i), "No Match", "-"])

        return results


class BA_Shapes_Heuristics():
    def __init__(self):
        self.num_heuristics = 8

        self.base = self._build_base_heuristic()
        self.base_middle = self._build_base_middle_heuristic()
        self.base_top = self._build_base_top_heuristic()
        self.base_bottom = self._build_base_bottom_heuristic()

        self.heuristic_functions = []
        self.heuristic_functions.append(self.eval_heuristic1)
        self.heuristic_functions.append(self.eval_heuristic2)
        self.heuristic_functions.append(self.eval_heuristic3)
        self.heuristic_functions.append(self.eval_heuristic4)
        self.heuristic_functions.append(self.eval_heuristic5)
        self.heuristic_functions.append(self.eval_heuristic6)
        self.heuristic_functions.append(self.eval_heuristic7)
        self.heuristic_functions.append(self.eval_heuristic8)

    def _build_base_heuristic(self):
        """Simple House"""
        G = nx.Graph()
        G.add_edge(1, 2) # roof side 1
        G.add_edge(1, 3) # roof side 2
        G.add_edge(2, 3) # middle connection
        G.add_edge(2, 4) # side 1
        G.add_edge(3, 5) # side 2
        G.add_edge(4, 5) # bottom connection

        return G

    def _build_base_middle_heuristic(self):
        """Top Node and Arm"""
        G = self.base.copy()
        G.add_edge(2, 6)

        return G

    def _build_base_top_heuristic(self):
        """Top Node and Arm"""
        G = self.base.copy()
        G.add_edge(1, 6)

        return G

    def _build_base_bottom_heuristic(self):
        """Top Node and Arm"""
        G = self.base.copy()
        G.add_edge(4, 6)

        return G


    def eval_heuristic1(self, G2, node):
        """House Top"""
        if nx.is_isomorphic(self.base, G2):
            ns = list(G2.neighbors(node))
            if len(ns) == 2:
                for n in ns:
                    if len(list(G2.neighbors(n))) == 2:
                        return 1, "Top Node", False

                return 1, "Top Node", True

        return 1, "Top Node", False


    def eval_heuristic2(self, G2, node):
        """House Middle"""
        if nx.is_isomorphic(self.base, G2):
            if len(list(G2.neighbors(node))) == 3:
                return 2, "Middle Node", True

        return 2, "Middle Node", False


    def eval_heuristic3(self, G2, node):
        """House Bottom"""
        ret = False
        if nx.is_isomorphic(self.base, G2):
            ns = list(G2.neighbors(node))
            if len(ns) == 2:
                n1 = False
                n2 = False
                for n in ns:
                    if len(list(G2.neighbors(n))) == 3:
                        n1 = True
                    elif len(list(G2.neighbors(n))) == 2:
                        n2 = True

        return 3, "Bottom Node", (n1 and n2)


    def eval_heuristic4(self, G2, node):
        """House Top + Middle Arm"""
        if nx.is_isomorphic(self.base_middle, G2):
            ns = list(G2.neighbors(node))
            if len(ns) == 2:
                for n in ns:
                    if len(list(G2.neighbors(n))) != 3 and len(list(G2.neighbors(n))) != 4:
                        return 4, "Top Node and Middle Arm", False

                return 4, "Top Node and Middle Arm", True

        return 4, "Top Node and Middle Arm", False


    def eval_heuristic5(self, G2, node):
        """House Middle + Middle Arm Close"""
        if nx.is_isomorphic(self.base_middle, G2):
            ns = list(G2.neighbors(node))
            if len(ns) == 4:
                for n in ns:
                    if len(list(G2.neighbors(n))) != 2 and len(list(G2.neighbors(n))) != 3:
                        return 5, "Middle Node and Middle Arm Close", True

                return 5, "Middle Node and Middle Arm Close", True

        return 5, "Middle Node and Middle Arm Close", False


    def eval_heuristic6(self, G2, node):
        """House Middle + Middle Arm Far"""
        if nx.is_isomorphic(self.base_middle, G2):
            ns = list(G2.neighbors(node))
            if len(ns) == 3:
                for n in ns:
                    if len(list(G2.neighbors(n))) != 2 and len(list(G2.neighbors(n))) != 4:
                        return 6, "Middle Node and Middle Arm Far", False

                return 6, "Middle Node and Middle Arm Far", True

        return 6, "Middle Node and Middle Arm Far", False


    def eval_heuristic7(self, G2, node):
        """House Bottom + Middle Arm Close"""
        ret = False
        if nx.is_isomorphic(self.base_middle, G2):
            ns = list(G2.neighbors(node))
            if len(ns) == 2:
                for n in ns:
                    if len(list(G2.neighbors(n))) == 4:
                        ret = True

        return 7, "Bottom Node and Middle Arm Close", ret


    def eval_heuristic8(self, G2, node):
        """House Bottom + Middle Arm Far"""
        ret = False
        if nx.is_isomorphic(self.base_middle, G2):
            ns = list(G2.neighbors(node))
            if len(ns) == 2:
                for n in ns:
                    if len(list(G2.neighbors(n))) == 3:
                        ret = True

        return 8, "Bottom Node and Middle Arm Far", ret


    def eval(self, sample_graphs):
        results = []
        for i, (G, node) in enumerate(sample_graphs):
            ifFound = False
            for heuristic in self.heuristic_functions:
                ret = heuristic(G, node)

                if ret[2]:
                    results.append([str(i), str(ret[0]), ret[1]])
                    ifFound = True
                    break

            if not ifFound:
                results.append([str(i), "No Match", "-"])

        return results


def plot_heuristics_table(heuristics, sample_graphs, layer_num, clustering_type, reduction_type, path):
    data = heuristics.eval(sample_graphs)
    headings = ["Cluster", "Heuristic", "Description"]

    fig, ax = plt.subplots(figsize=(10, int(0.25 * len(data))))
    ax.set_title(f"Concepts Identified per {clustering_type} Cluster in {reduction_type} Activation Space of Layer {layer_num}")
    ax.axis('off')
    ax.table(cellText=data, colLabels=headings, loc="center", rowLoc="center", cellLoc="center", colLoc="center", fontsize=18)

    plt.savefig(os.path.join(path, f"{layer_num}layer_{clustering_type}_heuristics.png"))
    plt.show()
