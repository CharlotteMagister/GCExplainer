import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import networkx as nx

from torch_geometric.nn import MessagePassing, GCNConv, DenseGCNConv, GINConv, GraphConv
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_mean_pool, GlobalAttention

class BA_Shapes_GCN(nn.Module):
    def __init__(self, num_conv_layers, num_in_features, num_hidden_features, num_classes):
        super(BA_Shapes_GCN, self).__init__()

        self.name = "BA-Shapes"

        self.conv0 = GCNConv(num_in_features, num_hidden_features)
        self.conv1 = GCNConv(num_hidden_features, num_hidden_features)
        self.conv2 = GCNConv(num_hidden_features, num_hidden_features)
        self.conv3 = GCNConv(num_hidden_features, num_hidden_features)

        # linear layers
        self.linear = nn.Linear(num_hidden_features, num_classes)

    def forward(self, x, edge_index):
        x = self.conv0(x, edge_index)
        x = F.relu(x)

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = self.linear(x)

        return F.log_softmax(x, dim=-1)


class Tree_Cycle_GCN(nn.Module):
    def __init__(self, num_conv_layers, num_in_features, num_hidden_features, num_classes):
        super(Tree_Cycle_GCN, self).__init__()

        # convolutional layers
        # hidden_features = 20
        self.conv0 = DenseGCNConv(num_in_features, num_hidden_features)

        self.conv1 = DenseGCNConv(num_hidden_features, num_hidden_features)

        self.conv2 = DenseGCNConv(num_hidden_features, num_hidden_features)

        # linear layers
        self.linear = nn.Linear(num_hidden_features, num_classes)

    def forward(self, x, edge_index):
        x = self.conv0(x, edge_index)
        x = F.relu(x)

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.linear(x)

        return F.log_softmax(x, dim=-1).squeeze()


class Pool(torch.nn.Module):
    def __init__(self):
        super(Pool, self).__init__()

    def forward(self, x, batch):
        return global_mean_pool(x, batch)


# Learned from: https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharingand

# class Mutag_GCN(torch.nn.Module):
#     def __init__(self, num_node_features, num_classes):
#         super(Mutag_GCN, self).__init__()
#
#         num_hidden_units = 32
#         self.nn0 = nn.Sequential(nn.Linear(num_node_features, num_hidden_units), nn.ReLU(), nn.Linear(num_hidden_units, num_hidden_units))
#         self.nn1 = nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), nn.ReLU(), nn.Linear(num_hidden_units, num_hidden_units))
#         self.nn2 = nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), nn.ReLU(), nn.Linear(num_hidden_units, num_hidden_units))
#
#         self.conv0 = GINConv(self.nn0)
#         self.conv1 = GINConv(self.nn1)
#         self.conv2 = GINConv(self.nn2)
#
#         # self.conv0 = GCNConv(num_node_features, num_hidden_units)
#         # self.conv1 = GCNConv(num_hidden_units, num_hidden_units)
#         # self.conv2 = GCNConv(num_hidden_units, num_hidden_units)
#
#         # self.pool0 = GlobalAttention(nn.Linear(num_hidden_units, 1))
#         # self.pool1 = GlobalAttention(nn.Linear(num_hidden_units, 1))
#         # self.pool2 = GlobalAttention(nn.Linear(num_hidden_units, 1))
#
#         self.pool0 = Pool()
#         self.pool1 = Pool()
#         self.pool2 = Pool()
#
#         self.lin = nn.Linear(num_hidden_units, num_classes)
#
#     def forward(self, x, edge_index, batch):
#         # 1. Obtain node embeddings
#         x = self.conv0(x, edge_index)
#         x = F.relu(x)
#
#         _ = self.pool0(x, batch)
#
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#
#         _ = self.pool1(x, batch)
#
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#
#         x = self.pool2(x, batch)
#
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin(x)
#
#         self.last = x
#
#         return x


class Mutag_GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(Mutag_GCN, self).__init__()

        num_hidden_units = 30
        self.conv0 = GCNConv(num_node_features, num_hidden_units)
        self.conv1 = GCNConv(num_hidden_units, num_hidden_units)
        self.conv2 = GCNConv(num_hidden_units, num_hidden_units)
        self.conv3 = GCNConv(num_hidden_units, num_hidden_units)
        # self.conv4 = GCNConv(num_hidden_units, num_hidden_units)
        # self.conv5 = GCNConv(num_hidden_units, num_hidden_units)
        # self.conv6 = GCNConv(num_hidden_units, num_hidden_units)

        self.pool0 = Pool()
        self.pool1 = Pool()
        self.pool2 = Pool()
        self.pool3 = Pool()
        # self.pool4 = Pool()
        # self.pool5 = Pool()
        # self.pool6 = Pool()

        self.lin = nn.Linear(num_hidden_units, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv0(x, edge_index)
        x = F.relu(x)

        _ = self.pool0(x, batch)

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        _ = self.pool1(x, batch)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        _ = self.pool2(x, batch)

        x = self.conv3(x, edge_index)
        # x = F.relu(x)

        x = self.pool3(x, batch)

        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class Reddit_GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(Reddit_GCN, self).__init__()

        num_hidden_units = 30
        self.conv0 = GCNConv(num_node_features, num_hidden_units)
        self.conv1 = GCNConv(num_hidden_units, num_hidden_units)
        self.conv2 = GCNConv(num_hidden_units, num_hidden_units)
        self.conv3 = GCNConv(num_hidden_units, num_hidden_units)

        self.pool0 = Pool()
        self.pool1 = Pool()
        self.pool2 = Pool()
        self.pool3 = Pool()

        self.lin = nn.Linear(num_hidden_units, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv0(x, edge_index)
        x = F.relu(x)

        _ = self.pool0(x, batch)

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        _ = self.pool1(x, batch)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        _ = self.pool2(x, batch)

        x = self.conv3(x, edge_index)
        # x = F.relu(x)

        x = self.pool3(x, batch)

        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


global activation_list
activation_list = {}


def get_activation(idx):
    '''Learned from: https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6'''
    def hook(model, input, output):
        activation_list[idx] = output.detach()

    return hook


def register_hooks(model):
    # register hooks to extract activations
    if isinstance(model, Mutag_GCN):
        for name, m in model.named_modules():
            if isinstance(m, GlobalAttention):
                m.register_forward_hook(get_activation(f"{name}"))
            if isinstance(m, nn.Linear):
                m.register_forward_hook(get_activation(f"{name}"))
            if isinstance(m, Pool):
                m.register_forward_hook(get_activation(f"{name}"))

    else:
        for name, m in model.named_modules():
            if isinstance(m, GCNConv) or isinstance(m, DenseGCNConv):
                m.register_forward_hook(get_activation(f"{name}"))

    return model


def weights_init(m):
    if isinstance(m, GCNConv):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
        torch.nn.init.uniform_(m.bias.data)


def test(model, node_data_x, node_data_y, edge_list, mask):
    # enter evaluation mode
    model.eval()

    correct = 0
    pred = model(node_data_x, edge_list).max(dim=1)[1]

    correct += pred[mask].eq(node_data_y[mask]).sum().item()
    return correct / (len(node_data_y[mask]))


def train(model, data, epochs, lr, path):
    # register hooks to track activation
    model = register_hooks(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # list of accuracies
    train_accuracies, test_accuracies, train_losses, test_losses = list(), list(), list(), list()

    # get data
    x = data["x"]
    edges = data["edges"]
    y = data["y"]
    train_mask = data["train_mask"]
    test_mask = data["test_mask"]

    # iterate for number of epochs
    for epoch in range(epochs):
            # set mode to training
            model.train()
            optimizer.zero_grad()

            # input data
            out = model(x, edges)

            # calculate loss
            loss = F.nll_loss(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                test_loss = F.nll_loss(out[test_mask], y[test_mask])

                # get accuracy
                train_acc = test(model, x, y, edges, train_mask)
                test_acc = test(model, x, y, edges, test_mask)

            ## add to list and print
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())

            print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
                  format(epoch, loss.item(), train_acc, test_acc), end = "\r")

    # plut accuracy graph
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Testing Accuracy")
    plt.title(f"Accuracy of {model.name} Model during Training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(path, f"model_accuracy_plot.png"))
    plt.show()

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Testing Loss")
    plt.title(f"Loss of {model.name} Model during Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(path, f"model_loss_plot.png"))
    plt.show()

    # save model
    torch.save(model.state_dict(), os.path.join(path, "model.pkl"))

    with open(os.path.join(path, "activations.txt"), 'wb') as file:
        pickle.dump(activation_list, file)


def test_graph_class(model, dataloader):
    # enter evaluation mode
    correct = 0
    for data in dataloader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    return correct / len(dataloader.dataset)


def train_graph_class(model, train_loader, test_loader, full_loader, epochs, lr, path):
    # register hooks to track activation
    model = register_hooks(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    # list of accuracies
    train_accuracies, test_accuracies, train_loss, test_loss = list(), list(), list(), list()

    for epoch in range(epochs):
        model.train()

        running_loss = 0
        num_batches = 0
        for data in train_loader:
            model.train()

            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.batch)

            # calculate loss
            one_hot = torch.nn.functional.one_hot(data.y, num_classes=2).type_as(out)
            if out.shape[1] == 1 or one_hot.shape[1] == 1:
                print("What ", out.shape)
                print(out)
                print("What2 ", data.y.shape, " ", one_hot.shape)
                print(data.y)
                print(one_hot)
            loss = criterion(out, one_hot)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            running_loss += loss.item()
            num_batches += 1

            optimizer.step()

        # get accuracy
        train_acc = test_graph_class(model, train_loader)
        test_acc = test_graph_class(model, test_loader)

        # add to list and print
        model.eval()
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # get testing loss
        test_running_loss = 0
        test_num_batches = 0
        for data in test_loader:
            out = model(data.x, data.edge_index, data.batch)
            one_hot = torch.nn.functional.one_hot(data.y, num_classes=2).type_as(out)
            if out.shape[1] == 1 or one_hot.shape[1] == 1:
                print("What ", out.shape)
                print(out)
                print("What2 ", data.y.shape, " ", one_hot.shape)
                print(data.y)
                print(one_hot)
            test_running_loss += criterion(out, one_hot).item()
            test_num_batches += 1

        train_loss.append(running_loss / num_batches)
        test_loss.append(test_running_loss / test_num_batches)

        print('Epoch: {:03d}, Train Loss: {:.5f}, Test Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
                  format(epoch, train_loss[-1], test_loss[-1], train_acc, test_acc))

    # plut accuracy graph
    plt.plot(train_accuracies, label="Train accuracy")
    plt.plot(test_accuracies, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper right')
    plt.show()

    plt.plot(train_loss, label="Train loss")
    plt.plot(test_loss, label="Test loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.show()


    for data in full_loader:
        out = model(data.x, data.edge_index, data.batch)

    torch.save(model.state_dict(), os.path.join(path, "model.pkl"))

    with open(os.path.join(path, "activations.txt"), 'wb') as file:
        pickle.dump(activation_list, file)
