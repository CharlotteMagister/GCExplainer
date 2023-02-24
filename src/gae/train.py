from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.random.set_random_seed(1234)

import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.input_data import load_data, bhshapes
from gae.model import GCNModelAE  # , GCNModelVAE
from gae.preprocessing import (
    preprocess_graph,
    construct_feed_dict,
    sparse_to_tuple,
    mask_test_edges,
)

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
flags.DEFINE_integer("epochs", 200, "Number of epochs to train.")
flags.DEFINE_integer("hidden1", 32, "Number of units in hidden layer 1.")
flags.DEFINE_integer("hidden2", 16, "Number of units in hidden layer 2.")
flags.DEFINE_float(
    "weight_decay", 0.0, "Weight for L2 loss on embedding matrix."
)
flags.DEFINE_float("dropout", 0.0, "Dropout rate (1 - keep probability).")

flags.DEFINE_string("model", "gcn_ae", "Model string.")
flags.DEFINE_string("dataset", "cora", "Dataset string.")
flags.DEFINE_integer("features", 1, "Whether to use features (1) or not (0).")
flags.DEFINE_integer("num_clusters", 5, "How many clusters.")
flags.DEFINE_float("alpha", 0.5, "How much to scale clustering loss")
flags.DEFINE_float("beta", 1.0, "How much to scale edge prediction loss")
flags.DEFINE_integer("num_bh_nodes", 300, "How many nodes in BH graph")
flags.DEFINE_integer("num_bh_houses", 80, "How many nodes in BH graph")
flags.DEFINE_string(
    "bh_features",
    "ones",
    "Feature type for bh shapes identity ~ all unique, ones ~ all 1",
)
flags.DEFINE_string("save_prefix", "bashapes", "prefix to use for save files")

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Load data
# adj, features = load_data(dataset_str)
print("LOADING DATASET")
adj, features = bhshapes()

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix(
    (adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape
)
adj_orig.eliminate_zeros()

print("MASKING EDGES")
(
    adj_train,
    train_edges,
    val_edges,
    val_edges_false,
    test_edges,
    test_edges_false,
) = mask_test_edges(adj)
adj = adj_train

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

print("PREPROCESSING")
# Some preprocessing
adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    "features": tf.sparse_placeholder(tf.float32),
    "adj": tf.sparse_placeholder(tf.float32),
    "adj_orig": tf.sparse_placeholder(tf.float32),
    "dropout": tf.placeholder_with_default(0.0, shape=()),
}

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

print("CREATING MODEL")
# Create model
model = None
if model_str == "gcn_ae":
    model = GCNModelAE(
        placeholders, num_features, features_nonzero, FLAGS.num_clusters
    )
elif model_str == "gcn_vae":
    model = GCNModelVAE(
        placeholders, num_features, num_nodes, features_nonzero
    )

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = (
    adj.shape[0]
    * adj.shape[0]
    / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
)

print("CREATING OPTIMISER")
# Optimizer
with tf.name_scope("optimizer"):
    if model_str == "gcn_ae":
        opt = OptimizerAE(
            preds=model.reconstructions,
            labels=tf.reshape(
                tf.sparse_tensor_to_dense(
                    placeholders["adj_orig"], validate_indices=False
                ),
                [-1],
            ),
            embeds=model.z_mean,
            pos_weight=pos_weight,
            norm=norm,
            vars=model.vars,
            cluster_vars=model.cluster_vars,
            cluster_assignments=model.cluster_assignments,
        )
    elif model_str == "gcn_vae":
        opt = OptimizerVAE(
            preds=model.reconstructions,
            labels=tf.reshape(
                tf.sparse_tensor_to_dense(
                    placeholders["adj_orig"], validate_indices=False
                ),
                [-1],
            ),
            model=model,
            num_nodes=num_nodes,
            pos_weight=pos_weight,
            norm=norm,
        )

print("STARTING TF SESSION")
# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []


def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders["dropout"]: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

# Train model
print("STARTING TRAINING")
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(
        adj_norm, adj_label, features, placeholders
    )
    feed_dict.update({placeholders["dropout"]: FLAGS.dropout})
    # Run single weight update
    outs = sess.run(
        {
            "update_weights": opt.opt_op,
            "encoder_loss": opt.cost,
            "decoder_acc": opt.accuracy,
            "node_embeds": opt.embeds,
            "cluster_centres": opt.centres,
            "distances_to_centres": opt.cluster_distances,
            "cluster_assignments": opt.cluster_assignments,
            "node_residuals": opt.clustering_loss_partial,
            "clustering_loss": opt.clustering_loss_avg,
            "change_in_centres": opt.update,
            "update_centres": opt.update_centres,
            "nodes_per_cluster": opt.cluster_counts,
        },
        feed_dict=feed_dict,
    )

    # Compute average loss
    avg_cost = outs["encoder_loss"]
    avg_accuracy = outs["decoder_acc"]
    cluster_loss = outs["clustering_loss"]
    cluster_counts = outs["nodes_per_cluster"]

    roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    val_roc_score.append(roc_curr)

    print(
        "Epoch:",
        "%04d" % (epoch + 1),
        "train_loss=",
        "{:.5f}".format(avg_cost),
        "cluster_loss=",
        "{:.5f}".format(cluster_loss),
        "cluster_counts=",
        "{}".format(cluster_counts.tolist()),
        "train_acc=",
        "{:.5f}".format(avg_accuracy),
        "val_roc=",
        "{:.5f}".format(val_roc_score[-1]),
        "val_ap=",
        "{:.5f}".format(ap_curr),
        "time=",
        "{:.5f}".format(time.time() - t),
    )

feed_dict.update({placeholders["dropout"]: 0})
emb = sess.run(model.z_mean, feed_dict=feed_dict)
# print(emb)
# print(type(emb))
print(emb.shape)
np.save(f"{FLAGS.save_prefix}.embeds", emb)


print("Optimization Finished!")

roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
print("Test ROC score: " + str(roc_score))
print("Test AP score: " + str(ap_score))
