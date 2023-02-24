# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

from sklearn.cluster import KMeans

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(
        self,
        preds,
        labels,
        embeds,
        pos_weight,
        norm,
        vars,
        cluster_vars,
        cluster_assignments,
    ):
        preds_sub = preds
        labels_sub = labels
        centres = list((cluster_vars.values()))[0]

        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=preds_sub, targets=labels_sub, pos_weight=pos_weight
            )
        )
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate
        )  # Adam Optimizer

        # self.grads_vars = self.optimizer.compute_gradients(self.cost)

        # For debugging (print in train.py)
        self.embeds = embeds
        self.centres = centres
        self.cluster_distances = tf.norm(
            centres - tf.expand_dims(embeds, 1), axis=2
        )
        self.cluster_assignments = cluster_assignments

        # Distance of each node to closest centre
        self.cluster_centre_distance = (
            tf.gather(centres, cluster_assignments) - embeds
        )
        # Squared Euclidean distance to each node's cluster centre (i.e. "residual")
        self.clustering_loss_partial = tf.norm(
            self.cluster_centre_distance, axis=1
        )
        # Total clustering loss (sum of squared distances)
        self.clustering_loss_total = tf.reduce_sum(
            self.clustering_loss_partial
        )
        # Mean clustering loss (less dependant on num nodes)
        self.clustering_loss_avg = tf.reduce_mean(self.clustering_loss_partial)

        self.cluster_counts = tf.tensor_scatter_nd_add(
            tf.zeros(FLAGS.num_clusters),  # 5
            tf.expand_dims(self.cluster_assignments, axis=1),  # 700,1
            tf.ones(tf.shape(self.cluster_assignments)),  # 700
        )

        self.update = tf.expand_dims(
            tf.math.divide_no_nan(1.0, self.cluster_counts), axis=1
        ) * tf.tensor_scatter_nd_add(
            tf.zeros(centres.shape),  # 5,16
            tf.expand_dims(self.cluster_assignments, axis=1),  # 700,1
            self.cluster_centre_distance,  # 700,16
        )

        self.update_centres = centres.assign_add(-self.update)

        self.opt_op = self.optimizer.minimize(
            (FLAGS.beta * self.cost)
            + (FLAGS.alpha * self.clustering_loss_avg),
            var_list=vars.values(),
        )

        self.correct_prediction = tf.equal(
            tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
            tf.cast(labels_sub, tf.int32),
        )
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32)
        )


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=preds_sub, targets=labels_sub, pos_weight=pos_weight
            )
        )
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate
        )  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(
            tf.reduce_sum(
                1
                + 2 * model.z_log_std
                - tf.square(model.z_mean)
                - tf.square(tf.exp(model.z_log_std)),
                1,
            )
        )
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(
            tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
            tf.cast(labels_sub, tf.int32),
        )
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32)
        )
