from gae.layers import (
    GraphConvolution,
    GraphConvolutionSparse,
    InnerProductDecoder,
)

# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


class Cluster_Model(object):
    def __init__(self, num_clusters, **kwargs):
        allowed_kwargs = {"name", "logging"}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, (
                "Invalid keyword argument: " + kwarg
            )

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, (
                "Invalid keyword argument: " + kwarg
            )
        name = kwargs.get("name")
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get("logging", False)
        self.logging = logging

        self.num_clusters = num_clusters

        self.vars = {}
        self.cluster_vars = {}

    def _build(self):
        raise NotImplementedError

    def _build_centers(self, num_clusters):
        raise NotImplementedError

    def build(self):
        """Wrapper for _build()"""
        with tf.variable_scope(self.name + "_params"):
            self._build()
        with tf.variable_scope(self.name + "_clustering"):
            self._build_centers(self.num_clusters)
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + "_params"
        )
        self.vars = {var.name: var for var in variables}
        cluster_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + "_clustering"
        )
        self.cluster_vars = {var.name: var for var in cluster_variables}
        print(f"{self.name+'_params'}: {self.vars}")
        print(f"{self.name+'_clustering'}: {self.cluster_vars}")

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAE(Cluster_Model):
    def __init__(
        self,
        placeholders,
        num_features,
        features_nonzero,
        num_clusters,
        **kwargs,
    ):
        super(GCNModelAE, self).__init__(num_clusters, **kwargs)

        self.inputs = placeholders["features"]
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders["adj"]
        self.dropout = placeholders["dropout"]
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(
            input_dim=self.input_dim,
            output_dim=FLAGS.hidden1,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=tf.nn.relu,
            dropout=self.dropout,
            logging=self.logging,
        )(self.inputs)

        self.embeddings = GraphConvolution(
            input_dim=FLAGS.hidden1,
            output_dim=FLAGS.hidden2,
            adj=self.adj,
            act=lambda x: x,
            dropout=self.dropout,
            logging=self.logging,
        )(self.hidden1)

        self.z_mean = self.embeddings

        self.reconstructions = InnerProductDecoder(
            input_dim=FLAGS.hidden2, act=lambda x: x, logging=self.logging
        )(self.embeddings)

    def _build_centers(self, k):
        self.centres = tf.Variable(
            tf.random.uniform((k, FLAGS.hidden2), seed=1234)
        )

        self.cluster_assignments = tf.math.argmin(
            tf.norm(
                self.centres - tf.expand_dims(self.embeddings, 1),
                axis=2,
            ),
            axis=1,
        )


# class GCNModelVAE(Model):
#     def __init__(
#         self, placeholders, num_features, num_nodes, features_nonzero, **kwargs
#     ):
#         super(GCNModelVAE, self).__init__(**kwargs)

#         self.inputs = placeholders["features"]
#         self.input_dim = num_features
#         self.features_nonzero = features_nonzero
#         self.n_samples = num_nodes
#         self.adj = placeholders["adj"]
#         self.dropout = placeholders["dropout"]
#         self.build()

#     def _build(self):
#         self.hidden1 = GraphConvolutionSparse(
#             input_dim=self.input_dim,
#             output_dim=FLAGS.hidden1,
#             adj=self.adj,
#             features_nonzero=self.features_nonzero,
#             act=tf.nn.relu,
#             dropout=self.dropout,
#             logging=self.logging,
#         )(self.inputs)

#         self.z_mean = GraphConvolution(
#             input_dim=FLAGS.hidden1,
#             output_dim=FLAGS.hidden2,
#             adj=self.adj,
#             act=lambda x: x,
#             dropout=self.dropout,
#             logging=self.logging,
#         )(self.hidden1)

#         self.z_log_std = GraphConvolution(
#             input_dim=FLAGS.hidden1,
#             output_dim=FLAGS.hidden2,
#             adj=self.adj,
#             act=lambda x: x,
#             dropout=self.dropout,
#             logging=self.logging,
#         )(self.hidden1)

#         self.z = self.z_mean + tf.random_normal(
#             [self.n_samples, FLAGS.hidden2]
#         ) * tf.exp(self.z_log_std)

#         self.reconstructions = InnerProductDecoder(
#             input_dim=FLAGS.hidden2, act=lambda x: x, logging=self.logging
#         )(self.z)
