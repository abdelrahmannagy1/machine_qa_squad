import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

class QAModel:
    """
        Question Answering Module
    """

    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        self.id2word = id2word
        self.word2id = word2id
        self.FLAGS = FLAGS
        with tf.variable_scope("QAModel",initialzer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform = True)):
            self.add_placeholders()
            self.add_embedding_layer()
            self.build_graph()
            self.add_loss()
        #Trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)
        #Optimizer and updates
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        #Savers
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()


    def add_placeholders(self):


    def build_graph(self):


    def add_loss(self):

    def add_embedding_layer(self):