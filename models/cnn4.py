import tensorflow as tf
import numpy as np


class CNN4(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, pretrained_embeddings,
      filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        init_embeddings = tf.Variable(pretrained_embeddings)
        embedding_size = pretrained_embeddings.shape[1]
        
        embedded_x1 = self.create_embedding(self.input_x1, init_embeddings)
        embedded_x2 = self.create_embedding(self.input_x2, init_embeddings)

        r1 = self.create_tower(embedded_x1, sequence_length, embedding_size, filter_sizes, num_filters, "0", False)
        r2 = self.create_tower(embedded_x2, sequence_length, embedding_size, filter_sizes, num_filters, "0", True)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            features = tf.concat(1, [r1, r2, r1 - r2, tf.multiply(r1, r2)])
            num_filters_total = num_filters * len(filter_sizes)
            feature_length = 4 * num_filters_total

            num_hidden = int(np.sqrt(feature_length))
            W3= tf.get_variable(
                "W3",
                shape=[feature_length, num_hidden],
                initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="b3")
            H = tf.nn.relu(tf.nn.xw_plus_b(features, W3, b3, name="hidden"))

            W4 = tf.get_variable(
                "W4",
                shape=[num_hidden, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b4")

            l2_loss += tf.nn.l2_loss(W4)
            l2_loss += tf.nn.l2_loss(b4)
            self.scores = tf.nn.xw_plus_b(H, W4, b4, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            # self.y = tf.expand_dims(tf.cast(tf.argmax(self.input_y, 1), tf.float32), -1, name="y")
            # self.loss = tf.nn.l2_loss(self.scores - self.y)

            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.y_truth = tf.argmax(self.input_y, 1, name="y_truth")
            self.correct_predictions = tf.equal(self.predictions, self.y_truth, name="correct_predictions")
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")


    def create_tower(self, embeddings, sequence_length, embedding_size, filter_sizes, num_filters, qid, reuse):
        W_name = "W" + qid
        b_name = "b" + qid
        h_name = "h" + qid
        conv_name = "conv" + qid
        pool_name = "pool" + qid
        norm_name = "norm" + qid

        with tf.variable_scope("inference", reuse=reuse):
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s-%s" % (qid, filter_size)):
                    print "embeddings: ", embeddings.get_shape()

                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name=W_name)
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name=b_name)
                    conv = tf.nn.conv2d(
                        embeddings,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name=conv_name)

                    print "W: ", W.get_shape()
                    print "conv: ", conv.get_shape()


                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name=h_name)

                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, 3, 1, 1],
                        strides=[1, 2, 1, 1],
                        padding='VALID',
                        name=pool_name)
                    print "pooled: ", pooled.get_shape()

                    # norm1
                    norm1 = tf.nn.lrn(pooled, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                                      name=norm_name)

                with tf.name_scope("conv-maxpool2-%s-%s" % (qid, filter_size)):
                    # Convolution Layer
                    filter_shape = [filter_size, 1, num_filters, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name=W_name + "2")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name=b_name + "2")
                    conv = tf.nn.conv2d(
                        norm1,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name=conv_name + "2")

                    print "W2: ", W.get_shape()
                    print "conv2: ", conv.get_shape()

                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name=h_name + "2")

                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, h.get_shape()[1], 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name=pool_name + "2")
                    print "pooled2: ", pooled.get_shape()

                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            h_pool = tf.concat(3, pooled_outputs)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            # Add dropout
            with tf.name_scope("dropout-%s" % qid):
                h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        return h_drop


    def create_embedding(self, x, init_embeddings):
        lookups = tf.nn.embedding_lookup(init_embeddings, x)
        # shape = lookups.get_shape().as_list()
        # embeddings = tf.reshape(lookups, [-1, shape[1] * shape[2]]) 
        embeddings = tf.expand_dims(lookups, -1)
        return embeddings
