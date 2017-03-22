import tensorflow as tf
import numpy as np


class SiameseLSTM(object):
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
        self.input_x1_length = tf.placeholder(tf.int32, [None], name="input_x1_length")
        self.input_x2_length = tf.placeholder(tf.int32, [None], name="input_x2_length")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        init_embeddings = tf.Variable(pretrained_embeddings)
        embedding_size = pretrained_embeddings.shape[1]
        
        embedded_x1 = self.create_embedding(self.input_x1, init_embeddings)
        embedded_x2 = self.create_embedding(self.input_x2, init_embeddings)

        r1 = self.create_tower(embedded_x1, sequence_length, embedding_size, filter_sizes, num_filters, self.input_x1_length, "0", False)
        r2 = self.create_tower(embedded_x2, sequence_length, embedding_size, filter_sizes, num_filters, self.input_x2_length, "0", True)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            features = tf.concat(1, [r1, r2, r1 - r2, tf.multiply(r1, r2)])
            num_filters_total = num_filters * len(filter_sizes)
            feature_length = 4 * r1.get_shape().as_list()[1]

            num_hidden1 = 256  #int(np.sqrt(feature_length))
            num_hidden2 = 256  #int(np.sqrt(feature_length))

            W3= tf.get_variable(
                "W3",
                shape=[feature_length, num_hidden1],
                initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.constant(0.1, shape=[num_hidden1]), name="b3")
            H3 = tf.nn.relu(tf.nn.xw_plus_b(features, W3, b3, name="hidden"))

            W4 = tf.get_variable(
                "W4",
                shape=[num_hidden1, num_hidden2],
                initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.constant(0.1, shape=[num_hidden2]), name="b4")
            H4 = tf.nn.relu(tf.nn.xw_plus_b(H3, W4, b4, name="hidden"))

            W5 = tf.get_variable(
                "W5",
                shape=[num_hidden2, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b5")

            l2_loss += tf.nn.l2_loss(W5)
            l2_loss += tf.nn.l2_loss(b5)
            self.scores = tf.nn.xw_plus_b(H4, W5, b5, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.y_truth = tf.argmax(self.input_y, 1, name="y_truth")
            self.correct_predictions = tf.equal(self.predictions, self.y_truth, name="correct_predictions")
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")


    def create_tower(self, embeddings, sequence_length, embedding_size, filter_sizes, num_filters, x_lengths, qid, reuse):
        W_name = "W" + qid
        b_name = "b" + qid
        h_name = "h" + qid
        conv_name = "conv" + qid
        pool_name = "pool" + qid

        with tf.variable_scope("inference", reuse=reuse):
            with tf.name_scope("lstm"):
                x = embeddings

                # Initial state of the LSTM cell memory
                state_size = 5
                num_layers = 3
                cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size, state_is_tuple=True)
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
                outputs, states  = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell,
                    cell_bw=cell,
                    dtype=tf.float32,
                    sequence_length=x_lengths,
                    inputs=x)
                 
                output_fw, output_bw = outputs
                states_fw, states_bw = states

                encoded = tf.stack([output_fw, output_bw], axis=3)
                encoded = tf.concat(3, encoded)
                encoded = tf.reshape(encoded, [-1, sequence_length * state_size * 2])

            # Add dropout
            with tf.name_scope("dropout-%s" % qid):
                h_drop = tf.nn.dropout(encoded, self.dropout_keep_prob)
        return h_drop


    def create_embedding(self, x, init_embeddings):
        embeddings = tf.nn.embedding_lookup(init_embeddings, x)
        return embeddings
