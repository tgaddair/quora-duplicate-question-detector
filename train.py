#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import sklearn as sk
import os
import time
import datetime
import data_helpers
from tensorflow.contrib import learn

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from models.baseline_nn import BaselineNN
from models.siamese_cnn import SiameseCNN
from models.siamese_lstm import SiameseLSTM

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("training_data_file", "./data/training.full.tsv", "Data source for the training data.")
tf.flags.DEFINE_string("embeddings_file", "./glove.6B/glove.6B.100d.txt", "Data source for the pretrained word embeddings")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 50)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("use_cached_embeddings", True, "Cache embeddings locally on disk for repeated runs")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
q1, q2, y, q1_lengths, q2_lengths = data_helpers.load_data_and_labels(FLAGS.training_data_file)

# Build vocabulary
max_question_length = max(max([len(x.split(" ")) for x in q1]), max([len(x.split(" ")) for x in q2]))
vocab_processor = learn.preprocessing.VocabularyProcessor(max_question_length)
print "max_question_length: ", max_question_length

x_text = q1 + q2
x = np.array(list(vocab_processor.fit_transform(x_text)))
x1_sliced = x[:len(q1)]
x2_sliced = x[len(q1):]

# The models are not perfectly symmetric in the combination layer, so we can flip the order of the
# questions to synthesize additional training examples
x1 = np.concatenate((x1_sliced, x2_sliced), axis=0)
x2 = np.concatenate((x2_sliced, x1_sliced), axis=0)
y = np.concatenate((y, y), axis=0)
x1_lengths = np.concatenate((q1_lengths, q2_lengths), axis=0)
x2_lengths = np.concatenate((q2_lengths, q1_lengths), axis=0)

# Create word embeddings
print "Loading word embeddings..."
vocab_dict = vocab_processor.vocabulary_._mapping
pretrained_embeddings = data_helpers.load_embeddings(FLAGS.embeddings_file,
                                                     vocab_dict,
                                                     FLAGS.embedding_dim,
                                                     FLAGS.use_cached_embeddings)

# Randomly shuffle data
print "Shuffling data..."
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x1_shuffled = x1[shuffle_indices]
x2_shuffled = x2[shuffle_indices]
y_shuffled = y[shuffle_indices]
q1_lengths_shuffled = x1_lengths[shuffle_indices]
q2_lengths_shuffled = x2_lengths[shuffle_indices]

# Split train/test set
print "Splitting training/dev..."
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x1_train, x1_dev = x1_shuffled[:dev_sample_index], x1_shuffled[dev_sample_index:]
x2_train, x2_dev = x2_shuffled[:dev_sample_index], x2_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
x1_lengths_train, x1_lengths_dev = q1_lengths_shuffled[:dev_sample_index], q1_lengths_shuffled[dev_sample_index:]
x2_lengths_train, x2_lengths_dev = q2_lengths_shuffled[:dev_sample_index], q2_lengths_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # cnn = BaselineNN(
        #     sequence_length=x1_train.shape[1],
        #     num_classes=y_train.shape[1],
        #     pretrained_embeddings=pretrained_embeddings)

        cnn = SiameseCNN(
            sequence_length=x1_train.shape[1],
            num_classes=y_train.shape[1],
            pretrained_embeddings=pretrained_embeddings,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x1_batch, x2_batch, y_batch, x1_lengths_batch, x2_lengths_batch, epoch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x1: x1_batch,
              cnn.input_x2: x2_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
              cnn.input_x1_length: x1_lengths_batch,
              cnn.input_x2_length: x2_lengths_batch,
            }
            _, step, summaries, loss, accuracy, scores, predictions, y_truth = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.scores, cnn.predictions, cnn.y_truth],
                feed_dict)

            labels = [0, 1]
            precision = precision_score(y_truth, predictions, labels)
            recall = recall_score(y_truth, predictions, labels)
            f1 = f1_score(y_truth, predictions, labels)

            time_str = datetime.datetime.now().isoformat()
            print("{}: \tepoch {}, \tstep {}, \tloss {:g}, \tacc {:g}, \tprec {:g}, \trec {:g}, \tf1 {:g}".format(time_str, epoch, step, loss, accuracy, precision, recall, f1))
            # print "scores: ", scores
            # print "predictions: ", predictions
            # print "y: ", y
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x1_batch, x2_batch, y_batch, x1_lengths_batch, x2_lengths_batch, epoch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x1: x1_batch,
              cnn.input_x2: x2_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0,
              cnn.input_x1_length: x1_lengths_batch,
              cnn.input_x2_length: x2_lengths_batch,
            }
            step, summaries, loss, accuracy, scores, predictions, y_truth = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.scores, cnn.predictions, cnn.y_truth],
                feed_dict)

            labels = [0, 1]
            precision = precision_score(y_truth, predictions, labels)
            recall = recall_score(y_truth, predictions, labels)
            f1 = f1_score(y_truth, predictions, labels)
            print "scores: ", scores
            print "predictions: ", predictions
            print "y_truth: ", y_truth

            time_str = datetime.datetime.now().isoformat()
            print("{}: \tepoch {}, \tstep {}, \tloss {:g}, \tacc {:g}, \tprec {:g}, \trec {:g}, \tf1 {:g}".format(time_str, epoch, step, loss, accuracy, precision, recall, f1))
            if writer:
                writer.add_summary(summaries, step)

        # Training loop. For each batch...
        dataset = list(zip(x1_train, x2_train, y_train, x1_lengths_train, x2_lengths_train))
        for epoch in range(FLAGS.num_epochs):
            print "\n\n##########################"
            print "# Epoch: ", epoch
            print "##########################\n\n"

            # Generate batches
            batches = data_helpers.batch_iter(dataset, FLAGS.batch_size)
            for batch in batches:
                x1_batch, x2_batch, y_batch, x1_lengths_batch, x2_lengths_batch = zip(*batch)
                train_step(x1_batch, x2_batch, y_batch, x1_lengths_batch, x2_lengths_batch, epoch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x1_dev, x2_dev, y_dev, x1_lengths_dev, x2_lengths_dev, epoch, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
