import numpy as np
import tensorflow as tf
from nltk import ngrams
import data_helpers


tf.flags.DEFINE_string("training_data_file", "./data/test.full.tsv", "Data source for the training data.")


shingles = [1, 2, 3, 4]


def jaccard_similarity(s1, s2):
	return (1.0 * len(s1.intersection(s2))) / len(s1.union(s2))


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
print("Loading data...")
x1, x2, y_truth, x1_lengths, x2_lengths = data_helpers.load_data_and_labels(FLAGS.training_data_file)
dataset = list(zip(x1, x2, y_truth))

best_t = 0
best_accuracy = 0
# for t in np.arange(0.05, 0.35, 0.01):
for t in [ 0.13 ]:
	correct = 0
	positive_predictions = 0
	tp = 0
	fp = 0
	fn = 0
	for datum in dataset:
		q1, q2, y = datum
		
		s1 = set()
		s2 = set()
		for n in shingles:
			for ngram in ngrams(q1.split(), n):
				s1.add(ngram)
			for ngram in ngrams(q2.split(), n):
				s2.add(ngram)
		
		j = jaccard_similarity(s1, s2)
		label_truth = 1 if y[1] == 1 else 0
		label_pred = 1 if j >= t else 0

		if label_truth == label_pred:
			correct += 1
			if label_truth == 1:
				tp += 1

		if label_pred == 1:
			positive_predictions += 1
			if label_truth == 0:
				fp += 1

		if label_pred == 0 and label_truth == 1:
			fn += 1

		# print "q1: ", q1
		# print "q2: ", q2
		# print "j: ", j
		# print "pred: %d, truth: %d" % (label_pred, label_truth)

	accuracy = (1.0 * correct) / len(dataset)
	precision = (1.0 * tp) / (tp + fp) if (tp + fp) > 0 else 0
	recall = (1.0 * tp) / (tp + fn) if (tp + fn) > 0 else 0
	f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
	positive_rate = ((1.0 * positive_predictions) / len(dataset))
	if accuracy > best_accuracy:
		best_t = t
		best_accuracy = accuracy
	print "t: %.2f \tacc %.4f \tprec %.4f \trec %.4f \tf1 %.4f \tpos %.4f \tbest_t %.2f \tbest_acc %.4f" % (t, accuracy, precision, recall, f1,positive_rate, best_t, best_accuracy)
