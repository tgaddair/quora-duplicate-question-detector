import numpy as np
import re
import itertools
import os.path
from collections import Counter
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


NUM = "<NUM>"


def clean_str(string, lower=True):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    result = string.strip()
    if lower:
        result = result.lower()
    return result


def load_data_and_labels(training_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    training_data = list(open(training_data_file, "r").readlines())
    training_data = [s.strip() for s in training_data]

    q1 = []
    q2 = []
    labels = []
    q1_lengths = []
    q2_lengths = []
    for line in training_data:
        elements = line.split('\t')

        q1_length = len(elements[1].split())
        q2_length = len(elements[2].split())
        if q1_length > 59 or q2_length > 59:
            continue

        q1.append(elements[1].lower())
        q2.append(elements[2].lower())
        labels.append([0, 1] if elements[0] == '1' else [1, 0])
        q1_lengths.append(q1_length)
        q2_lengths.append(q2_length)

    labels = np.concatenate([labels], 0)
    q1_lengths = np.concatenate([q1_lengths], 0)
    q2_lengths = np.concatenate([q2_lengths], 0)

    return [q1, q2, labels, q1_lengths, q2_lengths]


def batch_iter(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1

    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


def load_word_vector_mapping(embeddings_file):
    """
    Load word vector mapping using @vocab_fstream, @vector_fstream.
    Assumes each line of the vocab file matches with those of the vector
    file.
    """
    ret = OrderedDict()
    for row in list(open(embeddings_file, "r").readlines()):
        elements = row.strip().split()
        vocab = elements[0]
        ret[vocab] = np.array(list(map(float, elements[1:])))
    return ret


def normalize(word):
    """
    Normalize words that are numbers or have casing.
    """
    # if word.isdigit(): return NUM
    # else: return word.lower()
    return word.lower()


def load_embeddings(embeddings_file, vocab_dict, embedding_dim, use_cache=True):
    embeddings_cache_file = embeddings_file + ".cache.npy"
    if use_cache and os.path.isfile(embeddings_cache_file):
        embeddings = np.load(embeddings_cache_file)
        return embeddings

    embeddings = np.array(np.random.randn(len(vocab_dict) + 1, embedding_dim), dtype=np.float32)
    embeddings[0] = 0.
    for word, vec in load_word_vector_mapping(embeddings_file).items():
        word = normalize(word)
        if word in vocab_dict:
            embeddings[vocab_dict[word]] = vec

    np.save(embeddings_cache_file, embeddings)
    logger.info("Initialized embeddings.")
    return embeddings
