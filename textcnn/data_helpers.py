import numpy as np
import re
import itertools
from collections import Counter
from gensim.models import word2vec


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = []
    for line in open(positive_data_file, "r", encoding='utf-8').readlines()[:10000]:
        if 128 < len(line.strip().split()) < 512:
            positive_examples.append(line.strip().split())
        # temp = []
        # for word in line.strip().split():
        #     if word in model:
        #         temp.extend(model[word])
        #     else:
        #         temp.extend([0.0]*32)
        # positive_examples.append(temp)
    negative_examples = []
    for line in open(negative_data_file, "r", encoding='utf-8').readlines()[:10000]:
        if 256 < len(line.split()) < 512:
            negative_examples.append(line.strip().split())
        # temp = []
        # for word in line.strip().split():
        #     if word in model:
        #         temp.extend(model[word])
        #     else:
        #         temp.extend([0.0] * 32)
        # negative_examples.append(temp)
    # Split by words
    x_text = np.array(positive_examples + negative_examples)
    # x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    model = word2vec.Word2Vec.load('model.txt')
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            temp = data[shuffle_indices]
        else:
            temp = data
        shuffled_data = []
        for line in temp:
            total = []
            for word in line:
                total.extend(model[word])
            total.extend([0.0]*(512*32-len(total)))
            shuffled_data.append(total)
        shuffled_data = np.array(shuffled_data)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
