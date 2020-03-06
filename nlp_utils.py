import math
import os
import pickle
import string

# Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from nltk import FreqDist

nlp = spacy.load("en_core_web_sm")
from spacy.lang.en.stop_words import STOP_WORDS

punkt_set = set(string.punctuation)


def plot_graph_loss(file_name, model_name):
    values = pd.read_table(file_name, sep=',')
    data = pd.DataFrame()
    data['epoch'] = list(values['epoch'].get_values() + 1) + \
                    list(values['epoch'].get_values() + 1)
    data['loss name'] = ['training'] * len(values) + \
                        ['validation'] * len(values)
    data['loss'] = list(values['loss'].get_values()) + \
                   list(values['val_loss'].get_values())
    sns.set(style='darkgrid', context='poster', font='Verdana')
    f, ax = plt.subplots()
    sns.lineplot(x='epoch', y='loss', hue='loss name', style='loss name',
                 dashes=False, data=data, palette='Set2')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend().texts[0].set_text('')
    ax.set_ylim([0, None])

    x = data['epoch']
    x_int = list(range(math.floor(min(x)), math.ceil(max(x)) + 1))
    plt.xticks(x_int)

    plt.title(model_name)
    plt.show()


def load_embeddings(file_name, vocabulary=None):
    """
    Loads word embeddings from the file with the given name.
    :param file_name: name of the file containing word embeddings
    :type file_name: str
    :param vocabulary: captions vocabulary
    :type vocabulary: numpy.array
    :return: word embeddings
    :rtype: dict
    """
    embeddings = dict()
    with open(file_name, 'r', encoding='utf-8') as doc:
        line = doc.readline()
        while line != '':
            line = line.rstrip('\n').lower()
            parts = line.split(' ')
            vals = np.array(parts[1:], dtype=np.float)
            if vocabulary is None or parts[0] in vocabulary:
                embeddings[parts[0]] = vals
            line = doc.readline()
        return embeddings


def load_embedding_weights(vocabulary, embedding_size):
    """
    Creates and loads embedding weights.
    :param vocabulary: captions vocabulary
    :type vocabulary: numpy.array
    :param embedding_size: embedding size
    :type embedding_size: int
    :return: embedding weights
    :rtype: numpy.array
    """
    if False and os.path.exists('data/embedding_matrix.pkl'):
        with open('data/embedding_matrix.pkl', 'rb') as f:
            embedding_matrix = pickle.load(f)
    else:
        print('Creating embedding weights...')
        embeddings = load_embeddings(f'data/glove.6B.{embedding_size}d.txt', vocabulary)
        embedding_matrix = np.zeros((len(vocabulary), embedding_size))
        for i in range(len(vocabulary)):
            if vocabulary[i] in embeddings.keys():
                embedding_matrix[i] = embeddings[vocabulary[i]]
            else:
                embedding_matrix[i] = np.random.standard_normal(embedding_size)
        with open('data/embedding_matrix.pkl', 'wb') as f:
            pickle.dump(embedding_matrix, f)
    return embedding_matrix


def contains_digit_iter_set(s, digits=set('0123456789')):
    for c in s:
        if c in digits:
            return True
    return False


def remove_digit_tokens(freq_dist):
    keys = list(freq_dist.keys())
    for key in keys:
        if contains_digit_iter_set(key):
            del freq_dist[key]


def preprocess_tokens(tokens: list, min_token_len: int = 2):
    return set([t.lower() for t in tokens if len(t) > min_token_len]).difference(punkt_set).difference(STOP_WORDS)


def generate_vocab(tokens: list, min_token_len: int = 2, threshold: int = 2, remove_numbers=True):
    freq_dist = FreqDist(tokens)
    if remove_numbers:
        remove_digit_tokens(freq_dist)
    tokens = preprocess_tokens(tokens=list(freq_dist.keys()), min_token_len=min_token_len)
    removed_tokens = set(freq_dist.keys()).difference(tokens)
    for t in removed_tokens:
        freq_dist.pop(t, None)
    [freq_dist.pop(t, None) for t in tokens if freq_dist[t] < threshold]
    return freq_dist


import gc
import definitions as defs


def load_embedding_matrix(tokenizer,
                          EMBEDDING_FILE=f'{defs.EMBEDDINGS_DIR}/glove.6B.50d.txt',
                          embed_size=50,
                          vocab_size_plus_one=True):
    embeddings_index = dict()
    # Transfer the embedding weights into a dictionary by iterating through every line of the file.
    f = open(EMBEDDING_FILE, encoding='utf8')
    for line in f:
        # split up line into an indexed array
        values = line.split()
        # first index is word
        word = values[0]
        # store the rest of the values in the array as a new array
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs  # 50 dimensions
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    gc.collect()
    # We get the mean and standard deviation of the embedding weights so that we could maintain the
    # same statistics for the rest of our own random generated weights.
    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    nb_words = len(tokenizer.word_index)
    if vocab_size_plus_one:
        nb_words += 1
    # We are going to set the embedding size to the pretrained dimension as we are replicating it.
    # the size will be Number of Words in Vocab X Embedding Size
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    gc.collect()

    # With the newly created embedding matrix, we'll fill it up with the words that we have in both
    # our own dictionary and loaded pre-trained embedding.
    embedded_count = 0
    for word, i in tokenizer.word_index.items():
        # tokenizer reserves the 0 index for padding, thus i -= 1
        i -= 1
        # then we see if this word is in glove's dictionary, if yes, get the corresponding weights
        embedding_vector = embeddings_index.get(word)
        # and store inside the embedding matrix that we will train later on.
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            embedded_count += 1
    print('total embedded:', embedded_count, 'common words')

    del embeddings_index
    gc.collect()

    # finally, return the embedding matrix
    return embedding_matrix
