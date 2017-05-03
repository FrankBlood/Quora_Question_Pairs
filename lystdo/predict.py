# -*- coding:utf8 -*-

'''
Single model may achieve LB scores at around 0.29+ ~ 0.30+
Average ensembles can easily get 0.28+ or less
Don't need to be an expert of feature engineering
All you need is a GPU!!!!!!!

The code is tested on Keras 2.0.0 using Tensorflow backend, and Python 2.7
'''

########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model import *

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

########################################
## set directories and parameters
########################################
BASE_DIR = '../data/'
EMBEDDING_FILE = BASE_DIR + 'vectors.bin'
# EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = BASE_DIR + 'train_lower_stemmer.csv'
TEST_DATA_FILE = BASE_DIR + 'test_lower_stemmer.csv'
# TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
# TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

act = 'relu'
# re_weight = False # whether to re-weight classes to fit the 17.5% share in test set
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

########################################
## index word vectors
########################################
print('Indexing word vectors')
# word2vec = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

########################################
## process texts in datasets
########################################
print('Processing text dataset')

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

texts_1 = [] 
texts_2 = []
labels = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_wordlist(values[3], False, False))
        texts_2.append(text_to_wordlist(values[4], False, False))
        labels.append(int(values[5]))
print('Found %s texts in train.csv' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
test_ids = []
with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(text_to_wordlist(values[1], False, False))
        test_texts_2.append(text_to_wordlist(values[2], False, False))
        test_ids.append(values[0])
print('Found %s texts in test.csv' % len(test_texts_1))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_ids)

########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## sample train/validation data
########################################
#np.random.seed(1234)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344

########################################
## add class weight
########################################
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

########################################
## define the model structure
########################################
model_num = 0
for (num_lstm, num_dense, rate_drop_lstm, rate_drop_dense) in [(275, 300, 0.3, 0.3)]:
    STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)
    model_num += 1
    # model = lystdo(nb_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, num_lstm, num_dense, rate_drop_lstm, rate_drop_dense, act)
    # model_name = 'lystdo'
    # model = lstm_add(nb_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, num_lstm, num_dense, rate_drop_lstm, rate_drop_dense, act)
    # model_name = 'lstm_add'
    # model = lstm_multiply(nb_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, num_lstm, num_dense, rate_drop_lstm, rate_drop_dense, act)
    # model_name = 'lstm_multiply'
    # model = bilstm_concat(nb_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, num_lstm, num_dense, rate_drop_lstm, rate_drop_dense, act)
    # model_name = 'bilstm_concat'

    # model = gru_concat(nb_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, num_lstm, num_dense, rate_drop_lstm, rate_drop_dense, act)
    # model_name = 'gru_concat'
    # model = gru_add(nb_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, num_lstm, num_dense, rate_drop_lstm, rate_drop_dense, act)
    # model_name = 'gru_add'
    # model = gru_add_multiply(nb_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, num_lstm, num_dense, rate_drop_lstm, rate_drop_dense, act)
    # model_name = 'gru_add_multiply'
    # model = gru_multiply(nb_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, num_lstm, num_dense, rate_drop_lstm, rate_drop_dense, act)
    # model_name = 'gru_multiply'
    # model = bigru_concat(nb_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, num_lstm, num_dense, rate_drop_lstm, rate_drop_dense, act)
    # model_name = 'bigru_concat'
    # model = bigru_multiply(nb_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, num_lstm, num_dense, rate_drop_lstm, rate_drop_dense, act)
    # model_name = 'bigru_multiply'
    model = bigru_add_multiply(nb_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, num_lstm, num_dense, rate_drop_lstm, rate_drop_dense, act)
    model_name = 'bigru_add_multiply'

    # model = bilstm_distance_angle(nb_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, num_lstm, num_dense, rate_drop_lstm, rate_drop_dense, act)
    # model_name = 'bilstm_distance_angle'

    model.summary()
    print(STAMP)

    bst_model_path = './models/' + model_name + STAMP + '.h5'

    model.load_weights(bst_model_path)

    preds = model.predict([test_data_1, test_data_2], batch_size=512, verbose=1)
    preds += model.predict([test_data_2, test_data_1], batch_size=512, verbose=1)

preds /= model_num * 2

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
submission.to_csv('./results/' + model_name + str(model_num) + STAMP + '.csv', index=False)
