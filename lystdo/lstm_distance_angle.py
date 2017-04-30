'''Train a Bidirectional LSTM on the IMDB sentiment classification task.

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Input, Bidirectional, Merge, recurrent, RepeatVector
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta, Adamax, Nadam
from keras.callbacks import ModelCheckpoint
import os, sys, re, random
from data_loader import *

# config
with open('../../data/config.json') as fp:
    CONFIG = json.load(fp)

def model(max_features, maxlen):
    model_q1 = Sequential()
    # model_q1.add(Embedding(max_features, 128, input_length=maxlen))
    model_q1.add(Bidirectional(LSTM(64), input_shape=(maxlen, 128)))
    # model_q1.add(Bidirectional(LSTM(64)))
    model_q1.add(Dropout(0.5))
    model_q1.add(RepeatVector(maxlen))

    model_q2 = Sequential()
    # model_q2.add(Embedding(max_features, 128, input_length=maxlen))
    model_q2.add(Bidirectional(LSTM(64), input_shape=(maxlen, 128)))
    # model_q2.add(Bidirectional(LSTM(64)))
    model_q2.add(Dropout(0.5))
    model_q2.add(RepeatVector(maxlen))

    distance = Sequential()
    distance.add(Merge([model_q1, model_q2], mode='sum'))
    angle = Sequential()
    angle.add(Merge([model_q1, model_q2], mode='dot'))
    
    model = Sequential()
    model.add(Merge([distance, angle], mode='concat'))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.5))
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    # rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    # sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    # adagrad = Adagrad(lr=0.01, epsilon=1e-06)
    # adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    vocab, word_idx, idx_word, vocab_size, maxlen = CONFIG['vocab'], CONFIG['word_idx'], CONFIG['idx_word'], CONFIG['vocab_size'], CONFIG['maxlen']
    # print('type vocab', type(vocab))
    # print('type word_idx', type(word_idx))
    # print('type idx_word', type(idx_word))
    print('vocab_size', vocab_size)
    print('maxlen', maxlen)
    maxlen=50
    batch_size = 48
    print('Loading data...')
    data = read_data(sys.argv[1])
    # data2 = read_data(sys.argv[2]) 
    print('Train...')
    model = model(vocab_size, maxlen)
    print(model)
    for layer in model.layers:
        print(layer.output_shape)
    data = data[:400000]
    for i in xrange(5):
        # random.shuffle(data)
        if os.path.exists('./models/model_birnn_distance_angle.hdf5'):
            model.load_weights('./models/model_birnn_distance_angle.hdf5')
        checkpointer = ModelCheckpoint(filepath='./models/model_birnn_distance_angle.hdf5', verbose=1)
        # train_data = data + data2
        # X_q1, X_q2, Y = vectorize(data, word_idx, maxlen)
        # random.shuffle(data)
        model.fit_generator(vectorize_generator(data, word_idx, idx_word, maxlen, batch_size=batch_size),
                            samples_per_epoch=400000,
                            nb_epoch=4,
                            show_accuracy=True,
                            callbacks=[checkpointer])
                            # validation_data=(vectorize_generator(data[400:600], word_idx, idx_word, maxlen, batch_size=batch_size)))
        # model.evalute_generator(vectorize_generator(random.shuffle(data[:400000]), word_idx, idx_word, maxlen, batch_size=batch_size))
        # model.fit([X_q1, X_q2], Y, 
        #           batch_size=batch_size,
        #           nb_epoch=4,
        #           show_accuracy=True,
        #           callbacks=[checkpointer],
        #           validation_split=0.0)
    # X_q1_test, X_q2_test, Y = vectorize(data2, word_idx, maxlen)
    # result = model.predict_proba([X_q1[404290:], X_q2[404290:]], batch_size=32)
    # print(result)
    # print(len(result))
    # print(np.shape(result))
    # with open('result.csv', 'a') as fw:
    #     for i in result:
    #         fw.write(str(i[0])+'\n')
