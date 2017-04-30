# -*- coding:utf8 -*-

'''
The code is tested on Keras 2.0.0 using Tensorflow backend, and Python 2.7
'''

########################################
## import packages
########################################
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Input, Bidirectional, Merge, recurrent, RepeatVector, Activation
from keras.layers.merge import concatenate, add, dot, multiply
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta, Adamax, Nadam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

########################################
## lystdo define the model structure
########################################
def lystdo(nb_words, EMBEDDING_DIM, \
           embedding_matrix, MAX_SEQUENCE_LENGTH, \
           num_lstm, num_dense, rate_drop_lstm, \
           rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
    model.summary()
    # print(STAMP)
    return model

########################################
## lstm add
########################################
def lstm_add(nb_words, EMBEDDING_DIM, \
           embedding_matrix, MAX_SEQUENCE_LENGTH, \
           num_lstm, num_dense, rate_drop_lstm, \
           rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = add([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
    model.summary()
    # print(STAMP)
    return model

########################################
## lstm multiply
########################################
def lstm_multiply(nb_words, EMBEDDING_DIM, \
           embedding_matrix, MAX_SEQUENCE_LENGTH, \
           num_lstm, num_dense, rate_drop_lstm, \
           rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = multiply([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
    model.summary()
    # print(STAMP)
    return model

########################################
## bilstm concat
########################################
def bilstm_concat(nb_words, EMBEDDING_DIM, \
           embedding_matrix, MAX_SEQUENCE_LENGTH, \
           num_lstm, num_dense, rate_drop_lstm, \
           rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = Bidirectional(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
    model.summary()
    # print(STAMP)
    return model

########################################
## GRU concat
########################################
def gru_concat(nb_words, EMBEDDING_DIM, \
           embedding_matrix, MAX_SEQUENCE_LENGTH, \
           num_lstm, num_dense, rate_drop_lstm, \
           rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
    model.summary()
    # print(STAMP)
    return model

########################################
## GRU add
########################################
def gru_add(nb_words, EMBEDDING_DIM, \
           embedding_matrix, MAX_SEQUENCE_LENGTH, \
           num_lstm, num_dense, rate_drop_lstm, \
           rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = add([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
    model.summary()
    # print(STAMP)
    return model

########################################
## GRU multiply
########################################
def gru_multiply(nb_words, EMBEDDING_DIM, \
           embedding_matrix, MAX_SEQUENCE_LENGTH, \
           num_lstm, num_dense, rate_drop_lstm, \
           rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = multiply([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
    model.summary()
    # print(STAMP)
    return model

########################################
## bigru concat
########################################
def bigru_concat(nb_words, EMBEDDING_DIM, \
           embedding_matrix, MAX_SEQUENCE_LENGTH, \
           num_lstm, num_dense, rate_drop_lstm, \
           rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = Bidirectional(GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
    model.summary()
    # print(STAMP)
    return model

########################################
## bigru multiply
########################################
def bigru_multiply(nb_words, EMBEDDING_DIM, \
           embedding_matrix, MAX_SEQUENCE_LENGTH, \
           num_lstm, num_dense, rate_drop_lstm, \
           rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = Bidirectional(GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = multiply([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
    model.summary()
    # print(STAMP)
    return model

########################################
## bigru multiply no dense
########################################
def bigru_multiply_no_dense(nb_words, EMBEDDING_DIM, \
           embedding_matrix, MAX_SEQUENCE_LENGTH, \
           num_lstm, num_dense, rate_drop_lstm, \
           rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = Bidirectional(GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = multiply([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    # merged = Dense(num_dense, activation=act)(merged)
    # merged = Dropout(rate_drop_dense)(merged)
    # merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
    model.summary()
    # print(STAMP)
    return model

########################################
## bilstm distance angle
########################################
def bilstm_distance_angle(nb_words, EMBEDDING_DIM, \
           embedding_matrix, MAX_SEQUENCE_LENGTH, \
           num_lstm, num_dense, rate_drop_lstm, \
           rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    model_q1 = lstm_layer(embedded_sequences_1)
    # model_q1 = RepeatVector(MAX_SEQUENCE_LENGTH)(model_q1)
    # print(model_q1.output_shape)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    model_q2 = lstm_layer(embedded_sequences_2)
    # model_q2 = RepeatVector(MAX_SEQUENCE_LENGTH)(model_q2)
    # print(model_q2.output_shape)

    # distance = Merge([model_q1, model_q2], mode='add')
    distance = add([model_q1, model_q2])
    # distance = Dropout(rate_drop_dense)(distance)
    # distance = BatchNormalization()(distance)
    # print(distance.output_shape)

    # angle = Merge([model_q1, model_q2], mode='dot')
    angle = multiply([model_q1, model_q2])
    # angle = Dropout(rate_drop_dense)(angle)
    # angle = BatchNormalization()(angle)
    # print(angle.output_shape)

    # merged = Merge([distance, angle], mode='concat')
    merged = concatenate([distance, angle])
    merged = Dropout(rate_drop_dense)(merged)
    # merged = BatchNormalization()(merged)

    # merged = Dense(num_dense, activation=act)(merged)
    # merged = Dropout(rate_drop_dense)(merged)
    # merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)

    # try using different optimizers and different optimizer configs
    # rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    # sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    # adagrad = Adagrad(lr=0.01, epsilon=1e-06)
    # adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    # model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy',
    #           optimizer='nadam',
    #           metrics=['acc'])
    model.summary()
    # print(STAMP)
    return model

########################################
## bilstm distance angle
########################################
def bilstm_distance_angle_(nb_words, EMBEDDING_DIM, \
                          embedding_matrix, MAX_SEQUENCE_LENGTH, \
                          num_lstm, num_dense, rate_drop_lstm, \
                          rate_drop_dense, act):
    model_q1 = Sequential()
    model_q1.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
    model_q1.add(Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    model_q1.add(Bidirectional(LSTM(num_lstm)))
    # model_q1.add(Bidirectional(LSTM(64)))
    model_q1.add(Dropout(rate_drop_lstm))
    model_q1.add(RepeatVector(num_lstm))

    model_q2 = Sequential()
    model_q2.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
    model_q2.add(Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    model_q2.add(Bidirectional(LSTM(num_lstm)))
    # model_q2.add(Bidirectional(LSTM(64)))
    model_q2.add(Dropout(rate_drop_lstm))
    model_q2.add(RepeatVector(num_lstm))

    distance = Sequential()
    distance.add(Merge([model_q1, model_q2], mode='sum'))
    angle = Sequential()
    angle.add(Merge([model_q1, model_q2], mode='dot'))

    model = Sequential()
    model.add(Merge([distance, angle], mode='concat'))
    model.add(Bidirectional(LSTM(num_lstm)))
    model.add(Dropout(rate_drop_lstm))
    model.add(Dense(num_dense, activation=act))
    model.add(Dropout(rate_drop_dense))
    model.add(BatchNormalization())
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
    model.summary()
    print(STAMP)
    return model

# reference
# def model(max_features, maxlen):
#     model_q1 = Sequential()
#     # model_q1.add(Embedding(max_features, 128, input_length=maxlen))
#     model_q1.add(Bidirectional(LSTM(64), input_shape=(maxlen, 128)))
#     # model_q1.add(Bidirectional(LSTM(64)))
#     model_q1.add(Dropout(0.5))
#     model_q1.add(RepeatVector(maxlen))

#     model_q2 = Sequential()
#     # model_q2.add(Embedding(max_features, 128, input_length=maxlen))
#     model_q2.add(Bidirectional(LSTM(64), input_shape=(maxlen, 128)))
#     # model_q2.add(Bidirectional(LSTM(64)))
#     model_q2.add(Dropout(0.5))
#     model_q2.add(RepeatVector(maxlen))

#     distance = Sequential()
#     distance.add(Merge([model_q1, model_q2], mode='sum'))
#     angle = Sequential()
#     angle.add(Merge([model_q1, model_q2], mode='dot'))
    
#     model = Sequential()
#     model.add(Merge([distance, angle], mode='concat'))
#     model.add(Bidirectional(LSTM(128)))
#     model.add(Dropout(0.5))
#     # model.add(Dense(128, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))

#     # try using different optimizers and different optimizer configs
#     # rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
#     # sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#     # adagrad = Adagrad(lr=0.01, epsilon=1e-06)
#     # adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
#     # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#     # adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#     # nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
#     model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
#     return model
